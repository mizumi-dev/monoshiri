"""
モノシリ RAGパイプライン
ベクトルDB検索 → コンテキスト構築 → ローカルLLM回答生成
透明性原則に基づき全回答にエビデンスを付与する。

TODO(仕様書更新): v3.4仕様書は「該当なし時のフォールバック回答」を記載しているが、
  ADR-002に基づきフォールバック廃止済み。関連文書なしの場合は正直に
  「見つかりません」と返す設計。仕様書のエラーハンドリングセクションを更新すること。
"""
from __future__ import annotations
import logging

from core.config import TOP_K, MAX_EVIDENCE, RAG_CONTEXT_MAX_LENGTH
from core.chromadb_store import get_collection
from core.embedder import embed_query
from core.llm import generate_answer, stream_answer

logger = logging.getLogger(__name__)


# ─── 不十分回答判定パターン ───────────────────────────────────
# LLM回答に以下のパターンが含まれる場合、「資料から十分に回答できていない」
# と判定し、回答末尾に外部資料調査の案内を付加する。
INSUFFICIENT_ANSWER_PATTERNS = [
    "推測", "推察",
    "示されていません", "示されておりません",
    "特定できません", "特定することはできません",
    "記述はありません", "記載はありません", "記載されていません",
    "情報はありません", "情報が含まれていません", "情報が不足",
    "明確ではありません", "明示されていません",
    "答えられません", "回答できません",
    "見つかりません", "見つかりませんでした",
]

EXTERNAL_NOTICE = (
    "\n\n───\n"
    "※ この回答は、読み込んでいる社内資料を元にしています。"
    "資料内に十分な情報が含まれていない可能性があります。"
    "必要に応じて、外部資料もご確認ください。"
)


def _is_insufficient_answer(answer: str) -> bool:
    """LLM回答が「資料から十分に回答できていない」かを判定する"""
    return any(p in answer for p in INSUFFICIENT_ANSWER_PATTERNS)


# ─── ベクトル検索 ─────────────────────────────────────────────

def search_similar(
    query: str,
    top_k: int = TOP_K,
    similarity_threshold: float = 0.5,
) -> list[dict]:
    """
    クエリに類似したドキュメントチャンクを検索する。

    Args:
        query: ユーザーの質問文
        top_k: 検索上限件数
        similarity_threshold: 類似度閾値（0〜1、高いほど絞り込み）

    Returns:
        [{"text": str, "metadata": dict, "similarity": float}]
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    # クエリをEmbeddingに変換
    query_embedding = embed_query(query)

    actual_top_k = min(top_k, collection.count())

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=actual_top_k,
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"][0]:
        return []

    items = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance: 0=完全一致, 2=完全に異なる
        # コサイン類似度 = 1 - distance（正規化Embeddingの場合）
        similarity = max(0.0, 1.0 - dist)

        if similarity >= similarity_threshold:
            items.append({
                "text": doc,
                "metadata": meta,
                "similarity": similarity,
            })

    return items


# ─── コンテキスト構築 ─────────────────────────────────────────

def build_context(
    similar_items: list[dict],
    max_length: int = RAG_CONTEXT_MAX_LENGTH,
) -> str:
    """
    検索結果からLLMに渡すコンテキストを構築する。
    max_length文字を超えたらそこで打ち切る。
    """
    context_parts = []
    total_len = 0

    for item in similar_items:
        meta = item["metadata"]
        file_name = meta.get("file_name", "不明")
        page = meta.get("page")
        slide = meta.get("slide")

        # ファイル出典の表示
        location = f"【{file_name}"
        if page:
            location += f" p.{page}"
        elif slide:
            location += f" スライド{slide}"
        location += "】"

        part = f"{location}\n{item['text']}"

        if total_len + len(part) > max_length and context_parts:
            break  # 1件もあればそこで打ち切る

        context_parts.append(part)
        total_len += len(part)

    # BUG-018修正: "---" はMarkdown水平線としてモデルに誤解釈されやすい。
    #   明示的な区切り文字列に変更してチャンク境界を明確化する。
    return "\n\n".join(context_parts)


# ─── エビデンス抽出 ───────────────────────────────────────────

def extract_evidence(
    similar_items: list[dict],
    max_items: int = MAX_EVIDENCE,
) -> list[dict]:
    """
    回答の根拠となるエビデンス（出典情報）を抽出する。
    同一ファイルの同一ページ/スライドは重複排除する。

    Returns:
        [{
            "file_name": str,
            "file_path": str,
            "folder_path": str,
            "page": int|None,
            "slide": int|None,
            "similarity": float,
        }]
    """
    seen: set[tuple] = set()
    evidence: list[dict] = []

    for item in similar_items:
        meta = item["metadata"]
        file_path = meta.get("file_path", "")
        page = meta.get("page")
        slide = meta.get("slide")

        key = (file_path, page, slide)
        if key in seen:
            continue
        seen.add(key)

        evidence.append({
            "file_name": meta.get("file_name", "不明"),
            "file_path": file_path,
            "folder_path": meta.get("folder_path", ""),
            "page": page,
            "slide": slide,
            "similarity": round(item["similarity"], 3),
        })

        if len(evidence) >= max_items:
            break

    return evidence


# ─── メイン回答生成 ───────────────────────────────────────────

def answer_question(
    question: str,
    model_name: str,
    similarity_threshold: float = 0.5,
    top_k: int = TOP_K,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    質問に対してRAGパイプラインで回答を生成する。

    Args:
        question: ユーザーの質問
        model_name: 使用するローカルLLM名
        similarity_threshold: 類似度閾値
        top_k: 検索上限件数

    Returns:
        {
            "answer": str,
            "evidence": list[dict],
            "found": bool,  # 閾値以上の文書が見つかったか
        }
    """
    # 1. ベクトル検索（閾値あり）
    similar_items = search_similar(
        query=question,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    found = len(similar_items) > 0

    if not found:
        # ADR-002修正: フォールバック廃止。関連文書なし → 正直に返す。
        return {
            "answer": (
                "ご質問に関連する文書が見つかりませんでした。\n\n"
                "以下をご確認ください：\n"
                "・インデックス管理画面で対象フォルダがインデックス済みか確認する\n"
                "・別のキーワードで質問してみる"
            ),
            "evidence": [],
            "found": False,
        }

    context = build_context(similar_items)
    evidence = extract_evidence(similar_items)
    prefix = ""

    # 2. LLMで回答生成
    try:
        answer = generate_answer(
            model_name=model_name,
            question=question,
            context=context,
            chat_history=chat_history,
        )
        # 回答が「不十分」と判定された場合は外部資料調査の案内を付加
        if _is_insufficient_answer(answer):
            answer = answer.rstrip() + EXTERNAL_NOTICE
        return {
            "answer": answer,
            "evidence": evidence,
            "found": found,
        }

    except FileNotFoundError as e:
        return {
            "answer": f"モデルエラー: {e}",
            "evidence": evidence,
            "found": found,
        }
    except Exception as e:
        logger.error(f"回答生成エラー: {e}")
        return {
            "answer": (
                f"回答の生成中にエラーが発生しました。\n\n"
                f"詳細: {str(e)[:300]}"
            ),
            "evidence": evidence,
            "found": found,
        }


def stream_question(
    question: str,
    model_name: str,
    similarity_threshold: float = 0.5,
    top_k: int = TOP_K,
    chat_history: list[dict] | None = None,
):
    """
    RAGパイプラインのストリーミング版。

    Returns:
        Generator that yields:
          - dict {"type": "evidence", "data": evidence_list}  最初に1回
          - dict {"type": "token",    "data": str}            トークンごとに
          - dict {"type": "error",    "data": str}            エラー時
    """
    # 1. ベクトル検索
    similar_items = search_similar(
        query=question,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )
    found = len(similar_items) > 0

    if not found:
        # ADR-002修正: フォールバック廃止。関連文書なし → 正直に返す。
        yield {"type": "evidence", "data": [], "found": False}
        yield {"type": "token", "data": (
            "ご質問に関連する文書が見つかりませんでした。\n\n"
            "以下をご確認ください：\n"
            "・インデックス管理画面で対象フォルダがインデックス済みか確認する\n"
            "・別のキーワードで質問してみる"
        )}
        return

    context = build_context(similar_items)
    evidence = extract_evidence(similar_items)

    # 2. エビデンスを先に返す（UIで即座に表示するため）
    yield {"type": "evidence", "data": evidence, "found": found}

    # 3. ストリーミング推論
    try:
        accumulated = []
        for token in stream_answer(
            model_name=model_name,
            question=question,
            context=context,
            chat_history=chat_history,
        ):
            accumulated.append(token)
            yield {"type": "token", "data": token}
        # 回答完了後、「不十分」判定で外部資料案内を追加トークンとして出力
        final_answer = "".join(accumulated)
        if _is_insufficient_answer(final_answer):
            yield {"type": "token", "data": EXTERNAL_NOTICE}
    except Exception as e:
        import traceback as _tb
        tb_str = _tb.format_exc()
        logger.error(f"ストリーミング回答生成エラー（詳細）:\n{tb_str}")
        yield {"type": "error", "data": str(e)}
