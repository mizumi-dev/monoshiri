"""
モノシリ RAGパイプライン
ベクトルDB検索 → コンテキスト構築 → ローカルLLM回答生成
透明性原則に基づき全回答にエビデンスを付与する。
"""
from __future__ import annotations
import logging

from core.config import TOP_K, MAX_EVIDENCE, RAG_CONTEXT_MAX_LENGTH
from core.chromadb_store import get_collection
from core.embedder import embed_query
from core.llm import generate_answer

logger = logging.getLogger(__name__)


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

    return "\n\n---\n\n".join(context_parts)


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
        # 閾値以下でも最近傍を取得（「関連情報」として提示）
        fallback_items = search_similar(
            query=question,
            top_k=5,
            similarity_threshold=0.0,  # 閾値なし
        )

        if not fallback_items:
            return {
                "answer": (
                    "関連する文書が見つかりませんでした。\n\n"
                    "インデックス管理画面からフォルダをインデックス化してから質問してください。"
                ),
                "evidence": [],
                "found": False,
            }

        # フォールバック: 最近傍で回答
        context = build_context(fallback_items)
        evidence = extract_evidence(fallback_items)
        prefix = "直接の回答は見つかりませんでしたが、関連する情報としてこちらがあります。\n\n"
        items_for_answer = fallback_items
    else:
        context = build_context(similar_items)
        evidence = extract_evidence(similar_items)
        prefix = ""
        items_for_answer = similar_items

    # 2. LLMで回答生成
    try:
        answer = generate_answer(
            model_name=model_name,
            question=question,
            context=context,
        )
        return {
            "answer": prefix + answer,
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
