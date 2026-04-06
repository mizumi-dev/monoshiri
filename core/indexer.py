"""
モノシリ インデックス作成モジュール
ファイル抽出 → チャンク分割 → Embedding → ChromaDB格納のパイプラインを担当する。
差分検知により変更ファイルのみ再処理する。
"""
from __future__ import annotations
import json
import logging
import uuid
from pathlib import Path

from core.config import (
    CHROMA_DIR, CHROMA_COLLECTION,
    CHUNK_SIZE, CHUNK_OVERLAP, SKIP_LOG_FILE,
)
from core.extractor import extract_text, scan_folder
from core.hash_manager import (
    compute_hash, get_diff, load_hashes, save_hashes,
    make_folder_id, delete_folder_hashes,
)
from core.embedder import embed_texts

logger = logging.getLogger(__name__)

# ChromaDBバッチサイズ（一度にEmbeddingする件数）
BATCH_SIZE = 50


# ─── ChromaDB ──────────────────────────────────────────────

def get_chroma_client():
    """ChromaDBクライアントを取得する"""
    import chromadb
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    """ChromaDBコレクションを取得または作成する"""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def get_index_stats() -> dict:
    """インデックスの統計情報を返す"""
    try:
        collection = get_collection()
        return {"total_chunks": collection.count()}
    except Exception:
        return {"total_chunks": 0}


# ─── テキスト分割 ────────────────────────────────────────────

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    テキストをチャンクに分割する。
    LangChainのRecursiveCharacterTextSplitterを使用。
    日本語の区切り文字を考慮した分割を行う。
    """
    if not text or not text.strip():
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            # 日本語対応の区切り文字
            separators=["\n\n", "\n", "。", "．", ".", "、", "，", "　", " ", ""],
        )
        chunks = splitter.split_text(text)
        return [c for c in chunks if c.strip()]
    except ImportError:
        # フォールバック: シンプルな文字数ベース分割
        return _simple_split(text, chunk_size, overlap)


def _simple_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """LangChainが使えない場合のフォールバック分割"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # 自然な区切り点を探す
            for sep in ["。\n", "\n\n", "。", "\n"]:
                pos = text.rfind(sep, start + overlap, end)
                if pos > start + overlap:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, start + 1)
    return chunks


# ─── スキップログ管理 ─────────────────────────────────────────

def load_skip_log() -> list[dict]:
    """スキップログを読み込む"""
    if SKIP_LOG_FILE.exists():
        try:
            with open(SKIP_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_skip_log(skip_log: list[dict]) -> None:
    """スキップログを保存する"""
    with open(SKIP_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(skip_log, f, ensure_ascii=False, indent=2)


# ─── 差分スキャン ────────────────────────────────────────────

def scan_and_diff(
    folder_paths: list[Path],
) -> tuple[list[tuple[Path, str]], list[str]]:
    """
    フォルダをスキャンして差分を検出する。

    Returns:
        (files_to_process, deleted_chroma_paths)
        - files_to_process: [(file_path, folder_id), ...] 処理すべきファイルとそのフォルダID
        - deleted_chroma_paths: ChromaDBから削除すべきファイルパス文字列
    """
    files_to_process: list[tuple[Path, str]] = []
    deleted_chroma_paths: list[str] = []

    for folder_path in folder_paths:
        if not folder_path.exists():
            logger.warning(f"フォルダが存在しません: {folder_path}")
            continue

        folder_id = make_folder_id(folder_path)
        folder_files = scan_folder(folder_path)
        new_or_modified, unchanged, deleted_paths = get_diff(folder_id, folder_files)

        files_to_process.extend([(f, folder_id) for f in new_or_modified])
        deleted_chroma_paths.extend(deleted_paths)

    return files_to_process, deleted_chroma_paths


# ─── ChromaDBからの削除 ──────────────────────────────────────

def delete_from_chroma(file_paths: list[str]) -> None:
    """指定ファイルのチャンクをChromaDBから削除する"""
    if not file_paths:
        return
    collection = get_collection()
    for path in file_paths:
        try:
            results = collection.get(where={"file_path": path})
            if results["ids"]:
                collection.delete(ids=results["ids"])
        except Exception as e:
            logger.warning(f"ChromaDB削除エラー {path}: {e}")


def delete_folder_from_index(folder_path: Path) -> None:
    """フォルダをインデックスから完全削除する（UI用）"""
    try:
        collection = get_collection()
        results = collection.get(where={"folder_path": str(folder_path)})
        if results["ids"]:
            collection.delete(ids=results["ids"])
    except Exception as e:
        logger.error(f"フォルダ削除エラー: {e}")

    folder_id = make_folder_id(folder_path)
    delete_folder_hashes(folder_id)


# ─── バッチEmbedding & ChromaDB格納 ─────────────────────────

def flush_batch(
    collection,
    batch_texts: list[str],
    batch_ids: list[str],
    batch_metadatas: list[dict],
) -> None:
    """バッチをEmbeddingしてChromaDBに格納する"""
    if not batch_texts:
        return
    embeddings = embed_texts(batch_texts)
    collection.add(
        ids=batch_ids,
        embeddings=embeddings,
        documents=batch_texts,
        metadatas=batch_metadatas,
    )


# ─── メインインデックス処理 ───────────────────────────────────

def process_single_file(
    file_path: Path,
    folder_id: str,
    collection,
    batch_texts: list[str],
    batch_ids: list[str],
    batch_metadatas: list[dict],
    skip_log: list[dict],
) -> tuple[int, str | None]:
    """
    1ファイルを処理してバッチに追加する。

    Returns:
        (chunks_added, skip_reason)
    """
    # テキスト抽出
    page_chunks, skip_reason = extract_text(file_path)

    if skip_reason:
        skip_log.append({
            "file_path": str(file_path),
            "file_name": file_path.name,
            "reason": skip_reason,
        })
        return 0, skip_reason

    chunks_added = 0

    for page_chunk in page_chunks:
        text = page_chunk["text"]
        page = page_chunk.get("page")
        slide = page_chunk.get("slide")

        sub_chunks = split_text(text)
        for chunk_text in sub_chunks:
            if not chunk_text.strip():
                continue

            chunk_id = str(uuid.uuid4())
            metadata: dict = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "folder_path": str(file_path.parent),
            }
            if page is not None:
                metadata["page"] = page
            if slide is not None:
                metadata["slide"] = slide

            batch_texts.append(chunk_text)
            batch_ids.append(chunk_id)
            batch_metadatas.append(metadata)
            chunks_added += 1

    # ハッシュ更新
    file_hash = compute_hash(file_path)
    if file_hash:
        hashes = load_hashes(folder_id)
        hashes[str(file_path)] = file_hash
        save_hashes(folder_id, hashes)

    return chunks_added, None
