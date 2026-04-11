"""
モノシリ インデックス作成モジュール
ファイル抽出 → チャンク分割 → Embedding → ChromaDB格納のパイプラインを担当する。
差分検知により変更ファイルのみ再処理する。

アーキテクチャ:
- Producer-Consumerパターンで抽出とEmbeddingを並列化
- threading.Eventによるキャンセル機構
- IndexManagerシングルトンでUIとバックグラウンド処理を分離
"""
from __future__ import annotations
import json
import logging
import queue
import threading
import time
import uuid
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from core.config import (
    CHROMA_DIR, CHROMA_COLLECTION,
    CHUNK_SIZE, CHUNK_OVERLAP, SKIP_LOG_FILE,
)
from core.extractor import extract_text, scan_folder
from core.hash_manager import (
    compute_hash, get_diff, load_hashes, save_hashes,
    make_folder_id, delete_folder_hashes, clear_all_hashes,
)
from core.embedder import embed_texts

logger = logging.getLogger(__name__)

# ChromaDBバッチサイズ（一度にEmbeddingする件数）
BATCH_SIZE = 500

# ファイル抽出の並列数（I/Oバウンドなので多めに）
EXTRACT_WORKERS = 8

# Producerキューの最大サイズ（メモリ制限）
PRODUCER_QUEUE_SIZE = 30

# _SENTINEL: Producerがキュー終端を伝えるシグナル
_SENTINEL = None


# ─── ChromaDB ──────────────────────────────────────────────

# PERF修正: PersistentClient をモジュールレベルでシングルトン管理。
_chroma_client_instance = None


def get_chroma_client():
    """ChromaDBクライアントを取得する（シングルトン）"""
    global _chroma_client_instance
    if _chroma_client_instance is None:
        import chromadb
        _chroma_client_instance = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client_instance


def reset_chroma_client() -> None:
    """インデックス再作成後などクライアントをリセットする際に呼ぶ"""
    global _chroma_client_instance
    _chroma_client_instance = None


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
            separators=["\n\n", "\n", "。", "．", "、", "，", "　", " ", ""],
        )
        chunks = splitter.split_text(text)
        return [c for c in chunks if c.strip()]
    except ImportError:
        return _simple_split(text, chunk_size, overlap)


def _simple_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """LangChainが使えない場合のフォールバック分割"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
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
    force: bool = False,
) -> tuple[list[tuple[Path, str]], list[str]]:
    """
    フォルダをスキャンして差分を検出する。

    Args:
        folder_paths: 対象フォルダ一覧
        force: True の場合はSHA-256計算をスキップし全ファイルを対象にする
               （強制再インデックス時に使用、スキャン速度が大幅に向上）

    Returns:
        (files_to_process, deleted_chroma_paths)
    """
    files_to_process: list[tuple[Path, str]] = []
    deleted_chroma_paths: list[str] = []

    for folder_path in folder_paths:
        if not folder_path.exists():
            logger.warning(f"フォルダが存在しません: {folder_path}")
            continue

        folder_id = make_folder_id(folder_path)
        folder_files = scan_folder(folder_path)

        if force:
            # 強制再インデックス: ハッシュ計算をスキップして全ファイルを対象
            files_to_process.extend([(f, folder_id) for f in folder_files])
        else:
            new_or_modified, unchanged, deleted_paths = get_diff(folder_id, folder_files)
            files_to_process.extend([(f, folder_id) for f in new_or_modified])
            deleted_chroma_paths.extend(deleted_paths)

    return files_to_process, deleted_chroma_paths


# ─── ChromaDBからの削除 ──────────────────────────────────────

def delete_from_chroma(file_paths: list[str]) -> None:
    """指定ファイルのチャンクをChromaDBから削除する（$in演算子で一括削除）"""
    if not file_paths:
        return
    collection = get_collection()
    # $in演算子で一括フィルタ。50件ずつバッチ処理（ChromaDB制限対策）
    BATCH = 50
    for i in range(0, len(file_paths), BATCH):
        batch = file_paths[i: i + BATCH]
        try:
            where = (
                {"file_path": {"$in": batch}}
                if len(batch) > 1
                else {"file_path": batch[0]}
            )
            results = collection.get(where=where)
            if results["ids"]:
                collection.delete(ids=results["ids"])
        except Exception as e:
            logger.warning(f"ChromaDB一括削除エラー: {e}")


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


# ─── ファイル単体の抽出・チャンク化 ─────────────────────────────

def extract_file_chunks(
    file_path: Path,
    folder_id: str,
) -> tuple[list[dict], list[dict], str | None]:
    """
    1ファイルを抽出してチャンクリストを返す。スレッドセーフ。

    Returns:
        (chunks, hash_updates, skip_reason)
    """
    page_chunks, skip_reason = extract_text(file_path)

    if skip_reason:
        return [], [], skip_reason

    chunks: list[dict] = []
    for page_chunk in page_chunks:
        text = page_chunk["text"]
        page = page_chunk.get("page")
        slide = page_chunk.get("slide")

        for chunk_text in split_text(text):
            if not chunk_text.strip():
                continue
            metadata: dict = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "folder_path": str(file_path.parent),
            }
            if page is not None:
                metadata["page"] = page
            if slide is not None:
                metadata["slide"] = slide

            chunks.append({
                "text": chunk_text,
                "id": str(uuid.uuid4()),
                "metadata": metadata,
            })

    file_hash = compute_hash(file_path)
    hash_updates = [{"folder_id": folder_id, "file_path": str(file_path), "hash": file_hash}] if file_hash else []

    return chunks, hash_updates, None


# ─── メインインデックス処理（パイプライン並列化） ───────────────

def run_indexing_pipeline(
    files_to_process: list[tuple[Path, str]],
    collection,
    skip_log: list[dict],
    cancel_event: threading.Event,
    progress_callback=None,
) -> dict:
    """
    Producer-ConsumerパターンでEmbeddingとファイル抽出を並列化する。
    """
    total = len(files_to_process)
    if total == 0:
        return {"processed": 0, "skipped": 0, "cancelled": False}

    chunk_queue: queue.Queue = queue.Queue(maxsize=PRODUCER_QUEUE_SIZE)

    processed_files = 0
    skipped_count = 0
    done_count = 0
    hash_buffer: dict[str, dict[str, str]] = {}

    # ── Producer: ファイル抽出を並列で実行 ───────────────────────
    def producer():
        pool = ThreadPoolExecutor(max_workers=EXTRACT_WORKERS)
        try:
            futures = {
                pool.submit(extract_file_chunks, fp, fid): (fp, fid)
                for fp, fid in files_to_process
            }
            for future in as_completed(futures):
                fp, fid = futures[future]
                if cancel_event.is_set():
                    # 未開始のfutureをキャンセルしてスレッドプールを即時解放
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    result = future.result()
                    chunk_queue.put((fp, fid, result))
                except Exception as e:
                    logger.error(f"抽出エラー {fp}: {e}")
                    chunk_queue.put((fp, fid, ([], [], str(e))))
        finally:
            pool.shutdown(wait=False)
            chunk_queue.put(_SENTINEL)

    producer_thread = threading.Thread(target=producer, daemon=True, name="monoshiri-producer")
    producer_thread.start()

    # ── Consumer: バッチ溜めてEmbedding → ChromaDB ───────────────
    batch_texts: list[str] = []
    batch_ids: list[str] = []
    batch_metas: list[dict] = []

    def flush():
        nonlocal processed_files
        if not batch_texts:
            return
        def _cancel_check():
            return cancel_event.is_set()
        try:
            embeddings = embed_texts(batch_texts, cancel_check=_cancel_check)
            collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metas,
            )
            processed_files += len(batch_texts)
        except CancelledError:
            raise
        except Exception as e:
            logger.error(f"バッチEmbeddingエラー: {e}")
            raise
        finally:
            batch_texts.clear()
            batch_ids.clear()
            batch_metas.clear()

    def flush_hashes():
        for fid, path_hash_map in hash_buffer.items():
            hashes = load_hashes(fid)
            hashes.update(path_hash_map)
            save_hashes(fid, hashes)
        hash_buffer.clear()

    cancelled = False
    try:
        while True:
            if cancel_event.is_set():
                cancelled = True
                break

            try:
                item = chunk_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _SENTINEL:
                break

            fp, fid, (chunks, hash_updates, skip_reason) = item
            done_count += 1

            if skip_reason:
                skip_log.append({
                    "file_path": str(fp),
                    "file_name": fp.name,
                    "reason": skip_reason,
                })
                skipped_count += 1
            else:
                for c in chunks:
                    batch_texts.append(c["text"])
                    batch_ids.append(c["id"])
                    batch_metas.append(c["metadata"])

                for hu in hash_updates:
                    fid_key = hu["folder_id"]
                    if fid_key not in hash_buffer:
                        hash_buffer[fid_key] = {}
                    hash_buffer[fid_key][hu["file_path"]] = hu["hash"]

            if progress_callback:
                progress_callback(done_count, total, fp.name)

            if len(batch_texts) >= BATCH_SIZE:
                flush()
                flush_hashes()

        if not cancelled and batch_texts:
            flush()
        flush_hashes()

    except CancelledError:
        cancelled = True
        flush_hashes()
    except Exception as e:
        logger.error(f"Consumer例外: {e}")
        flush_hashes()
        raise
    finally:
        producer_thread.join(timeout=5)

    return {
        "processed": done_count - skipped_count,
        "skipped": skipped_count,
        "cancelled": cancelled,
    }


# ─── IndexManager（シングルトン）────────────────────────────────

class IndexManager:
    """
    シングルトン・インデックス処理マネージャー。
    バックグラウンドスレッドでインデックス処理を実行し、
    Streamlitのrerunをまたいで進捗状態を保持する。
    """

    _instance: Optional["IndexManager"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "IndexManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()

        self.status: str = "idle"
        self.phase_label: str = ""
        self.progress: float = 0.0
        self.done_count: int = 0
        self.total_count: int = 0
        self.current_file: str = ""
        self.start_time: float = 0.0
        self.elapsed: float = 0.0
        self.result: dict = {}
        self.error_msg: str = ""

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    @property
    def is_finished(self) -> bool:
        return self.status in ("done", "cancelled", "error")

    def start_force_reindex(self, folders: list[Path]) -> bool:
        """
        全ハッシュをクリアして全件再インデックスを開始する（IMPROVE-001用）。
        force=True でスキャン時のSHA-256計算をスキップするため高速。
        """
        if self.is_running:
            return False
        clear_all_hashes()
        logger.info("全件再インデックス: ハッシュキャッシュをクリアしました")
        return self.start(folders, force=True)

    def start(self, folders: list[Path], force: bool = False) -> bool:
        """
        インデックス処理をバックグラウンドで開始する。
        force=True の場合はスキャン時のSHA-256計算をスキップ。
        """
        if self.is_running:
            return False

        self._cancel_event.clear()
        self._update(
            status="scanning",
            phase_label="📂 フォルダをスキャン中...",
            progress=0.0,
            done_count=0,
            total_count=0,
            current_file="",
            result={},
            error_msg="",
            start_time=time.time(),
            elapsed=0.0,
        )

        self._worker = threading.Thread(
            target=self._run,
            args=(list(folders), force),
            daemon=True,
            name="monoshiri-indexer",
        )
        self._worker.start()
        return True

    def cancel(self) -> None:
        self._cancel_event.set()
        self._update(phase_label="⏹️ キャンセル中...")

    def reset(self) -> None:
        if not self.is_running:
            self._update(status="idle", progress=0.0)

    def get_elapsed_str(self) -> str:
        elapsed = time.time() - self.start_time if self.start_time else 0
        if elapsed >= 60:
            return f"{int(elapsed // 60)}分{int(elapsed % 60)}秒"
        return f"{int(elapsed)}秒"

    def _update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
            if self.start_time:
                self.elapsed = time.time() - self.start_time
            if self.total_count > 0:
                self.progress = min(self.done_count / self.total_count, 1.0)

    def _run(self, folders: list[Path], force: bool = False) -> None:
        """インデックス処理のメインロジック（バックグラウンドスレッドで実行）"""
        try:
            self._update(status="scanning", phase_label="📂 フォルダをスキャン中...")
            files_to_process, deleted_paths = scan_and_diff(folders, force=force)

            if self._cancel_event.is_set():
                self._update(status="cancelled", phase_label="キャンセルしました")
                return

            if deleted_paths:
                self._update(
                    status="deleting",
                    phase_label=f"🗑️ 削除ファイルを処理中（{len(deleted_paths)}件）...",
                )
                delete_from_chroma(deleted_paths)

            if not files_to_process:
                self._update(
                    status="done",
                    phase_label="✅ インデックスは最新の状態です",
                    progress=1.0,
                    result={"processed": 0, "skipped": 0, "no_change": True},
                )
                return

            total = len(files_to_process)
            self._update(
                status="indexing",
                total_count=total,
                phase_label="🔄 インデックスを作成中...",
            )

            delete_from_chroma([str(f) for f, _ in files_to_process])

            skip_log = load_skip_log()
            processing_paths = {str(f) for f, _ in files_to_process}
            skip_log = [s for s in skip_log if s["file_path"] not in processing_paths]

            collection = get_collection()

            def on_progress(done: int, total_: int, fname: str) -> None:
                self._update(done_count=done, current_file=fname)

            result = run_indexing_pipeline(
                files_to_process=files_to_process,
                collection=collection,
                skip_log=skip_log,
                cancel_event=self._cancel_event,
                progress_callback=on_progress,
            )

            save_skip_log(skip_log)

            final_status = "cancelled" if result.get("cancelled") else "done"
            self._update(
                status=final_status,
                progress=1.0,
                result=result,
                phase_label="✅ 完了" if final_status == "done" else "⏹️ キャンセルしました",
            )

        except Exception as e:
            logger.error(f"IndexManager エラー: {e}", exc_info=True)
            self._update(
                status="error",
                error_msg=str(e),
                phase_label="❌ エラーが発生しました",
            )
