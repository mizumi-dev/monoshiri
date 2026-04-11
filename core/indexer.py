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
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from core.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, SKIP_LOG_FILE,
)
from core.chromadb_store import get_chroma_client, get_collection, get_index_stats
from core.extractor import extract_text, scan_folder
from core.hash_manager import (
    compute_hash, get_diff, load_hashes, save_hashes,
    make_folder_id, delete_folder_hashes,
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
) -> tuple[list[tuple[Path, str]], list[str]]:
    """
    フォルダをスキャンして差分を検出する。

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

    Args:
        files_to_process: [(file_path, folder_id), ...]
        collection: ChromaDBコレクション
        skip_log: スキップログ（参照渡しで追記）
        cancel_event: キャンセルシグナル（set()されたら停止）
        progress_callback: 進捗通知 fn(done_count: int, total: int, current_file: str)

    Returns:
        {"processed": int, "skipped": int, "cancelled": bool}
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
        with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
            futures = {
                pool.submit(extract_file_chunks, fp, fid): (fp, fid)
                for fp, fid in files_to_process
            }
            for future in futures:
                if cancel_event.is_set():
                    future.cancel()
                fp, fid = futures[future]
                try:
                    result = future.result()
                    chunk_queue.put((fp, fid, result))
                except Exception as e:
                    logger.error(f"抽出エラー {fp}: {e}")
                    chunk_queue.put((fp, fid, ([], [], str(e))))
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
#
# Streamlitのrerunをまたいでインデックス処理の状態を保持する。
# DownloadManagerと同様のシングルトンパターン。
# UIはこのクラスを通じて処理を開始・キャンセル・状態参照する。

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

        # 状態（UI側から読み取る）
        self.status: str = "idle"
        # idle | scanning | deleting | indexing | done | cancelled | error
        self.phase_label: str = ""
        self.progress: float = 0.0
        self.done_count: int = 0
        self.total_count: int = 0
        self.current_file: str = ""
        self.start_time: float = 0.0
        self.elapsed: float = 0.0
        self.result: dict = {}
        self.error_msg: str = ""

    # ── 公開API ────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    @property
    def is_finished(self) -> bool:
        return self.status in ("done", "cancelled", "error")

    def start(self, folders: list[Path]) -> bool:
        """
        インデックス処理をバックグラウンドで開始する。
        既に実行中の場合はFalseを返す。
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
            args=(list(folders),),
            daemon=True,
            name="monoshiri-indexer",
        )
        self._worker.start()
        return True

    def cancel(self) -> None:
        """実行中のインデックス処理をキャンセルする"""
        self._cancel_event.set()
        self._update(phase_label="⏹️ キャンセル中...")

    def reset(self) -> None:
        """完了・エラー後にidle状態に戻す"""
        if not self.is_running:
            self._update(status="idle", progress=0.0)

    def get_elapsed_str(self) -> str:
        """経過時間を文字列で返す"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        if elapsed >= 60:
            return f"{int(elapsed // 60)}分{int(elapsed % 60)}秒"
        return f"{int(elapsed)}秒"

    # ── 内部処理 ────────────────────────────────────────────────

    def _update(self, **kwargs) -> None:
        """状態を更新する（スレッドセーフ）"""
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
            if self.start_time:
                self.elapsed = time.time() - self.start_time
            if self.total_count > 0:
                self.progress = min(self.done_count / self.total_count, 1.0)

    def _run(self, folders: list[Path]) -> None:
        """インデックス処理のメインロジック（バックグラウンドスレッドで実行）"""
        try:
            # ── Embeddingモデルの事前ロード ──
            # flush()の中で初回ロードが起きると進捗バーが長時間止まる。
            # インデックス開始時に明示的にロードして「準備中」を表示する。
            self._update(status="loading_model", phase_label="🤖 Embeddingモデルを準備中（初回のみ時間がかかります）...")
            from core.embedder import get_embedding_model
            get_embedding_model()  # キャッシュ済みなら瞬時に完了

            if self._cancel_event.is_set():
                self._update(status="cancelled", phase_label="キャンセルしました")
                return

            # ── スキャン & 差分検出 ──
            self._update(status="scanning", phase_label="📂 フォルダをスキャン中...")
            files_to_process, deleted_paths = scan_and_diff(folders)

            if self._cancel_event.is_set():
                self._update(status="cancelled", phase_label="キャンセルしました")
                return

            # ── 削除済みファイルの処理 ──
            if deleted_paths:
                self._update(
                    status="deleting",
                    phase_label=f"🗑️ 削除ファイルを処理中（{len(deleted_paths)}件）...",
                )
                delete_from_chroma(deleted_paths)

            # ── 変更なしの場合 ──
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
                phase_label=f"🔄 インデックスを作成中...",
            )

            # 変更ファイルの旧エントリを削除
            delete_from_chroma([str(f) for f, _ in files_to_process])

            # スキップログの準備
            skip_log = load_skip_log()
            processing_paths = {str(f) for f, _ in files_to_process}
            skip_log = [s for s in skip_log if s["file_path"] not in processing_paths]

            collection = get_collection()

            def on_progress(done: int, total_: int, fname: str) -> None:
                self._update(done_count=done, current_file=fname)

            # ── インデックス処理実行 ──
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
     