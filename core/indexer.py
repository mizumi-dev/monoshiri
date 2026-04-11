"""
モノシリ インデックス作成モジュール
ファイル抽出 → チャンク分割 → Embedding → ChromaDB格納のパイプラインを担当する。
差分検知により変更ファイルのみ再処理する。

TODO(仕様書更新): v3.4仕様書はチャンク分割を「600トークン/100トークンオーバーラップ」と記載しているが、
  実装は「1200文字/500文字オーバーラップ」（文字ベース）。仕様書を実装に合わせて更新すること。
TODO(仕様書更新): v3.4仕様書の「最小チャンクマージ（50トークン未満→前チャンクに結合）」は未実装。
  Phase 2でMeCabトークナイザ統合時に対応予定。

アーキテクチャ:
- Producer-Consumerパターンで抽出とEmbeddingを並列化
- threading.Eventによるキャンセル機構
- IndexManagerシングルトンでUIとバックグラウンド処理を分離
"""
from __future__ import annotations
import json
import logging
import os
import queue
import signal
import threading
import time
import uuid
from concurrent.futures import CancelledError, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from core.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, SKIP_LOG_FILE, CHROMA_COLLECTION,
)
from core.chromadb_store import get_chroma_client, get_collection, get_index_stats, reset_chroma_client
from core.extractor import extract_text, scan_folder
from core.hash_manager import (
    compute_hash, compute_lightweight_hash, get_diff, load_hashes, save_hashes,
    make_folder_id, delete_folder_hashes, clear_all_hashes,
)
from core.embedder import embed_texts, get_embedding_model, get_optimal_batch_size
from core.checkpoint_manager import (
    is_chunk_processed, mark_chunks_processed,
    mark_file_incomplete, mark_file_complete,
    get_incomplete_files, remove_file_chunks_from_checkpoint
)
from core.chromadb_store import delete_chunks_by_file

logger = logging.getLogger(__name__)

# multiprocessing用のグローバル設定
_MAX_WORKERS = min(8, (os.cpu_count() or 4))  # CPUコア数に基づく最適ワーカー数

# ChromaDBバッチサイズ（一度にEmbeddingする件数）
# GPU効率化: VRAMに応じて自動調整（embedder.get_optimal_batch_size()で取得）
# 注意: この値は初期値。実際にはembedderがVRAMに応じて自動調整
BATCH_SIZE = None  # None = 自動検出

# 少量のチャンクでも定期的にflushして進捗とGPU処理を早める
FILES_PER_FLUSH = 4

# ファイル抽出の並列数（I/Oバウンドなので多めに）
EXTRACT_WORKERS = 8

# Producerキューの最大サイズ（メモリ制限）
# GPU効率化: 抽出が速くなってもQueueが空にならないように大幅拡大
PRODUCER_QUEUE_SIZE = 300

# _SENTINEL: Producerがキュー終端を伝えるシグナル
_SENTINEL = None

# ─── テキスト分割（キャッシュ化）────────────────────────────────

# スプリッターのキャッシュ（初期化コスト削減）
_splitter_cache: dict[tuple[int, int], object] = {}


def _get_splitter(chunk_size: int, overlap: int):
    """キャッシュされたRecursiveCharacterTextSplitterを取得"""
    key = (chunk_size, overlap)
    if key not in _splitter_cache:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        _splitter_cache[key] = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "．", "、", "，", "　", " ", ""],
        )
    return _splitter_cache[key]


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    テキストをチャンクに分割する。
    LangChainのRecursiveCharacterTextSplitterを使用（キャッシュ化済み）。
    日本語の区切り文字を考慮した分割を行う。
    """
    if not text or not text.strip():
        return []

    try:
        splitter = _get_splitter(chunk_size, overlap)
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
    status_callback=None,
) -> dict:
    """
    Producer-ConsumerパターンでEmbeddingとファイル抽出を並列化する。
    チャンク単位のチェックポイントにより、強制終了時も再開可能。
    """
    total = len(files_to_process)
    if total == 0:
        return {"processed": 0, "skipped": 0, "cancelled": False}

    # ── 不完全ファイルの検出と再処理準備 ────────────────────────
    # folder_idを取得（ファイルパスから推定）
    if files_to_process:
        folder_id = make_folder_id(files_to_process[0][0].parent)
        
        incomplete_files = get_incomplete_files(folder_id)
        if incomplete_files:
            logger.info(f"[Checkpoint] 不完全ファイル: {len(incomplete_files)}件")
            for fp_str, info in incomplete_files.items():
                fp = Path(fp_str)
                if fp.exists():
                    # 不完全ファイルの既存チャンクを削除
                    deleted = delete_chunks_by_file(fp)
                    if deleted > 0:
                        logger.info(f"[Checkpoint] {fp.name}: {deleted}チャンク削除、再処理準備完了")
                    # checkpointからも削除
                    remove_file_chunks_from_checkpoint(fp)

    if status_callback:
        status_callback("warming", "")
    logger.info("Embeddingモデルのウォームアップを開始します")
    get_embedding_model()
    logger.info("Embeddingモデルのウォームアップが完了しました")
    if status_callback:
        status_callback("extracting", "")

    chunk_queue: queue.Queue = queue.Queue(maxsize=PRODUCER_QUEUE_SIZE)

    processed_files = 0
    skipped_count = 0
    done_count = 0
    hash_buffer: dict[str, dict[str, str]] = {}

    # ── Producer: ファイル抽出を並列で実行（ファイルサイズに応じた動的並列度）────────────────
    def _get_file_size_category(file_path: Path) -> str:
        """ファイルサイズに応じた分類"""
        try:
            size = file_path.stat().st_size
            if size < 1 * 1024 * 1024:  # < 1MB
                return "small"
            elif size > 10 * 1024 * 1024:  # > 10MB
                return "large"
            else:
                return "medium"
        except Exception:
            return "medium"  # 不明な場合は中サイズ扱い

    def producer():
        # ファイルをサイズで分類
        small_files = [(fp, fid) for fp, fid in files_to_process if _get_file_size_category(fp) == "small"]
        medium_files = [(fp, fid) for fp, fid in files_to_process if _get_file_size_category(fp) == "medium"]
        large_files = [(fp, fid) for fp, fid in files_to_process if _get_file_size_category(fp) == "large"]

        logger.info(f"ファイル分類: 小={len(small_files)}, 中={len(medium_files)}, 大={len(large_files)}")

        # 小ファイル: 高並列（ProcessPoolExecutorでGIL回避）
        if small_files:
            logger.info(f"小ファイル処理開始: {len(small_files)}ファイル（並列度: {_MAX_WORKERS}）")
            pool = ProcessPoolExecutor(max_workers=_MAX_WORKERS)
            try:
                futures = {pool.submit(extract_file_chunks, fp, fid): (fp, fid) for fp, fid in small_files}
                for future in as_completed(futures):
                    if cancel_event.is_set():
                        logger.info("[Cancel] キャンセル検出、プロセスプールを強制終了します")
                        # 全プロセスを強制終了
                        for proc in pool._processes.values():
                            try:
                                proc.terminate()
                                proc.join(timeout=1)
                                if proc.is_alive():
                                    proc.kill()
                            except Exception:
                                pass
                        pool.shutdown(wait=False, cancel_futures=True)
                        chunk_queue.put(_SENTINEL)
                        return
                    fp, fid = futures[future]
                    try:
                        result = future.result()
                        logger.info(f"抽出完了[小]: {fp.name}")
                        chunk_queue.put((fp, fid, result))
                    except Exception as e:
                        logger.error(f"抽出エラー[小] {fp}: {e}")
                        chunk_queue.put((fp, fid, ([], [], str(e))))
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

        # 中ファイル: 適度な並列（ProcessPoolExecutorでGIL回避）
        if medium_files:
            logger.info(f"中ファイル処理開始: {len(medium_files)}ファイル（並列度: {max(2, _MAX_WORKERS // 2)}）")
            pool = ProcessPoolExecutor(max_workers=max(2, _MAX_WORKERS // 2))
            try:
                futures = {pool.submit(extract_file_chunks, fp, fid): (fp, fid) for fp, fid in medium_files}
                for future in as_completed(futures):
                    if cancel_event.is_set():
                        logger.info("[Cancel] キャンセル検出、プロセスプールを強制終了します")
                        for proc in pool._processes.values():
                            try:
                                proc.terminate()
                                proc.join(timeout=1)
                                if proc.is_alive():
                                    proc.kill()
                            except Exception:
                                pass
                        pool.shutdown(wait=False, cancel_futures=True)
                        chunk_queue.put(_SENTINEL)
                        return
                    fp, fid = futures[future]
                    try:
                        result = future.result()
                        logger.info(f"抽出完了[中]: {fp.name}")
                        chunk_queue.put((fp, fid, result))
                    except Exception as e:
                        logger.error(f"抽出エラー[中] {fp}: {e}")
                        chunk_queue.put((fp, fid, ([], [], str(e))))
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

        # 大ファイル: 低並列（ProcessPoolExecutorで2 workers、メモリ保護）
        if large_files:
            logger.info(f"大ファイル処理開始: {len(large_files)}ファイル（並列度: 2）")
            pool = ProcessPoolExecutor(max_workers=2)
            try:
                futures = {pool.submit(extract_file_chunks, fp, fid): (fp, fid) for fp, fid in large_files}
                for future in as_completed(futures):
                    if cancel_event.is_set():
                        logger.info("[Cancel] キャンセル検出、プロセスプールを強制終了します")
                        for proc in pool._processes.values():
                            try:
                                proc.terminate()
                                proc.join(timeout=1)
                                if proc.is_alive():
                                    proc.kill()
                            except Exception:
                                pass
                        pool.shutdown(wait=False, cancel_futures=True)
                        chunk_queue.put(_SENTINEL)
                        return
                    fp, fid = futures[future]
                    try:
                        result = future.result()
                        logger.info(f"抽出完了[大]: {fp.name}")
                        chunk_queue.put((fp, fid, result))
                    except Exception as e:
                        logger.error(f"抽出エラー[大] {fp}: {e}")
                        chunk_queue.put((fp, fid, ([], [], str(e))))
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

        chunk_queue.put(_SENTINEL)

    producer_thread = threading.Thread(target=producer, daemon=True, name="monoshiri-producer")
    producer_thread.start()

    # ── GPU常時稼働パターン: バッチ組み立て + Embedding並列化 ───────────────
    # ダブルバッファリング: バッチAを組み立て中に、バッチBをGPUで処理
    batch_texts_a: list[str] = []
    batch_ids_a: list[str] = []
    batch_metas_a: list[dict] = []
    batch_texts_b: list[str] = []
    batch_ids_b: list[str] = []
    batch_metas_b: list[dict] = []
    active_batch = 'a'  # 現在チャンクを溜めているバッチ

    # GPU Embedding用のQueueと結果格納
    embedding_queue: queue.Queue = queue.Queue(maxsize=2)  # 最大2バッチ先読み
    embedding_results: list = []
    embedding_error: list = []  # 例外格納用（長さ1のリスト）

    def embedding_worker():
        """GPU専用スレッド: 常時Embeddingを実行"""
        while True:
            try:
                batch_data = embedding_queue.get(timeout=0.5)
                if batch_data is _SENTINEL:
                    break

                texts, ids, metas = batch_data
                if not texts:
                    continue

                def _cancel_check():
                    return cancel_event.is_set()

                try:
                    if status_callback:
                        status_callback("embedding", f"{len(texts)}チャンク")
                    logger.info(f"[GPU] Embedding開始: {len(texts)}チャンク")

                    embeddings = embed_texts(texts, cancel_check=_cancel_check)

                    # ChromaDBに追加
                    collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=metas,
                    )

                    # チェックポイント保存（チャンク単位で永続化）
                    mark_chunks_processed(ids)

                    processed_count = len(texts)
                    embedding_results.append(processed_count)
                    logger.info(f"[GPU] ChromaDB追加完了: {processed_count}チャンク（チェックポイント保存）")

                    if status_callback:
                        status_callback("extracting", "")

                except CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"[GPU] Embeddingエラー: {e}")
                    embedding_error.append(e)
                    raise

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[GPU] Embeddingワーカーエラー: {e}")
                embedding_error.append(e)
                break

    def flush_hashes():
        """ハッシュを保存し、ファイル完了をチェックポイントに記録"""
        for fid, path_hash_map in hash_buffer.items():
            hashes = load_hashes(fid)
            hashes.update(path_hash_map)
            save_hashes(fid, hashes)
            # 完了したファイルの不完全マークを削除
            for file_path in path_hash_map.keys():
                mark_file_complete(fid, file_path)
        hash_buffer.clear()

    def submit_batch(batch_texts, batch_ids, batch_metas):
        """バッチをGPU Embeddingキューに投入"""
        if not batch_texts:
            return 0
        # Queueが満杯なら待機（ブロッキング）
        embedding_queue.put((batch_texts.copy(), batch_ids.copy(), batch_metas.copy()))
        return len(batch_texts)

    # GPU Embeddingスレッド開始
    embedding_thread = threading.Thread(target=embedding_worker, daemon=True, name="monoshiri-gpu")
    embedding_thread.start()
    logger.info("[GPU] Embeddingスレッド開始")

    cancelled = False
    try:
        while True:
            if cancel_event.is_set():
                cancelled = True
                break

            try:
                item = chunk_queue.get(timeout=1.0)
            except queue.Empty:
                # Queueが空なら、埋めているバッチを送信
                if active_batch == 'a' and batch_texts_a:
                    submit_batch(batch_texts_a, batch_ids_a, batch_metas_a)
                    batch_texts_a.clear()
                    batch_ids_a.clear()
                    batch_metas_a.clear()
                elif active_batch == 'b' and batch_texts_b:
                    submit_batch(batch_texts_b, batch_ids_b, batch_metas_b)
                    batch_texts_b.clear()
                    batch_ids_b.clear()
                    batch_metas_b.clear()
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
                # スキップファイルは不完全マークを削除
                mark_file_complete(fid, str(fp))
            else:
                # アクティブなバッチにチャンクを追加（既に処理済みのチャンクはスキップ）
                processed_chunk_count = 0
                if active_batch == 'a':
                    for c in chunks:
                        # チャンク単位で重複チェック
                        if is_chunk_processed(c["id"]):
                            continue
                        batch_texts_a.append(c["text"])
                        batch_ids_a.append(c["id"])
                        batch_metas_a.append(c["metadata"])
                        processed_chunk_count += 1
                else:
                    for c in chunks:
                        if is_chunk_processed(c["id"]):
                            continue
                        batch_texts_b.append(c["text"])
                        batch_ids_b.append(c["id"])
                        batch_metas_b.append(c["metadata"])
                        processed_chunk_count += 1

                # 不完全マークを更新（実際に処理するチャンク数）
                if processed_chunk_count > 0:
                    file_hash = hash_updates[0]["hash"] if hash_updates else compute_lightweight_hash(fp)
                    mark_file_incomplete(
                        fid, str(fp), file_hash,
                        expected_chunks=len(chunks),
                        processed_chunks=processed_chunk_count
                    )

                for hu in hash_updates:
                    fid_key = hu["folder_id"]
                    if fid_key not in hash_buffer:
                        hash_buffer[fid_key] = {}
                    hash_buffer[fid_key][hu["file_path"]] = hu["hash"]

            if progress_callback:
                progress_callback(done_count, total, fp.name)
            if status_callback:
                status_callback("extracting", fp.name)

            # バッチが満タンになったら切り替え
            # VRAMに応じた動的BATCH_SIZEを取得
            effective_batch_size = BATCH_SIZE if BATCH_SIZE is not None else get_optimal_batch_size()
            current_batch = batch_texts_a if active_batch == 'a' else batch_texts_b
            if len(current_batch) >= effective_batch_size:
                # 満タンのバッチを送信
                if active_batch == 'a':
                    submit_batch(batch_texts_a, batch_ids_a, batch_metas_a)
                    batch_texts_a.clear()
                    batch_ids_a.clear()
                    batch_metas_a.clear()
                    active_batch = 'b'
                    logger.info("[Accumulator] バッチA送信完了、バッチBに切り替え")
                else:
                    submit_batch(batch_texts_b, batch_ids_b, batch_metas_b)
                    batch_texts_b.clear()
                    batch_ids_b.clear()
                    batch_metas_b.clear()
                    active_batch = 'a'
                    logger.info("[Accumulator] バッチB送信完了、バッチAに切り替え")

                flush_hashes()

        # 残りのバッチを送信
        if not cancelled:
            if batch_texts_a:
                submit_batch(batch_texts_a, batch_ids_a, batch_metas_a)
            if batch_texts_b:
                submit_batch(batch_texts_b, batch_ids_b, batch_metas_b)

        # Embeddingスレッドに終了信号
        embedding_queue.put(_SENTINEL)
        flush_hashes()

        # GPU処理完了待ち
        embedding_thread.join(timeout=300)  # 最大5分待機

    except CancelledError:
        cancelled = True
        flush_hashes()
    except Exception as e:
        logger.error(f"Consumer例外: {e}")
        flush_hashes()
        raise
    finally:
        # 全スレッドの終了を待機
        producer_thread.join(timeout=5)
        if embedding_thread.is_alive():
            embedding_queue.put(_SENTINEL)
            embedding_thread.join(timeout=10)

    # 総処理チャンク数を計算
    total_chunks = sum(embedding_results)

    return {
        "processed": done_count - skipped_count,
        "skipped": skipped_count,
        "cancelled": cancelled,
        "total_chunks": total_chunks,
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
                phase_label="🔄 既存チャンクを削除中...",
            )

            if force:
                # 全件再インデックス時: コレクションを丸ごと削除→再作成（高速）
                logger.info(f"全件再インデックス: コレクションを再作成します（{total}ファイル）")
                try:
                    client = get_chroma_client()
                    client.delete_collection(CHROMA_COLLECTION)
                except Exception as e:
                    logger.warning(f"コレクション削除エラー（無視して続行）: {e}")
                # 古いクライアント参照を破棄し、新クライアントで新コレクションを取得
                reset_chroma_client()
                collection = get_collection()
                logger.info("全件再インデックス: 新コレクションを作成しました")
            else:
                delete_from_chroma([str(f) for f, _ in files_to_process])
                collection = get_collection()

            self._update(phase_label="🔄 インデックスを作成中...")

            skip_log = load_skip_log()
            processing_paths = {str(f) for f, _ in files_to_process}
            skip_log = [s for s in skip_log if s["file_path"] not in processing_paths]

            def on_progress(done: int, total_: int, fname: str) -> None:
                self._update(done_count=done, current_file=fname)

            def on_status(stage: str, detail: str) -> None:
                if stage == "warming":
                    self._update(phase_label="🤖 Embeddingモデルを準備中...", current_file="")
                elif stage == "extracting":
                    self._update(
                        phase_label="📄 抽出・チャンク化中...",
                        current_file=detail or self.current_file,
                    )
                elif stage == "embedding":
                    self._update(
                        phase_label=f"🚀 Embedding・登録中... {detail}" if detail else "🚀 Embedding・登録中...",
                    )

            result = run_indexing_pipeline(
                files_to_process=files_to_process,
                collection=collection,
                skip_log=skip_log,
                cancel_event=self._cancel_event,
                progress_callback=on_progress,
                status_callback=on_status,
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
