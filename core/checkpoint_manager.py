"""
モノシリ チャンクチェックポイント管理モジュール

強制終了時やクラッシュ時にチャンク単位で再開できる機能を提供する。
ChromaDBへの追加直後にチェックポイントを保存し、重複処理を防止する。
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Optional

from core.config import DATA_DIR

logger = logging.getLogger(__name__)

# チェックポイントファイルパス
CHECKPOINT_FILE = DATA_DIR / "chunk_checkpoint.json"


def _load_checkpoint() -> dict:
    """チェックポイントファイルを読み込む"""
    if not CHECKPOINT_FILE.exists():
        return {
            "processed_chunks": [],
            "incomplete_files": {},
            "version": "1.0"
        }
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 必須キーが欠けていれば初期化
            data.setdefault("processed_chunks", [])
            data.setdefault("incomplete_files", {})
            data.setdefault("version", "1.0")
            return data
    except Exception as e:
        logger.warning(f"[Checkpoint] 読み込み失敗、初期化: {e}")
        return {
            "processed_chunks": [],
            "incomplete_files": {},
            "version": "1.0"
        }


def _save_checkpoint(data: dict) -> None:
    """チェックポイントファイルを保存"""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # 重複除去
        data["processed_chunks"] = list(set(data["processed_chunks"]))
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[Checkpoint] 保存失敗: {e}")


def is_chunk_processed(chunk_id: str) -> bool:
    """チャンクが既に処理済みかチェック"""
    checkpoint = _load_checkpoint()
    return chunk_id in checkpoint["processed_chunks"]


def mark_chunks_processed(chunk_ids: list[str]) -> None:
    """チャンクを処理済みとしてマーク"""
    if not chunk_ids:
        return
    checkpoint = _load_checkpoint()
    checkpoint["processed_chunks"].extend(chunk_ids)
    _save_checkpoint(checkpoint)
    logger.debug(f"[Checkpoint] {len(chunk_ids)}チャンクを保存")


def mark_file_incomplete(
    folder_id: str,
    file_path: str,
    file_hash: str,
    expected_chunks: int,
    processed_chunks: int
) -> None:
    """ファイルを不完全としてマーク（処理中のクラッシュ対策）"""
    checkpoint = _load_checkpoint()
    
    if folder_id not in checkpoint["incomplete_files"]:
        checkpoint["incomplete_files"][folder_id] = {}
    
    checkpoint["incomplete_files"][folder_id][file_path] = {
        "hash": file_hash,
        "expected_chunks": expected_chunks,
        "processed_chunks": processed_chunks,
        "last_update": time.strftime("%Y-%m-%dT%H:%M:%S")
    }
    _save_checkpoint(checkpoint)
    logger.debug(f"[Checkpoint] 不完全ファイルマーク: {Path(file_path).name}")


def mark_file_complete(folder_id: str, file_path: str) -> None:
    """ファイル処理完了としてマーク（不完全リストから削除）"""
    checkpoint = _load_checkpoint()
    
    if folder_id in checkpoint.get("incomplete_files", {}):
        if file_path in checkpoint["incomplete_files"][folder_id]:
            del checkpoint["incomplete_files"][folder_id][file_path]
            logger.debug(f"[Checkpoint] 不完全マーク削除: {Path(file_path).name}")
    
    _save_checkpoint(checkpoint)


def get_incomplete_files(folder_id: str) -> dict[str, dict]:
    """不完全なファイル一覧を取得 {file_path: info}"""
    checkpoint = _load_checkpoint()
    return checkpoint.get("incomplete_files", {}).get(folder_id, {})


def clear_all_checkpoints() -> None:
    """全チェックポイントを削除（強制再インデックス時）"""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("[Checkpoint] 全チェックポイント削除")


def remove_file_chunks_from_checkpoint(file_path: str) -> None:
    """特定ファイルのチャンクをチェックポイントから削除（再処理前）"""
    checkpoint = _load_checkpoint()
    
    # ファイルパスに紐づくチャンクIDを削除
    file_path_str = str(file_path)
    original_count = len(checkpoint["processed_chunks"])
    checkpoint["processed_chunks"] = [
        cid for cid in checkpoint["processed_chunks"]
        if not cid.startswith(f"{file_path_str}_chunk_")
    ]
    removed_count = original_count - len(checkpoint["processed_chunks"])
    
    if removed_count > 0:
        logger.info(f"[Checkpoint] {removed_count}チャンクを削除: {Path(file_path).name}")
    
    _save_checkpoint(checkpoint)
