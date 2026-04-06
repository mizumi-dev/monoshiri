"""
モノシリ SHA-256ハッシュ管理モジュール
差分インデックスのためのファイル変更検知を担当する。
変更のないファイルは再処理しない（効率化）。
"""
from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path

from core.config import HASH_DIR

logger = logging.getLogger(__name__)


def compute_hash(file_path: Path) -> str | None:
    """
    ファイルのSHA-256ハッシュを計算する。
    読み取れない場合はNoneを返す。
    """
    try:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, PermissionError) as e:
        logger.warning(f"ハッシュ計算失敗 {file_path.name}: {e}")
        return None


def make_folder_id(folder_path: Path) -> str:
    """フォルダパスから一意のIDを生成する（MD5ハッシュ）"""
    return hashlib.md5(str(folder_path.resolve()).encode()).hexdigest()


def _get_hash_file(folder_id: str) -> Path:
    return HASH_DIR / f"{folder_id}.json"


def load_hashes(folder_id: str) -> dict[str, str]:
    """保存済みハッシュを読み込む。初回は空dictを返す。"""
    hash_file = _get_hash_file(folder_id)
    if hash_file.exists():
        try:
            with open(hash_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_hashes(folder_id: str, hashes: dict[str, str]) -> None:
    """ハッシュをJSONファイルに保存する"""
    hash_file = _get_hash_file(folder_id)
    with open(hash_file, "w", encoding="utf-8") as f:
        json.dump(hashes, f, ensure_ascii=False, indent=2)


def update_file_hash(folder_id: str, file_path: Path, file_hash: str) -> None:
    """1ファイルのハッシュを更新する"""
    hashes = load_hashes(folder_id)
    hashes[str(file_path)] = file_hash
    save_hashes(folder_id, hashes)


def delete_folder_hashes(folder_id: str) -> None:
    """フォルダのハッシュファイルを削除する"""
    hash_file = _get_hash_file(folder_id)
    if hash_file.exists():
        hash_file.unlink()


def get_diff(
    folder_id: str,
    current_files: list[Path],
) -> tuple[list[Path], list[Path], list[str]]:
    """
    前回のインデックスとの差分を検出する。

    Args:
        folder_id: フォルダID
        current_files: 現在のファイル一覧

    Returns:
        (new_or_modified, unchanged, deleted_paths)
        - new_or_modified: 新規・変更されたファイル → 再インデックス対象
        - unchanged: 変更なしのファイル
        - deleted_paths: 削除されたファイルのパス文字列一覧
    """
    saved_hashes = load_hashes(folder_id)
    current_path_strs = {str(f) for f in current_files}

    new_or_modified: list[Path] = []
    unchanged: list[Path] = []

    for file_path in current_files:
        path_str = str(file_path)
        current_hash = compute_hash(file_path)
        if current_hash is None:
            # 読み取れないファイルはスキップ（変更なし扱い）
            continue

        if path_str not in saved_hashes or saved_hashes[path_str] != current_hash:
            new_or_modified.append(file_path)
        else:
            unchanged.append(file_path)

    # 削除されたファイル = 前回ハッシュに存在するが今回のスキャンにない
    deleted_paths = [p for p in saved_hashes if p not in current_path_strs]

    return new_or_modified, unchanged, deleted_paths
