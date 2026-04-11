"""
モノシリ SHA-256ハッシュ管理モジュール
差分インデックスのためのファイル変更検知を担当する。
変更のないファイルは再処理しない（効率化）。

v3.4 変更点:
  - ハッシュ管理ファイルを単一の .hash_manifest に統合
    （保存場所: %APPDATA%/monoshiri/cache/.hash_manifest）
  - フォルダ別分割から全フォルダ一元管理方式へ変更
  - キー形式: "{folder_id}:{file_path}" → ファイルパスの機密性を保護
"""
from __future__ import annotations
import hashlib
import json
import logging
import threading
from pathlib import Path

from core.config import HASH_MANIFEST_FILE

logger = logging.getLogger(__name__)

# スレッドセーフなマニフェスト書き込みロック
_manifest_lock = threading.Lock()


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


def compute_lightweight_hash(file_path: Path) -> str | None:
    """
    ファイルの軽量ハッシュを計算する（サイズ+mtimeのみ）。
    force=True時など、高速性が優先される場面で使用。
    完全性は低いが、I/O負荷が少ない。
    """
    try:
        stat = file_path.stat()
        # サイズとmtimeを組み合わせたハッシュ
        content = f"{stat.st_size}:{stat.st_mtime:.0f}"
        return hashlib.sha256(content.encode()).hexdigest()
    except (OSError, PermissionError) as e:
        logger.warning(f"軽量ハッシュ計算失敗 {file_path.name}: {e}")
        return None


def make_folder_id(folder_path: Path | str) -> str:
    """フォルダパスから一意のIDを生成する（SHA-256ハッシュ）"""
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)
    return hashlib.sha256(str(folder_path.resolve()).encode()).hexdigest()[:16]


# ─── マニフェスト全体の読み書き ────────────────────────────────

def _load_manifest() -> dict[str, str]:
    """
    .hash_manifest ファイルを読み込む。
    形式: { "<folder_id>:<file_path>": "<sha256>" }
    初回または破損時は空dictを返す。
    """
    if HASH_MANIFEST_FILE.exists():
        try:
            with open(HASH_MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"ハッシュマニフェスト読み込みエラー（再初期化）: {e}")
    return {}


def _save_manifest(manifest: dict[str, str]) -> None:
    """
    .hash_manifest ファイルを保存する（スレッドセーフ）。
    """
    HASH_MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HASH_MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, separators=(",", ":"))


def _make_key(folder_id: str, file_path: str) -> str:
    """マニフェストのキーを生成する"""
    return f"{folder_id}:{file_path}"


# ─── フォルダ単位のハッシュ操作（外部インターフェース） ──────────

def load_hashes(folder_id: str) -> dict[str, str]:
    """
    指定フォルダのハッシュ一覧を返す。
    戻り値: { file_path_str: sha256_hex }
    """
    manifest = _load_manifest()
    prefix = f"{folder_id}:"
    result = {}
    for key, value in manifest.items():
        if key.startswith(prefix):
            file_path = key[len(prefix):]
            result[file_path] = value
    return result


def save_hashes(folder_id: str, hashes: dict[str, str]) -> None:
    """
    指定フォルダのハッシュをマニフェストに保存する。
    既存の同フォルダエントリはすべて置き換える。
    """
    with _manifest_lock:
        manifest = _load_manifest()
        prefix = f"{folder_id}:"
        # 既存エントリを削除
        manifest = {k: v for k, v in manifest.items() if not k.startswith(prefix)}
        # 新エントリを追加
        for file_path, file_hash in hashes.items():
            manifest[_make_key(folder_id, file_path)] = file_hash
        _save_manifest(manifest)


def update_file_hash(folder_id: str, file_path: Path, file_hash: str) -> None:
    """1ファイルのハッシュを更新する（スレッドセーフ）"""
    with _manifest_lock:
        manifest = _load_manifest()
        manifest[_make_key(folder_id, str(file_path))] = file_hash
        _save_manifest(manifest)


def delete_folder_hashes(folder_id: str) -> None:
    """フォルダのハッシュエントリをマニフェストから削除する"""
    with _manifest_lock:
        manifest = _load_manifest()
        prefix = f"{folder_id}:"
        manifest = {k: v for k, v in manifest.items() if not k.startswith(prefix)}
        _save_manifest(manifest)


def clear_all_hashes() -> None:
    """全フォルダのハッシュを削除する（全件再インデックス用）"""
    with _manifest_lock:
        _save_manifest({})
    logger.info("全ハッシュマニフェストをクリアしました（全件再インデックス）")


def get_diff(
    folder_id: str,
    current_files: list[Path],
    use_full_hash: bool = False,
) -> tuple[list[Path], list[Path], list[str]]:
    """
    前回のインデックスとの差分を検出する。

    Args:
        folder_id: フォルダID
        current_files: 現在のファイル一覧
        use_full_hash: Trueの場合、SHA-256フルハッシュを使用（遅いが正確）
                        Falseの場合、軽量ハッシュ（サイズ+mtime）を使用（高速）

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

    # ハッシュ関数を選択（デフォルトは高速な軽量ハッシュ）
    hash_func = compute_hash if use_full_hash else compute_lightweight_hash
    hash_type = "フル" if use_full_hash else "軽量"

    for file_path in current_files:
        path_str = str(file_path)
        current_hash = hash_func(file_path)
        if current_hash is None:
            # 読み取れないファイルはスキップ（変更なし扱い）
            continue

        if path_str not in saved_hashes or saved_hashes[path_str] != current_hash:
            new_or_modified.append(file_path)
        else:
            unchanged.append(file_path)

    # 削除されたファイル = 前回ハッシュに存在するが今回のスキャンにない
    deleted_paths = [p for p in saved_hashes if p not in current_path_strs]

    logger.info(f"差分検出完了[{hash_type}]: 新規・変更={len(new_or_modified)}, 変更なし={len(unchanged)}, 削除={len(deleted_paths)}")

    return new_or_modified, unchanged, deleted_paths
