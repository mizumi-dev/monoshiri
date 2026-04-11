"""
モノシリ ダウンロードキューマネージャー

スレッドセーフな順次ダウンロード処理。
- 複数モデルをキューに積んで順次ダウンロード
- ファイルサイズ監視による進捗トラッキング
- キャンセル / リトライ対応
- Streamlit の @st.fragment(run_every=N) と組み合わせて UI に進捗を反映

ダウンロード実行ロジック（ネットワーク確認・urllib直接DL・HuggingFace Hub DL）も本モジュールで管理。
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Streamlit が downloader.py をホットリロードしてもシングルトンが消えないよう
# sys.modules に保存するためのキー（通常のモジュール名と衝突しない名前を使用）
_DM_PERSIST_KEY = "_monoshiri_download_manager_v1"


# ─── ネットワーク確認・ダウンロード実装 ─────────────────────────

def check_network_connectivity() -> tuple[bool, str]:
    """
    ネットワーク接続を簡易チェックする。

    Returns:
        (接続可能か, メッセージ)
    """
    from core.config import HF_MIRROR_URL
    test_urls = [
        HF_MIRROR_URL if HF_MIRROR_URL else "https://huggingface.co",
        "https://huggingface.co",
        "https://hf-mirror.com",
    ]
    seen: set[str] = set()
    for url in test_urls:
        if url in seen:
            continue
        seen.add(url)
        try:
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, timeout=10)
            return True, f"接続OK: {url}"
        except Exception:
            continue
    return False, (
        "HuggingFaceに接続できません。\n"
        "考えられる原因:\n"
        "  1. インターネット接続がない\n"
        "  2. ファイアウォール/セキュリティソフトがブロックしている\n"
        "  3. DNS解決に失敗している\n"
        "  4. プロキシ設定が必要\n"
        "\n対処法:\n"
        "  - ブラウザで https://huggingface.co にアクセスできるか確認\n"
        "  - セキュリティソフトの一時停止を試す\n"
        "  - 設定画面の「手動ダウンロード」からブラウザ経由でモデルを入手\n"
    )


def download_with_urllib(url: str, dest_path: Path, progress_callback=None) -> bool:
    """
    urllib を使ってファイルを直接ダウンロードする（huggingface_hub不要のフォールバック）。

    Args:
        url: ダウンロード元URL
        dest_path: 保存先パス
        progress_callback: 進捗コールバック（0.0〜1.0）

    Returns:
        成功した場合True
    """
    from core.config import DOWNLOAD_TIMEOUT
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    try:
        logger.info(f"直接ダウンロード開始: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "monoshiri/1.0"})
        response = urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT)

        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB

        with open(tmp_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total_size > 0:
                    progress_callback(downloaded / total_size)

        tmp_path.rename(dest_path)
        logger.info(f"直接ダウンロード完了: {dest_path}")
        return True

    except Exception as e:
        logger.error(f"直接ダウンロードエラー: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def download_model(model_name: str, progress_callback=None) -> tuple[bool, str]:
    """
    HuggingFaceからGGUFモデルをダウンロードする。
    リトライ・ミラー・直接ダウンロードの3段階フォールバック対応。

    Args:
        model_name: LLM_MODELSのキー
        progress_callback: 進捗コールバック（0.0〜1.0）

    Returns:
        (成功したか, メッセージ)
    """
    from core.config import (
        LLM_MODELS, MODELS_DIR,
        DOWNLOAD_MAX_RETRIES, HF_MIRROR_URL, GGUF_DIRECT_URLS,
    )

    if model_name not in LLM_MODELS:
        return False, f"未定義のモデル: {model_name}"

    model_info = LLM_MODELS[model_name]
    repo_id = model_info.get("repo_id")
    filename = model_info["filename"]

    if not repo_id:
        return False, f"カスタムモデルはHFダウンロード非対応: {model_name}"

    model_dir = MODELS_DIR / "llm"
    model_dir.mkdir(parents=True, exist_ok=True)
    dest_path = model_dir / filename

    if dest_path.exists():
        return True, "モデルは既にダウンロード済みです"

    errors = []

    # ── 方法1: urllib で直接ダウンロード（進捗コールバック対応・優先） ──
    # hf_hub_download は progress_callback 非対応のため、直接URLがある場合は urllib を優先する。
    # これにより UI の進捗バーがリアルタイム更新される。
    direct_url = GGUF_DIRECT_URLS.get(filename)
    if direct_url:
        effective_url = direct_url
        if HF_MIRROR_URL:
            effective_url = effective_url.replace("https://huggingface.co", HF_MIRROR_URL)
        logger.info(f"urllib で直接ダウンロード開始: {effective_url}")
        if download_with_urllib(effective_url, dest_path, progress_callback):
            return True, "ダウンロード完了"
        errors.append(f"直接ダウンロード失敗: {effective_url}")
        logger.warning("直接ダウンロード失敗。huggingface_hub にフォールバック...")

    # ── 方法2: huggingface_hub 経由（リトライ付き・フォールバック） ──
    # 直接URLがない場合、または urllib 失敗時のフォールバック。
    # ※進捗バーはファイルサイズ監視スレッドで概算更新される（精度低め）。
    try:
        from huggingface_hub import hf_hub_download

        for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
            try:
                logger.info(
                    f"[{attempt}/{DOWNLOAD_MAX_RETRIES}] "
                    f"huggingface_hub で {repo_id}/{filename} をダウンロード中..."
                )
                kwargs = dict(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                )
                if HF_MIRROR_URL:
                    kwargs["endpoint"] = HF_MIRROR_URL
                hf_hub_download(**kwargs)
                return True, "ダウンロード完了"

            except Exception as e:
                err_msg = str(e)
                errors.append(f"hf_hub 試行{attempt}: {err_msg}")
                logger.warning(f"ダウンロード試行 {attempt} 失敗: {err_msg}")
                if attempt < DOWNLOAD_MAX_RETRIES:
                    wait = 2 ** attempt
                    logger.info(f"{wait}秒後にリトライ...")
                    time.sleep(wait)

    except ImportError:
        errors.append("huggingface_hub がインストールされていません")

    # ── 全て失敗 ──
    if dest_path.exists():
        dest_path.unlink()

    error_detail = "\n".join(errors)
    manual_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    msg = (
        f"自動ダウンロードに失敗しました。\n\n"
        f"【エラー詳細】\n{error_detail}\n\n"
        f"【手動ダウンロード方法】\n"
        f"1. ブラウザで以下のURLを開いてダウンロード：\n"
        f"   {manual_url}\n"
        f"2. ダウンロードしたファイルを以下のフォルダに配置：\n"
        f"   {model_dir}\n"
        f"3. アプリを再起動\n\n"
        f"【その他の対処法】\n"
        f"- セキュリティソフトを一時停止してリトライ\n"
        f"- 設定画面で「HFミラーURL」を設定\n"
        f"  （例: https://hf-mirror.com）"
    )
    return False, msg


def download_hf_model(repo_id: str, filename: str, display_name: str) -> str | None:
    """
    任意のHuggingFaceリポジトリからGGUFをダウンロードしてカスタムモデルとして登録する。

    Returns:
        成功時は表示名（キー）、失敗時はNone
    """
    from core.config import LLM_MODELS, MODELS_DIR

    model_dir = MODELS_DIR / "llm"
    model_dir.mkdir(parents=True, exist_ok=True)
    dest_path = model_dir / filename

    try:
        from huggingface_hub import hf_hub_download
        logger.info(f"{repo_id}/{filename} をダウンロード中...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )

        key = f"{display_name}（カスタム）"
        LLM_MODELS[key] = {
            "repo_id": repo_id,
            "filename": filename,
            "size_gb": round(dest_path.stat().st_size / (1024 ** 3), 1),
            "min_ram_gb": 8,
            "speed": "不明",
            "description": f"HF: {repo_id}/{filename}",
            "chat_template": "chatml",
        }
        _save_custom_models()
        return key

    except Exception as e:
        logger.error(f"ダウンロードエラー: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return None


def _save_custom_models() -> None:
    """カスタムモデル定義を保存する（downloader内部用）"""
    from core.config import LLM_MODELS, MODELS_DIR
    custom_file = MODELS_DIR / "custom_models.json"
    custom = {
        name: info
        for name, info in LLM_MODELS.items()
        if "（カスタム）" in name
    }
    custom_file.parent.mkdir(parents=True, exist_ok=True)
    with open(custom_file, "w", encoding="utf-8") as f:
        json.dump(custom, f, ensure_ascii=False, indent=2)


# ─── ジョブ状態 ───────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING      = "pending"       # キュー待ち
    DOWNLOADING  = "downloading"   # ダウンロード中
    DONE         = "done"          # 完了
    FAILED       = "failed"        # 失敗
    CANCELLED    = "cancelled"     # キャンセル


STATUS_LABEL: dict[JobStatus, str] = {
    JobStatus.PENDING:     "⏳ 待機中",
    JobStatus.DOWNLOADING: "⬇️ ダウンロード中",
    JobStatus.DONE:        "✅ 完了",
    JobStatus.FAILED:      "❌ 失敗",
    JobStatus.CANCELLED:   "🚫 キャンセル",
}


# ─── ジョブデータクラス ───────────────────────────────────────

@dataclass
class DownloadJob:
    model_name: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0           # 0.0〜1.0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_mbps: float = 0.0
    error_msg: str = ""
    added_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    # ── 計算プロパティ ─────────────────────────

    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def eta_seconds(self) -> Optional[float]:
        """推定残り時間（秒）。計算不能な場合はNone"""
        if self.progress <= 0.01 or self.progress >= 1.0:
            return None
        elapsed = self.elapsed_seconds()
        if elapsed <= 0:
            return None
        return elapsed / self.progress * (1.0 - self.progress)

    def progress_label(self) -> str:
        """進捗を人間が読みやすい文字列で返す"""
        pct = int(self.progress * 100)
        mb = self.downloaded_bytes / (1024 ** 2)
        total_mb = self.total_bytes / (1024 ** 2)

        if self.total_bytes > 0:
            size_str = f"{mb:.0f}MB / {total_mb:.0f}MB"
        else:
            size_str = f"{mb:.0f}MB"

        speed_str = f"  {self.speed_mbps:.1f}MB/s" if self.speed_mbps > 0 else ""
        eta = self.eta_seconds()
        eta_str = ""
        if eta is not None:
            if eta >= 60:
                eta_str = f"  残り約{int(eta // 60)}分"
            else:
                eta_str = f"  残り約{int(eta)}秒"

        return f"{pct}%  ({size_str}){speed_str}{eta_str}"


# ─── ダウンロードマネージャー（シングルトン）─────────────────

class DownloadManager:
    """
    シングルトン・ダウンロードキューマネージャー。
    バックグラウンドスレッドで順次ダウンロードを処理する。
    """

    _instance: Optional["DownloadManager"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "DownloadManager":
        """シングルトンインスタンスを返す。
        Streamlit のホットリロードでモジュールが再ロードされても
        sys.modules 経由で同じインスタンスを返し、ダウンロード状態を保持する。
        """
        import sys
        with cls._instance_lock:
            # まず sys.modules から既存インスタンスを復元する
            existing = sys.modules.get(_DM_PERSIST_KEY)
            if existing is not None and hasattr(existing, '_jobs') and hasattr(existing, '_lock'):
                cls._instance = existing
                return cls._instance
            # なければ新規作成して sys.modules に保存
            if cls._instance is None:
                cls._instance = cls()
            sys.modules[_DM_PERSIST_KEY] = cls._instance
        return cls._instance

    def __init__(self) -> None:
        self._jobs: list[DownloadJob] = []
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None

    # ── キュー操作 ─────────────────────────────────────────────

    def add(self, model_name: str) -> bool:
        """
        モデルをキューに追加する。
        既にPENDINGまたはDOWNLOADING中の場合は無視。

        Returns:
            追加できた場合True
        """
        with self._lock:
            for job in self._jobs:
                if job.model_name == model_name and job.status in (
                    JobStatus.PENDING, JobStatus.DOWNLOADING
                ):
                    return False  # 既にキュー中
            job = DownloadJob(model_name=model_name)
            self._jobs.append(job)

        self._ensure_worker()
        return True

    def cancel(self, model_name: str) -> bool:
        """PENDING状態のジョブをキャンセルする"""
        with self._lock:
            for job in self._jobs:
                if job.model_name == model_name and job.status == JobStatus.PENDING:
                    job.status = JobStatus.CANCELLED
                    job.finished_at = datetime.now()
                    return True
        return False

    def retry(self, model_name: str) -> bool:
        """FAILED/CANCELLED状態のジョブを再キューに入れる"""
        with self._lock:
            for job in self._jobs:
                if job.model_name == model_name and job.status in (
                    JobStatus.FAILED, JobStatus.CANCELLED
                ):
                    job.status = JobStatus.PENDING
                    job.progress = 0.0
                    job.downloaded_bytes = 0
                    job.speed_mbps = 0.0
                    job.error_msg = ""
                    job.started_at = None
                    job.finished_at = None
                    break
            else:
                # 新規追加
                self._jobs.append(DownloadJob(model_name=model_name))

        self._ensure_worker()
        return True

    def remove(self, model_name: str) -> bool:
        """完了・失敗・キャンセル済みのジョブをリストから削除する"""
        with self._lock:
            before = len(self._jobs)
            self._jobs = [
                j for j in self._jobs
                if not (j.model_name == model_name and j.status in (
                    JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED
                ))
            ]
            return len(self._jobs) < before

    def clear_finished(self) -> None:
        """完了・失敗・キャンセル済みのジョブを一括削除する"""
        with self._lock:
            self._jobs = [
                j for j in self._jobs
                if j.status in (JobStatus.PENDING, JobStatus.DOWNLOADING)
            ]

    def get_jobs(self) -> list[DownloadJob]:
        """ジョブ一覧のスナップショットを返す（スレッドセーフ）"""
        with self._lock:
            return list(self._jobs)

    def get_job(self, model_name: str) -> Optional[DownloadJob]:
        """指定モデルのジョブを返す。なければNone"""
        with self._lock:
            for job in self._jobs:
                if job.model_name == model_name:
                    return job
        return None

    def is_active(self) -> bool:
        """PENDING or DOWNLOADING のジョブが存在するか"""
        with self._lock:
            return any(
                j.status in (JobStatus.PENDING, JobStatus.DOWNLOADING)
                for j in self._jobs
            )

    def queue_position(self, model_name: str) -> Optional[int]:
        """キュー内の位置（1始まり）を返す。存在しなければNone"""
        with self._lock:
            pending = [
                j for j in self._jobs if j.status == JobStatus.PENDING
            ]
            for i, j in enumerate(pending, 1):
                if j.model_name == model_name:
                    return i
        return None

    # ── ワーカースレッド ───────────────────────────────────────

    def _ensure_worker(self) -> None:
        """ワーカースレッドが動いていなければ起動する"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(
                target=self._worker,
                daemon=True,
                name="monoshiri-dl-worker",
            )
            self._worker_thread.start()

    def _worker(self) -> None:
        """キューを順次処理するワーカースレッド本体"""
        logger.info("DownloadManager: ワーカースレッド開始")
        while True:
            job = self._next_pending()
            if job is None:
                logger.info("DownloadManager: キュー空。ワーカー終了")
                break
            self._run_job(job)

    def _next_pending(self) -> Optional[DownloadJob]:
        """次のPENDINGジョブを返す"""
        with self._lock:
            for job in self._jobs:
                if job.status == JobStatus.PENDING:
                    return job
        return None

    def _run_job(self, job: DownloadJob) -> None:
        """1件のダウンロードを実行する"""
        from core.config import LLM_MODELS, MODELS_DIR

        # ── ジョブ開始 ──
        with self._lock:
            job.status = JobStatus.DOWNLOADING
            job.started_at = datetime.now()

        model_info = LLM_MODELS.get(job.model_name, {})
        size_gb = model_info.get("size_gb", 0)
        expected_bytes = int(size_gb * 1024 ** 3)
        filename = model_info.get("filename", "")
        dest_path = MODELS_DIR / "llm" / filename

        with self._lock:
            job.total_bytes = expected_bytes

        # ── ファイルサイズ監視スレッド起動 ──
        watcher_stop = threading.Event()
        watcher = threading.Thread(
            target=self._watch_file_size,
            args=(dest_path, expected_bytes, job, watcher_stop),
            daemon=True,
            name="monoshiri-dl-watcher",
        )
        watcher.start()

        # ── urllib直接DL用コールバック ──
        def progress_cb(value: float) -> None:
            with self._lock:
                job.progress = max(job.progress, min(value, 0.99))
                elapsed = job.elapsed_seconds()
                if elapsed > 0 and expected_bytes > 0:
                    job.downloaded_bytes = int(value * expected_bytes)
                    job.speed_mbps = round(
                        job.downloaded_bytes / elapsed / (1024 ** 2), 2
                    )

        # ── ダウンロード実行 ──
        try:
            logger.info(f"DownloadManager: ダウンロード開始 → {job.model_name}")
            success, msg = download_model(job.model_name, progress_callback=progress_cb)

            watcher_stop.set()
            watcher.join(timeout=2)

            with self._lock:
                job.finished_at = datetime.now()
                if success:
                    job.status = JobStatus.DONE
                    job.progress = 1.0
                    job.downloaded_bytes = expected_bytes if expected_bytes > 0 else job.downloaded_bytes
                    logger.info(f"DownloadManager: 完了 → {job.model_name}")
                else:
                    job.status = JobStatus.FAILED
                    job.error_msg = msg
                    logger.error(f"DownloadManager: 失敗 → {job.model_name}")

        except Exception as e:
            watcher_stop.set()
            watcher.join(timeout=2)
            with self._lock:
                job.status = JobStatus.FAILED
                job.error_msg = str(e)
                job.finished_at = datetime.now()
            logger.exception(f"DownloadManager: 例外 → {job.model_name}: {e}")

    def _watch_file_size(
        self,
        dest_path: Path,
        expected_bytes: int,
        job: DownloadJob,
        stop_event: threading.Event,
    ) -> None:
        """
        ダウンロード中のファイルサイズを定期監視して進捗を更新する。
        huggingface_hub は .incomplete 拡張子で一時ファイルを作成する。
        """
        if expected_bytes <= 0:
            return

        last_size = 0
        last_time = time.monotonic()

        while not stop_event.wait(0.5):
            # 一時ファイル候補
            # - urllib 使用時: .gguf.tmp  (download_with_urllib の tmp_path)
            # - huggingface_hub 使用時: .gguf.incomplete / stem.incomplete
            candidates = [
                dest_path.with_suffix(dest_path.suffix + ".tmp"),   # urllib
                dest_path.parent / (dest_path.name + ".incomplete"), # hf_hub (old)
                dest_path.parent / (dest_path.stem + ".incomplete"), # hf_hub (alt)
                dest_path,
            ]
            for path in candidates:
                if path.exists():
                    try:
                        size = path.stat().st_size
                        now = time.monotonic()
                        dt = now - last_time

                        if dt >= 0.5 and size >= last_size:
                            with self._lock:
                                job.downloaded_bytes = size
                                job.progress = min(
                                    max(job.progress, size / expected_bytes),
                                    0.99,
                                )
                                if dt > 0 and size > last_size:
                                    job.speed_mbps = round(
                                        (size - last_size) / dt / (1024 ** 2), 2
                                    )
                            last_size = size
                            last_time = now
                    except OSError:
                        pass
                    break
