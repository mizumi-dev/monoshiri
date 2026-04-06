"""
モノシリ ダウンロードキューマネージャー

スレッドセーフな順次ダウンロード処理。
- 複数モデルをキューに積んで順次ダウンロード
- ファイルサイズ監視による進捗トラッキング
- キャンセル / リトライ対応
- Streamlit の @st.fragment(run_every=N) と組み合わせて UI に進捗を反映
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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
        """シングルトンインスタンスを返す"""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
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
        from core.llm import download_model
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
            # HuggingFace Hub の一時ファイル候補
            candidates = [
                dest_path.parent / (dest_path.name + ".incomplete"),
                dest_path.parent / (dest_path.stem + ".incomplete"),
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
