"""
モノシリ 共通バックグラウンドワーカー

全ウィジェットから使われるバックグラウンドタスクを一箇所に集約する。
daemon=True の threading.Thread で実行し、完了時に QApplication.postEvent()
でメインスレッドへ安全にコールバックする。

設計意図:
  QThread を使うと、run() 内の重い import（chromadb, torch 等）が GIL を
  長時間保持し、Qt のイベントループをブロックする問題がある。
  daemon スレッド + QApplication.postEvent() 方式は Qt 公式にスレッドセーフが
  保証されており、イベントループをブロックせず app.quit() も即座に効く。
"""
from __future__ import annotations

import logging
import threading
from functools import partial
from typing import Any, Callable

from PyQt6.QtCore import QObject, QEvent
from PyQt6.QtWidgets import QApplication

_log = logging.getLogger(__name__)


class _CallbackEvent(QEvent):
    """メインスレッドで実行するコールバックを運ぶカスタムイベント"""
    TYPE = QEvent.Type(QEvent.registerEventType())

    def __init__(self, fn: Callable):
        super().__init__(self.TYPE)
        self.fn = fn


class _Dispatcher(QObject):
    """QApplication.postEvent() でメインスレッドへコールバックを安全にディスパッチする"""

    def event(self, e: QEvent) -> bool:
        if isinstance(e, _CallbackEvent):
            try:
                e.fn()
            except Exception:
                _log.exception("ワーカーコールバック実行エラー")
            return True
        return super().event(e)


# モジュールレベルのシングルトン（QApplication 初期化後に有効）
_dispatcher: _Dispatcher | None = None


def _get_dispatcher() -> _Dispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = _Dispatcher()
    return _dispatcher


def _run_in_background(task: Callable[[], Any], callback: Callable[..., None]) -> None:
    """task を daemon スレッドで実行し、結果を callback でメインスレッドに返す

    QApplication.postEvent() は Qt 公式にスレッドセーフが保証されている。
    pyqtSignal.emit() と異なり daemon thread からの呼び出しでも
    GIL デッドロックのリスクがない。
    """
    disp = _get_dispatcher()

    def _worker():
        result = task()
        app = QApplication.instance()
        if app is not None:
            app.postEvent(disp, _CallbackEvent(partial(callback, result)))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


# ════════════════════════════════════════════════════════════════
# 以下、各ワーカーを公開関数として提供する。
# 呼び出し側は  worker_xxx(callback, ...)  の形で使う。
# ════════════════════════════════════════════════════════════════


# ── サイドバー統計（MainWindow） ──────────────────────────────

class StatsWorker:
    """ChromaDB 統計 + フォルダ数 + 選択モデル名を取得する"""
    def __init__(self, config: dict, parent=None):
        self._config = config
        self._callback = None

    def start(self):
        _run_in_background(self._task, self._on_done)

    def _task(self) -> str:
        try:
            from core.chromadb_store import get_index_stats
            from core.config import load_config
            chunk_count = get_index_stats().get("total_chunks", 0)
            folder_count = len(load_config().get("folders", []))
            model = self._config.get("selected_model", "未設定")
            short_model = model.split("（")[0] if model else "未設定"
            return (
                f"📊 {chunk_count:,} チャンク\n"
                f"📂 {folder_count} フォルダ\n"
                f"🤖 {short_model}"
            )
        except Exception:
            return "統計取得エラー"

    def _on_done(self, text: str):
        if self._callback:
            self._callback(text)


class OllamaStatusWorker:
    """Ollama HTTP API の起動確認"""
    def __init__(self, parent=None):
        self._callback = None

    def start(self):
        _run_in_background(self._task, self._on_done)

    def _task(self) -> bool:
        try:
            from core.llm import check_ollama_running
            return check_ollama_running()
        except Exception:
            return False

    def _on_done(self, running: bool):
        if self._callback:
            self._callback(running)


class IndexStatsWorker:
    """ChromaDB のチャンク数を取得する"""
    def __init__(self, fmt: str = "インデックス済み: {count:,} チャンク", parent=None):
        self._fmt = fmt
        self._callback = None

    def start(self):
        _run_in_background(self._task, self._on_done)

    def _task(self) -> str:
        try:
            from core.chromadb_store import get_index_stats
            count = get_index_stats().get("total_chunks", 0)
            return self._fmt.format(count=count)
        except Exception:
            return "統計取得エラー"

    def _on_done(self, text: str):
        if self._callback:
            self._callback(text)


class InstalledModelsWorker:
    """[(model_name, model_info, is_selected)] を返す"""
    def __init__(self, config: dict, parent=None):
        self._config = config
        self._callback = None

    def start(self):
        _run_in_background(self._task, self._on_done)

    def _task(self) -> list:
        try:
            from core.llm import is_model_downloaded
            from core.config import LLM_MODELS
            selected = self._config.get("selected_model", "")
            return [
                (n, info, n == selected)
                for n, info in LLM_MODELS.items()
                if is_model_downloaded(n)
            ]
        except Exception:
            return []

    def _on_done(self, result: list):
        if self._callback:
            self._callback(result)


class RecommendedModelsWorker:
    """[(model_name, model_info, is_recommended, is_queued)] を返す"""
    def __init__(self, config: dict, parent=None):
        self._config = config
        self._callback = None

    def start(self):
        _run_in_background(self._task, self._on_done)

    def _task(self) -> list:
        try:
            from core.llm import is_model_downloaded, recommend_model
            from core.config import LLM_MODELS
            from core.downloader import DownloadManager, JobStatus
            recommended = recommend_model()
            dm = DownloadManager.get()
            result = []
            for name, info in LLM_MODELS.items():
                if is_model_downloaded(name) or not info.get("repo_id"):
                    continue
                job = dm.get_job(name)
                queued = job is not None and job.status in (JobStatus.PENDING, JobStatus.DOWNLOADING)
                result.append((name, info, name == recommended, queued))
            return result
        except Exception:
            return []

    def _on_done(self, result: list):
        if self._callback:
            self._callback(result)


class MemInfoWorker:
    """総RAM / 空きRAM / 推奨モデルの文字列を返す"""
    def __init__(self, parent=None):
        self._callback = None

    def start(self):
        _run_in_background(self._task, self._on_done)

    def _task(self) -> str:
        try:
            from core.llm import get_total_ram_gb, get_available_ram_gb, recommend_model
            total = get_total_ram_gb()
            avail = get_available_ram_gb()
            rec = recommend_model()
            return (
                f"総 RAM: <b>{total:.1f} GB</b>　"
                f"空き RAM: <b>{avail:.1f} GB</b>　"
                f"推奨: <b>{rec or '—'}</b>"
            )
        except Exception as e:
            return f"取得エラー: {e}"

    def _on_done(self, text: str):
        if self._callback:
            self._callback(text)


class CudaInfoWorker:
    """GPU名・VRAM 情報の文字列とスタイルを返す（CUDA / DirectML 両対応）"""
    def __init__(self, parent=None):
        self._callback = None  # callback(text, style)

    def start(self):
        _run_in_background(self._task, self._on_done)

    def _task(self) -> tuple[str, str]:
        green = "font-size: 11px; color: #27ae60;"
        orange = "font-size: 11px; color: #e67e22;"
        gray = "font-size: 11px; color: #888;"
        # 1. CUDA（最速）
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                return (f"✅ GPU (CUDA): {name}（VRAM: {vram:.1f}GB）", green)
        except ImportError:
            return ("⚠️ PyTorch未インストール", gray)

        # 2. DirectML（Windows 全GPU 汎用）
        try:
            import torch_directml
            if torch_directml.is_available():
                gpu_name = "不明"
                try:
                    import subprocess
                    r = subprocess.run(
                        ["wmic", "path", "win32_VideoController", "get", "Name"],
                        capture_output=True, text=True, timeout=5,
                    )
                    lines = [ln.strip() for ln in r.stdout.splitlines()
                             if ln.strip() and ln.strip() != "Name"]
                    if lines:
                        gpu_name = lines[0]
                except Exception:
                    pass
                return (f"✅ GPU (DirectML): {gpu_name}（NVIDIA/AMD/Intel 汎用モード）", green)
        except ImportError:
            pass

        # 3. Apple Silicon
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return ("✅ GPU (MPS): Apple Silicon", green)
        except ImportError:
            pass

        return ("⚠️ GPUバックエンドが見つかりません（CPU動作）", orange)

    def _on_done(self, result: tuple[str, str]):
        if self._callback:
            self._callback(*result)


class OllamaCheckWorker:
    """Ollama 起動確認 + インストール済みモデルの登録状態"""
    def __init__(self, parent=None):
        self._callback = None  # callback(running, reg_list)

    def start(self):
        _run_in_background(self._task, self._on_done)

    def _task(self) -> tuple[bool, list]:
        try:
            from core.llm import check_ollama_running, is_model_downloaded, is_registered_with_ollama
            from core.config import LLM_MODELS
            running = check_ollama_running()
            reg_list = []
            if running:
                for n in LLM_MODELS:
                    if is_model_downloaded(n):
                        reg_list.append((n, is_registered_with_ollama(n)))
            return (running, reg_list)
        except Exception:
            return (False, [])

    def _on_done(self, result: tuple[bool, list]):
        if self._callback:
            self._callback(*result)


class OllamaRegisterWorker:
    """Ollama にモデルを登録する"""
    def __init__(self, model_name: str, parent=None):
        self.model_name = model_name
        self._callback = None  # callback(model_name, success, msg)

    def start(self):
        _run_in_background(self._task, self._on_done)

    def _task(self) -> tuple[str, bool, str]:
        try:
            from core.llm import register_model_with_ollama
            register_model_with_ollama(self.model_name)
            return (self.model_name, True, "登録完了")
        except Exception as e:
            return (self.model_name, False, str(e))

    def _on_done(self, result: tuple[str, bool, str]):
        if self._callback:
            self._callback(*result)


class NetworkCheckWorker:
    """ネットワーク接続テスト"""
    def __init__(self, parent=None):
        self._callback = None  # callback(ok, msg)

    def start(self):
        _run_in_background(self._task, self._on_done)

    def _task(self) -> tuple[bool, str]:
        try:
            from core.downloader import check_network_connectivity
            return check_network_connectivity()
        except Exception as e:
            return (False, str(e))

    def _on_done(self, result: tuple[bool, str]):
        if self._callback:
            self._callback(*result)
