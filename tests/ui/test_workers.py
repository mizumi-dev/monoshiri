"""
ui_qt/workers.py の結合テスト（pytest-qt）

各ワーカーがコールバックを発火することを確認する。
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.ui


def _run_worker(qtbot, worker_instance, timeout_ms=8000):
    """ワーカー開始 → コールバック発火まで待つ。結果を返す"""
    results = []
    # 可変長引数に対応（CudaInfoWorker は (text, style) を返す）
    worker_instance._callback = lambda *args: results.append(args if len(args) > 1 else args[0])
    worker_instance.start()

    def _done():
        return len(results) > 0

    qtbot.waitUntil(_done, timeout=timeout_ms)
    return results[0]


class TestStatsWorker:
    def test_returns_string(self, qtbot):
        from ui_qt.workers import StatsWorker
        result = _run_worker(qtbot, StatsWorker(config={}))
        assert isinstance(result, str)
        assert "チャンク" in result or "エラー" in result


class TestOllamaStatusWorker:
    @pytest.mark.ollama
    def test_returns_bool(self, qtbot):
        from ui_qt.workers import OllamaStatusWorker
        result = _run_worker(qtbot, OllamaStatusWorker())
        assert isinstance(result, bool)


class TestIndexStatsWorker:
    def test_returns_formatted_string(self, qtbot):
        from ui_qt.workers import IndexStatsWorker
        result = _run_worker(qtbot, IndexStatsWorker())
        assert isinstance(result, str)


class TestInstalledModelsWorker:
    def test_returns_list(self, qtbot):
        from ui_qt.workers import InstalledModelsWorker
        result = _run_worker(qtbot, InstalledModelsWorker(config={}))
        assert isinstance(result, list)


class TestRecommendedModelsWorker:
    def test_returns_list(self, qtbot):
        from ui_qt.workers import RecommendedModelsWorker
        result = _run_worker(qtbot, RecommendedModelsWorker(config={}))
        assert isinstance(result, list)
        assert len(result) >= 1


class TestMemInfoWorker:
    def test_returns_string(self, qtbot):
        from ui_qt.workers import MemInfoWorker
        result = _run_worker(qtbot, MemInfoWorker())
        assert isinstance(result, str)
        assert "GB" in result or "MB" in result


class TestCudaInfoWorker:
    @pytest.mark.gpu
    def test_returns_cuda_info(self, qtbot):
        """CUDA 環境では GPU 情報と style が返る"""
        from ui_qt.workers import CudaInfoWorker
        result = _run_worker(qtbot, CudaInfoWorker())
        # (text, style) タプル
        assert isinstance(result, tuple)
        text, style = result
        assert isinstance(text, str)
        # CUDA 環境なので "CUDA" または "GPU" が含まれることを期待
        assert "GPU" in text or "CUDA" in text or "未検出" in text


class TestOllamaCheckWorker:
    @pytest.mark.ollama
    def test_returns_running_and_list(self, qtbot):
        from ui_qt.workers import OllamaCheckWorker
        result = _run_worker(qtbot, OllamaCheckWorker())
        # (running: bool, reg_list: list)
        assert isinstance(result, tuple)
        running, reg_list = result
        assert isinstance(running, bool)
        assert isinstance(reg_list, list)


class TestNetworkCheckWorker:
    def test_returns_ok_and_msg(self, qtbot):
        from ui_qt.workers import NetworkCheckWorker
        result = _run_worker(qtbot, NetworkCheckWorker(), timeout_ms=15000)
        # (ok: bool, msg: str)
        assert isinstance(result, tuple)
        ok, msg = result
        assert isinstance(ok, bool)
        assert isinstance(msg, str)
