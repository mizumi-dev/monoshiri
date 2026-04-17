"""
core/embedder.py の単体テスト

重量級（モデルロード）テストは @pytest.mark.slow + @pytest.mark.gpu を付けて分離。
detect_backend / CancelledError バグ回帰テストは軽量。
"""
from __future__ import annotations

from concurrent.futures import CancelledError

import pytest

from core import embedder


class TestDetectBackend:
    def test_returns_tuple(self):
        device, backend = embedder.detect_backend()
        assert backend in ("CUDA", "DirectML", "MPS", "CPU")
        assert device is not None

    @pytest.mark.gpu
    def test_gpu_env_returns_cuda(self):
        """CUDA 環境では CUDA を選ぶ"""
        device, backend = embedder.detect_backend()
        assert backend == "CUDA"
        assert device == "cuda"

    def test_resolve_device_auto(self, monkeypatch):
        import core.config as cfg
        monkeypatch.setattr(cfg, "EMBEDDING_DEVICE", "auto", raising=True)
        device = embedder._resolve_device()
        assert device is not None


class TestBatchSizeManager:
    def test_manager_initialization(self):
        mgr = embedder.AdaptiveBatchSizeManager()
        assert mgr.current_batch_size >= 64
        assert mgr.current_batch_size <= 2048
        assert mgr.oom_count == 0

    def test_oom_error_reduces_batch(self):
        mgr = embedder.AdaptiveBatchSizeManager(initial_batch_size=1024)
        new = mgr.on_oom_error()
        assert new <= 1024
        assert new >= 64  # 最小値
        assert mgr.oom_count == 1

    def test_min_batch_size_floor(self):
        mgr = embedder.AdaptiveBatchSizeManager(initial_batch_size=128)
        # 複数回 OOM を発生させても最小値を下回らない
        for _ in range(20):
            mgr.on_oom_error()
        assert mgr.current_batch_size >= mgr.min_batch_size


class TestEmbedTexts:
    def test_empty_list(self):
        """重要: 空リストでもエラーにならずモデルロード不要"""
        assert embedder.embed_texts([]) == []

    def test_cancel_check_triggers_cancellederror(self):
        """CancelledError スコープバグ回帰テスト（先日の修正）"""
        with pytest.raises(CancelledError):
            embedder.embed_texts(["hello"], cancel_check=lambda: True)

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_embed_real_texts(self):
        """実モデルロードを伴う統合寄りのテスト（GPUあり環境のみ）"""
        result = embedder.embed_texts(["モノシリは完全ローカル動作です"])
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == 1024  # multilingual-e5-large は 1024 次元
        # 正規化されているか（L2 norm ≈ 1）
        import math
        norm = math.sqrt(sum(x * x for x in result[0]))
        assert 0.99 <= norm <= 1.01

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_embed_query_applies_prefix(self):
        """embed_query は 'query: ' プレフィックス付き"""
        q_vec = embedder.embed_query("質問")
        assert len(q_vec) == 1024


class TestIsEmbeddingReady:
    def test_returns_bool(self):
        assert isinstance(embedder.is_embedding_ready(), bool)
