"""
core/llm.py の結合テスト（要 Ollama）
"""
from __future__ import annotations

import pytest

from core import llm


pytestmark = pytest.mark.ollama


class TestOllamaAPI:
    def test_check_ollama_running(self):
        assert llm.check_ollama_running() is True

    def test_list_ollama_models_not_empty(self):
        models = llm.list_ollama_models()
        assert isinstance(models, list)
        assert len(models) > 0


class TestModelRecommendation:
    def test_recommend_model_returns_string(self):
        name = llm.recommend_model()
        assert isinstance(name, str)
        assert len(name) > 0


@pytest.mark.slow
class TestGenerateAnswer:
    def test_generate_short_answer(self, ollama_model):
        """実 LLM で短い回答を生成"""
        answer = llm.generate_answer(
            model_name=ollama_model,
            question="日本の首都はどこですか？",
            context="【地理情報】\n日本の首都は東京です。",
            chat_history=None,
        )
        assert isinstance(answer, str)
        assert len(answer) > 0
        # "東京" という単語が含まれることを期待（モデル依存だが強い信号）
        assert "東京" in answer or "Tokyo" in answer.lower()

    def test_generate_with_low_max_tokens(self, ollama_model):
        """max_tokens を小さくしても動くこと"""
        answer = llm.generate_answer(
            model_name=ollama_model,
            question="短く答えて",
            context="【テスト】\nこれはテストです。",
            chat_history=None,
            max_tokens=50,
        )
        assert isinstance(answer, str)


@pytest.mark.slow
class TestWarmup:
    def test_warmup_returns_bool(self, ollama_model):
        result = llm.warmup_model(ollama_model)
        assert isinstance(result, bool)
