"""
結合テスト共通フィクスチャ

LLM モデル自動選択（優先順位）:
  1. monoshiri-qwen25-7b（標準推奨）
  2. monoshiri-qwen25-14b
  3. monoshiri-nemotron-9b
  4. monoshiri-qwen35-9b
  5. monoshiri-qwen25-3b
"""
from __future__ import annotations

import pytest


_LLM_PRIORITY = [
    "monoshiri-qwen25-7b",
    "monoshiri-qwen25-14b",
    "monoshiri-nemotron-9b",
    "monoshiri-qwen35-9b",
    "monoshiri-qwen25-3b",
]


@pytest.fixture(scope="session")
def ollama_model() -> str:
    """
    Ollama 登録済みモデルから優先順位で LLM_MODELS 表示名を選択して返す。
    generate_answer() は LLM_MODELS の表示名（キー）を期待する。
    """
    try:
        from core.llm import list_ollama_models
        from core.config import LLM_MODELS
        available = set(list_ollama_models())
    except Exception:
        pytest.skip("Ollama 未起動またはモデル一覧取得失敗")

    # ollama_name → 表示名の逆引き
    rev_map = {m["ollama_name"]: name for name, m in LLM_MODELS.items()}

    for ollama_name in _LLM_PRIORITY:
        for a in available:
            if a == ollama_name or a.startswith(f"{ollama_name}:"):
                if ollama_name in rev_map:
                    return rev_map[ollama_name]  # 表示名を返す
    pytest.skip(f"優先モデルが Ollama に登録されていません (候補: {_LLM_PRIORITY})")


@pytest.fixture(scope="session")
def ollama_name() -> str:
    """Ollama の短い登録名（monoshiri-xxx 形式）を返す"""
    try:
        from core.llm import list_ollama_models
        available = set(list_ollama_models())
    except Exception:
        pytest.skip("Ollama 未起動")
    for m in _LLM_PRIORITY:
        for a in available:
            if a == m or a.startswith(f"{m}:"):
                return m
    pytest.skip(f"優先モデルが未登録")
