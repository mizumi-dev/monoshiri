"""UI テスト用 conftest（ollama_model fixture を提供）"""
from __future__ import annotations

import pytest


@pytest.fixture
def use_production_data(monkeypatch):
    """
    本番データ（~/.monoshiri）を使うため isolate_data_paths の差し替えを解除する。
    autouse な isolate_data_paths がセットアップ後、このフィクスチャで元に戻す。
    """
    # MONOSHIRI_DATA_DIR を解除
    monkeypatch.delenv("MONOSHIRI_DATA_DIR", raising=False)
    # core.config の定数を本番値に戻す
    from pathlib import Path
    import core.config as cfg
    prod_data = Path.home() / ".monoshiri"
    monkeypatch.setattr(cfg, "DATA_DIR", prod_data, raising=True)
    monkeypatch.setattr(cfg, "CHROMA_DIR", prod_data / "chroma", raising=True)
    monkeypatch.setattr(cfg, "CACHE_DIR", prod_data / "cache", raising=True)
    monkeypatch.setattr(cfg, "LOGS_DIR", prod_data / "logs", raising=True)
    monkeypatch.setattr(cfg, "MODELS_DIR", prod_data / "models", raising=True)
    monkeypatch.setattr(cfg, "HASH_DIR", prod_data / "cache" / "hashes", raising=True)
    monkeypatch.setattr(cfg, "HASH_MANIFEST_FILE", prod_data / "cache" / ".hash_manifest", raising=True)
    monkeypatch.setattr(cfg, "SKIP_LOG_FILE", prod_data / "logs" / "skip_log.txt", raising=True)
    monkeypatch.setattr(cfg, "USAGE_FILE", prod_data / "usage.json", raising=True)
    monkeypatch.setattr(cfg, "HISTORY_FILE", prod_data / "history.json", raising=True)
    monkeypatch.setattr(cfg, "CONFIG_FILE", prod_data / "config.json", raising=True)

    import core.chromadb_store as cs
    monkeypatch.setattr(cs, "CHROMA_DIR", prod_data / "chroma", raising=True)
    cs._chroma_client_instance = None
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass

    return prod_data


_LLM_PRIORITY = [
    "monoshiri-qwen25-7b",
    "monoshiri-qwen25-14b",
    "monoshiri-nemotron-9b",
    "monoshiri-qwen35-9b",
    "monoshiri-qwen25-3b",
]


@pytest.fixture(scope="session")
def ollama_model() -> str:
    """Ollama 登録済みモデルから優先順位で LLM_MODELS 表示名を返す"""
    try:
        from core.llm import list_ollama_models
        from core.config import LLM_MODELS
        available = set(list_ollama_models())
    except Exception:
        pytest.skip("Ollama 未起動またはモデル一覧取得失敗")

    rev_map = {m["ollama_name"]: name for name, m in LLM_MODELS.items()}
    for ollama_name in _LLM_PRIORITY:
        for a in available:
            if a == ollama_name or a.startswith(f"{ollama_name}:"):
                if ollama_name in rev_map:
                    return rev_map[ollama_name]
    pytest.skip(f"優先モデルが Ollama に登録されていません (候補: {_LLM_PRIORITY})")
