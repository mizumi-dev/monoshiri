"""
core/config.py の単体テスト

仕様書 v4.0 の固定値と LLM_MODELS 5種定義の整合性を検証。
"""
from __future__ import annotations

import json

import pytest

import core.config as cfg


class TestFixedValues:
    """仕様書 v4.0 で固定されている定数"""

    def test_chunk_size_is_1200(self):
        assert cfg.CHUNK_SIZE == 1200

    def test_chunk_overlap_is_500(self):
        assert cfg.CHUNK_OVERLAP == 500

    def test_default_similarity_is_040(self):
        # ADR-002: 0.35 → 0.40
        assert cfg.DEFAULT_SIMILARITY == 0.40

    def test_top_k_is_8(self):
        # SPEED-007: 15 → 8
        assert cfg.TOP_K == 8

    def test_rag_context_max_length_is_1500(self):
        # SPEED-006: 3000 → 1500
        assert cfg.RAG_CONTEXT_MAX_LENGTH == 1500

    def test_free_limits(self):
        assert cfg.FREE_MAX_FILES == 500
        assert cfg.FREE_MAX_QUESTIONS == 300

    def test_sample_questions_has_3_items(self):
        assert len(cfg.SAMPLE_QUESTIONS) == 3
        assert all(isinstance(q, str) and q for q in cfg.SAMPLE_QUESTIONS)


class TestLLMModels:
    """プリセット LLM モデル定義の整合性"""

    def test_five_models_defined(self):
        assert len(cfg.LLM_MODELS) == 5

    def test_all_models_have_required_keys(self):
        required = {"repo_id", "filename", "size_gb", "min_ram_gb",
                    "speed", "description", "chat_template", "ollama_name"}
        for name, m in cfg.LLM_MODELS.items():
            missing = required - set(m.keys())
            assert not missing, f"{name} に欠落キー: {missing}"

    def test_all_chat_templates_are_chatml(self):
        for name, m in cfg.LLM_MODELS.items():
            assert m["chat_template"] == "chatml", f"{name} のテンプレートが chatml 以外"

    def test_ollama_names_start_with_monoshiri(self):
        for name, m in cfg.LLM_MODELS.items():
            assert m["ollama_name"].startswith("monoshiri-"), f"{name} の ollama_name"

    def test_all_models_have_max_tokens(self):
        for name in cfg.LLM_MODELS:
            assert name in cfg.LLM_MODEL_MAX_TOKENS, f"{name} の max_tokens 未定義"
            # ADR-002: 384〜512 の範囲
            assert 384 <= cfg.LLM_MODEL_MAX_TOKENS[name] <= 1024


class TestChatTemplate:
    def test_chatml_template_defined(self):
        assert "chatml" in cfg.CHAT_TEMPLATES
        t = cfg.CHAT_TEMPLATES["chatml"]
        assert "<|im_start|>" in t["system_prefix"]
        assert "<|im_end|>" in t["stop_tokens"]


class TestLoadSave:
    def test_load_config_default_when_no_file(self, isolate_data_paths):
        config = cfg.load_config()
        assert config["selected_model"] == ""
        assert config["folders"] == []
        assert config["similarity_threshold"] == cfg.DEFAULT_SIMILARITY
        assert config["plan"] == "free"
        assert config["embedding_device"] == "auto"

    def test_save_and_load_roundtrip(self, isolate_data_paths):
        payload = {
            "selected_model": "monoshiri-qwen25-7b",
            "folders": ["C:/docs"],
            "similarity_threshold": 0.5,
            "custom_models": {},
            "hf_mirror_url": "https://hf-mirror.com",
            "plan": "standard",
            "llm_n_gpu_layers": -1,
            "embedding_device": "cuda",
        }
        cfg.save_config(payload)
        assert cfg.CONFIG_FILE.exists()
        loaded = cfg.load_config()
        assert loaded["selected_model"] == "monoshiri-qwen25-7b"
        assert loaded["hf_mirror_url"] == "https://hf-mirror.com"
        assert loaded["llm_n_gpu_layers"] == -1

    def test_ollama_base_url_is_localhost(self):
        # ADR-001: Ollama HTTP API
        assert cfg.OLLAMA_BASE_URL == "http://localhost:11434"


class TestDataDirectoryCreation:
    def test_data_dirs_exist_after_import(self, isolate_data_paths):
        # conftest で monkeypatch + mkdir 済
        assert cfg.DATA_DIR.exists()
        assert cfg.CHROMA_DIR.exists()
        assert cfg.CACHE_DIR.exists()
        assert cfg.LOGS_DIR.exists()
