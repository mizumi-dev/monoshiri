"""
モノシリ 設定管理モジュール
パス・デフォルト値・モデル定義などのアプリ全体設定を管理する
"""
from __future__ import annotations
import json
from pathlib import Path

# アプリケーションルートディレクトリ
APP_DIR = Path(__file__).parent.parent

# データ保存先（ユーザーホームディレクトリ）
DATA_DIR = Path.home() / ".monoshiri"
CHROMA_DIR = DATA_DIR / "chroma"
HASH_DIR = DATA_DIR / "hashes"
MODELS_DIR = DATA_DIR / "models"
HISTORY_FILE = DATA_DIR / "history.json"
CONFIG_FILE = DATA_DIR / "config.json"
SKIP_LOG_FILE = DATA_DIR / "skip_log.json"

# 初回起動時にディレクトリを作成
for _d in [DATA_DIR, CHROMA_DIR, HASH_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Embeddingモデル設定（常時ローカル・外部送信なし）
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-large"
EMBEDDING_MODEL_DIR = MODELS_DIR / "embedding" / "multilingual-e5-large"

# ─── チャットテンプレート定義 ──────────────────────────────────
# 各モデルの "chat_template" キーで指定する
CHAT_TEMPLATES = {
    "chatml": {
        # Qwen2.5 / Qwen3.5 / Nemotron-Nano-Japanese 共通
        "system_prefix": "<|im_start|>system\n",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "assistant_prefix": "<|im_start|>assistant\n",
        "stop_tokens": ["<|im_end|>", "<|im_start|>"],
    },
}

# ローカルLLMモデル定義
LLM_MODELS: dict[str, dict] = {
    "Qwen2.5-7B-Instruct Q4_K_M（標準・推奨）": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "size_gb": 4.5,
        "min_ram_gb": 16,
        "speed": "普通",
        "description": "製造業ドキュメントに強い。精度重視の推奨モデル。",
        "chat_template": "chatml",
    },
    "Qwen2.5-3B-Instruct Q4_K_M（軽量）": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_gb": 2.0,
        "min_ram_gb": 8,
        "speed": "速い",
        "description": "RAM 8GB以上のPCで動作。低スペック向け。",
        "chat_template": "chatml",
    },
    "Qwen3.5-9B Q4_K_M（高性能・最新）": {
        "repo_id": "unsloth/Qwen3.5-9B-GGUF",
        "filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "size_gb": 5.7,
        "min_ram_gb": 16,
        "speed": "普通",
        "description": "Qwen最新世代。テキスト専用。多言語・RAG精度が大幅向上。",
        "chat_template": "chatml",
    },
    "Nemotron-Nano-9B-v2-Japanese Q4_K_M（日本語特化）": {
        "repo_id": "mmnga-o/NVIDIA-Nemotron-Nano-9B-v2-Japanese-gguf",
        "filename": "NVIDIA-Nemotron-Nano-9B-v2-Japanese-Q4_K_M.gguf",
        "size_gb": 6.5,
        "min_ram_gb": 16,
        "speed": "普通",
        "description": "NVIDIA製。日本語特化・128Kコンテキスト。製造業日本語資料に最適。",
        "chat_template": "chatml",
    },
}

# チャンク設定
# 日本語は1トークン≒2文字のため、500〜800トークン ≒ 1000〜1600文字
CHUNK_SIZE = 1200       # 文字数
CHUNK_OVERLAP = 200     # オーバーラップ文字数

# ChromaDB コレクション名
CHROMA_COLLECTION = "monoshiri_docs"

# デフォルト類似度閾値
DEFAULT_SIMILARITY = 0.5

# エビデンス最大表示件数
MAX_EVIDENCE = 10

# ベクトル検索上限
TOP_K = 20

# LLM回答生成の最大トークン数
LLM_MAX_TOKENS = 1024

# ─── ダウンロード設定 ──────────────────────────────────────────
# HuggingFaceダウンロードのリトライ回数
DOWNLOAD_MAX_RETRIES = 3

# ダウンロードタイムアウト（秒）
DOWNLOAD_TIMEOUT = 60

# HuggingFaceミラーURL（接続問題がある場合に設定画面から変更可能）
# 例: "https://hf-mirror.com"
HF_MIRROR_URL = ""

# GGUFモデルの直接ダウンロードURL（HuggingFace API経由でなく直接DLする場合）
GGUF_DIRECT_URLS: dict[str, str] = {
    "qwen2.5-7b-instruct-q4_k_m.gguf":
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
    "qwen2.5-3b-instruct-q4_k_m.gguf":
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
    "Qwen3.5-9B-Q4_K_M.gguf":
        "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
    "NVIDIA-Nemotron-Nano-9B-v2-Japanese-Q4_K_M.gguf":
        "https://huggingface.co/mmnga-o/NVIDIA-Nemotron-Nano-9B-v2-Japanese-gguf/resolve/main/NVIDIA-Nemotron-Nano-9B-v2-Japanese-Q4_K_M.gguf",
}

# RAGコンテキストの最大文字数
RAG_CONTEXT_MAX_LENGTH = 4000


def load_config() -> dict:
    """設定を読み込む（存在しない場合はデフォルト値を返す）"""
    global HF_MIRROR_URL
    defaults = {
        "selected_model": "",
        "folders": [],
        "similarity_threshold": DEFAULT_SIMILARITY,
        "custom_models": {},
        "hf_mirror_url": "",
    }
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # デフォルト値を補完
            for k, v in defaults.items():
                data.setdefault(k, v)
            # ミラーURLをモジュール変数に反映
            if data.get("hf_mirror_url"):
                HF_MIRROR_URL = data["hf_mirror_url"]
            return data
        except Exception:
            pass
    return defaults


def save_config(config: dict) -> None:
    """設定を保存する"""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_custom_models(config: dict) -> None:
    """カスタムモデルをLLM_MODELSに登録する"""
    for name, info in config.get("custom_models", {}).items():
        if name not in LLM_MODELS:
            LLM_MODELS[name] = info
