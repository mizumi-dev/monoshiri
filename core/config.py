"""
モノシリ 設定管理モジュール
パス・デフォルト値・モデル定義などのアプリ全体設定を管理する
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

# アプリケーションルートディレクトリ
APP_DIR = Path(__file__).parent.parent

# データ保存先
# 常に ~/.monoshiri を使用する（全OS共通）
# 理由: Windowsでの %APPDATA% 使用時に既存データ (~/.monoshiri) と分離してしまう問題を解消。
#       旧バージョンのデータ (chroma/, hashes/) もこのパスに存在する。
def _get_data_dir() -> Path:
    return Path.home() / ".monoshiri"

DATA_DIR = _get_data_dir()
CHROMA_DIR = DATA_DIR / "chroma"         # 旧バージョンの実データに合わせて "chroma" に統一
CACHE_DIR  = DATA_DIR / "cache"
LOGS_DIR   = DATA_DIR / "logs"
MODELS_DIR = DATA_DIR / "models"
HISTORY_FILE      = DATA_DIR / "history.json"
CONFIG_FILE       = DATA_DIR / "config.json"
SKIP_LOG_FILE     = LOGS_DIR / "skip_log.txt"   # v3.4: logs/skip_log.txt
HASH_MANIFEST_FILE = CACHE_DIR / ".hash_manifest"  # v3.4: cache/.hash_manifest（単一ファイル）
HASH_DIR          = CACHE_DIR / "hashes"          # フォルダ別ハッシュファイルの格納ディレクトリ
USAGE_FILE        = DATA_DIR / "usage.json"      # Free層使用量トラッキング

# 初回起動時にディレクトリを作成
for _d in [DATA_DIR, CHROMA_DIR, CACHE_DIR, LOGS_DIR, MODELS_DIR, HASH_DIR]:
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
    "Qwen2.5-14B-Instruct Q4_K_M（高精度・推奨）": {
        "repo_id": "bartowski/Qwen2.5-14B-Instruct-GGUF",
        "filename": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "size_gb": 8.9,
        "min_ram_gb": 32,
        "speed": "やや遅い",
        "description": "VRAM 8GB以上推奨（16GB以上で高速）。7Bより大幅に精度が高くハルシネーション少。",
        "chat_template": "chatml",
        "ollama_name": "monoshiri-qwen25-14b",
    },
    "Qwen2.5-7B-Instruct Q4_K_M（標準・推奨）": {
        "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.7,
        "min_ram_gb": 16,
        "speed": "普通",
        "description": "製造業ドキュメントに強い。VRAM 6GB以上で快適動作。",
        "chat_template": "chatml",
        "ollama_name": "monoshiri-qwen25-7b",
    },
    "Qwen2.5-3B-Instruct Q4_K_M（軽量）": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_gb": 2.0,
        "min_ram_gb": 8,
        "speed": "速い",
        "description": "RAM 8GB以上のPCで動作。低スペック向け。",
        "chat_template": "chatml",
        "ollama_name": "monoshiri-qwen25-3b",
    },
    "Qwen3.5-9B Q4_K_M（高性能・最新）": {
        "repo_id": "unsloth/Qwen3.5-9B-GGUF",
        "filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "size_gb": 5.7,
        "min_ram_gb": 16,
        "speed": "普通",
        "description": "Qwen最新世代。テキスト専用。多言語・RAG精度が大幅向上。",
        "chat_template": "chatml",
        "ollama_name": "monoshiri-qwen35-9b",
    },
    "Nemotron-Nano-9B-v2-Japanese Q4_K_M（日本語特化）": {
        "repo_id": "mmnga-o/NVIDIA-Nemotron-Nano-9B-v2-Japanese-gguf",
        "filename": "NVIDIA-Nemotron-Nano-9B-v2-Japanese-Q4_K_M.gguf",
        "size_gb": 6.5,
        "min_ram_gb": 16,
        "speed": "普通",
        "description": "NVIDIA製。日本語特化・128Kコンテキスト。製造業日本語資料に最適。",
        "chat_template": "chatml",
        "ollama_name": "monoshiri-nemotron-9b",
    },
}

# Ollama設定
OLLAMA_BASE_URL = "http://localhost:11434"

# チャンク設定
# 日本語は1トークン≒2文字のため、500〜800トークン ≒ 1000〜1600文字
CHUNK_SIZE = 1200       # 文字数
CHUNK_OVERLAP = 500     # オーバーラップ文字数（300→500: CHUNK-001 テーブル行境界分割の緩和）

# ChromaDB コレクション名
CHROMA_COLLECTION = "monoshiri_docs"

# ─── モデル別推論パラメータ ──────────────────────────────────────
# SPEED修正: MAX_TOKENSを512に削減（VRAM 8GB最適化）
# 社内文書RAGの典型回答は100〜300トークン。512で十分かつ生成速度が倍近くなる。
# 1024は「長い説明が必要な質問」にのみ使用。
LLM_MODEL_MAX_TOKENS: dict[str, int] = {
    "Qwen2.5-14B-Instruct Q4_K_M（高精度・推奨）": 512,
    "Qwen2.5-7B-Instruct Q4_K_M（標準・推奨）": 512,   # SPEED推奨モデル
    "Qwen2.5-3B-Instruct Q4_K_M（軽量）": 384,
    "Qwen3.5-9B Q4_K_M（高性能・最新）": 512,
    "Nemotron-Nano-9B-v2-Japanese Q4_K_M（日本語特化）": 512,
}

# ─── Free層制限 ──────────────────────────────────────────────────
# Phase 1 β期間中は上限到達時に通知のみ（課金要求なし）
FREE_MAX_FILES     = 500   # インデックス済みファイル上限
FREE_MAX_QUESTIONS = 300   # 月間質問数上限（暦月リセット）

# ─── サンプル質問（固定3件・v3.4確定）────────────────────────────
SAMPLE_QUESTIONS: list[str] = [
    "この資料で一番よく出てくるキーワードは？",
    "生産手順について教えて",
    "品質基準について説明して",
]

# デフォルト類似度閾値
# ADR-002修正: 0.35 → 0.40 に引き上げ
#   フォールバック廃止に伴い、再現率より精度を優先する設計に転換。
#   閾値以下は「見つかりません」で返す。ノイズチャンクを排除してハルシネーション抑制。
DEFAULT_SIMILARITY = 0.40

# エビデンス最大表示件数
MAX_EVIDENCE = 10

# ベクトル検索上限
# ADR-002修正: 30 → 15 に削減
#   フォールバック廃止によりノイズ対策が不要になった。
#   チャンク数削減でLLMへのコンテキスト品質向上 & 推論速度改善。
TOP_K = 8   # SPEED-007: 15→8（チャンク品質向上により件数削減でもコンテキスト品質を維持）

# LLM回答生成の最大トークン数（デフォルト）
# ADR-002修正: モデル別設定（LLM_MODEL_MAX_TOKENS）を優先使用。
#   このデフォルト値はカスタムモデルおよびフォールバック用。
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
    "Qwen2.5-14B-Instruct-Q4_K_M.gguf":
        "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
    "Qwen2.5-7B-Instruct-Q4_K_M.gguf":
        "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "qwen2.5-3b-instruct-q4_k_m.gguf":
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
    "Qwen3.5-9B-Q4_K_M.gguf":
        "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
    "NVIDIA-Nemotron-Nano-9B-v2-Japanese-Q4_K_M.gguf":
        "https://huggingface.co/mmnga-o/NVIDIA-Nemotron-Nano-9B-v2-Japanese-gguf/resolve/main/NVIDIA-Nemotron-Nano-9B-v2-Japanese-Q4_K_M.gguf",
}

# RAGコンテキストの最大文字数
# BUG-021修正: 6000 → 3000 に削減
#   Qwen3.5-9B が大きなコンテキストで thinking mode に入り、タイムアウトが発生する問題を解消。
#   3000文字 ≈ 1500トークン。システムプロンプト(600) + ユーザーメッセージ(300) + コンテキスト(1500)
#   合計 ~2400トークンで、Qwen3.5の実効コンテキスト長(4096)に余裕を持って収まる。
RAG_CONTEXT_MAX_LENGTH = 1500  # SPEED-006: 3000→1500（prefill高速化・n_ctx=2048に余裕を持たせる）

# ─── GPU設定 ──────────────────────────────────────────────────
# LLM推論のGPUオフロード層数
#   0  = CPU推論（デフォルト・GPU不要）
#  -1  = 全層をGPUにオフロード（VRAM十分な場合）
#  32  = 指定層数だけGPUを使用（VRAMが少ない場合に部分オフロード）
# ※有効にするにはCUDA対応版のllama-cpp-pythonが必要
LLM_N_GPU_LAYERS: int = 0

# Embeddingモデルのデバイス選択
#   "auto"  = CUDAが使えれば自動でGPUを使用（推奨）
#   "cuda"  = 強制GPU
#   "cpu"   = 強制CPU
EMBEDDING_DEVICE: str = "auto"


def load_config() -> dict:
    """設定を読み込む（存在しない場合はデフォルト値を返す）"""
    global HF_MIRROR_URL, LLM_N_GPU_LAYERS, EMBEDDING_DEVICE
    defaults = {
        "selected_model": "",
        "folders": [],
        "similarity_threshold": DEFAULT_SIMILARITY,
        "custom_models": {},
        "hf_mirror_url": "",
        "plan": "free",   # "free" | "standard" | "pro"
        "llm_n_gpu_layers": 0,
        "embedding_device": "auto",
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
            # GPU設定をモジュール変数に反映
            LLM_N_GPU_LAYERS = int(data.get("llm_n_gpu_layers", 0))
            EMBEDDING_DEVICE = data.get("embedding_device", "auto")
            return data
        except Exception:
            pass
    return defaults


def save_config(config: dict) -> None:
    """設定を保存する"""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
