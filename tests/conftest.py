"""
モノシリ テスト共通設定

設計方針:
  - 本番データ (~/.monoshiri) を汚さないため、テスト用一時ディレクトリに各種パスを差し替える
  - core/config.py の DATA_DIR 等の定数を monkeypatch で上書きする（モジュール import 済のため直接差し替え）
  - チェックポイント・ハッシュ・ChromaDB もテスト分離する
  - Ollama / GPU を要するテストはマーカーで分類し、環境が無い場合は自動 skip
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

# リポジトリルートを sys.path に追加（core/ を直接 import できるように）
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ─── セッションスコープの一時データディレクトリ ────────────────────────
@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Path:
    """テストセッション全体で使う隔離データディレクトリ"""
    d = tmp_path_factory.mktemp("monoshiri_test_data")
    return d


@pytest.fixture(autouse=True)
def isolate_data_paths(monkeypatch, tmp_path):
    """
    テストごとに MONOSHIRI_DATA_DIR 環境変数で本番データを隔離する。
    core/config.py がこの環境変数を読み取ってパス系を切り替える。

    ただし core モジュールは既に import 済でモジュール変数がキャプチャされているため、
    config + 依存モジュールの変数を明示的に差し替える必要がある。
    """
    data_dir = tmp_path / "data"
    chroma_dir = data_dir / "chroma"
    cache_dir = data_dir / "cache"
    logs_dir = data_dir / "logs"
    models_dir = data_dir / "models"
    hash_dir = cache_dir / "hashes"

    for d in [data_dir, chroma_dir, cache_dir, logs_dir, models_dir, hash_dir]:
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("MONOSHIRI_DATA_DIR", str(data_dir))

    import core.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(cfg, "CHROMA_DIR", chroma_dir, raising=True)
    monkeypatch.setattr(cfg, "CACHE_DIR", cache_dir, raising=True)
    monkeypatch.setattr(cfg, "LOGS_DIR", logs_dir, raising=True)
    monkeypatch.setattr(cfg, "MODELS_DIR", models_dir, raising=True)
    monkeypatch.setattr(cfg, "HASH_DIR", hash_dir, raising=True)
    monkeypatch.setattr(cfg, "HASH_MANIFEST_FILE", cache_dir / ".hash_manifest", raising=True)
    monkeypatch.setattr(cfg, "SKIP_LOG_FILE", logs_dir / "skip_log.txt", raising=True)
    monkeypatch.setattr(cfg, "USAGE_FILE", data_dir / "usage.json", raising=True)
    monkeypatch.setattr(cfg, "HISTORY_FILE", data_dir / "history.json", raising=True)
    monkeypatch.setattr(cfg, "CONFIG_FILE", data_dir / "config.json", raising=True)

    # 依存モジュールが import 時にキャプチャした値も差し替え
    try:
        import core.hash_manager as hm
        monkeypatch.setattr(hm, "HASH_MANIFEST_FILE", cache_dir / ".hash_manifest", raising=True)
    except ImportError:
        pass
    try:
        import core.checkpoint_manager as cpm
        monkeypatch.setattr(cpm, "CHECKPOINT_FILE", data_dir / "chunk_checkpoint.json", raising=True)
        monkeypatch.setattr(cpm, "DATA_DIR", data_dir, raising=True)
    except ImportError:
        pass
    try:
        import core.usage_tracker as ut
        monkeypatch.setattr(ut, "USAGE_FILE", data_dir / "usage.json", raising=True)
    except ImportError:
        pass
    try:
        import core.chromadb_store as cs
        monkeypatch.setattr(cs, "CHROMA_DIR", chroma_dir, raising=True)
        cs._chroma_client_instance = None
    except ImportError:
        pass

    # ChromaDB の SharedSystemClient グローバルキャッシュもクリア
    # （複数の PersistentClient インスタンス間の tenant 汚染を防ぐ）
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass

    yield data_dir

    # 後始末: ChromaDB シングルトン + SharedSystemClient を必ずリセット
    try:
        import core.chromadb_store as cs
        cs._chroma_client_instance = None
    except ImportError:
        pass
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass


# ─── テストファイル生成用フィクスチャ ────────────────────────────────
@pytest.fixture
def sample_txt_file(tmp_path) -> Path:
    """短い日本語テキストファイル"""
    f = tmp_path / "sample.txt"
    f.write_text(
        "モノシリは社内資料探索AIです。\n"
        "完全ローカル動作で、資料は一切外に出ません。\n"
        "製造業の中小企業向けに設計されています。\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def sample_txt_folder(tmp_path) -> Path:
    """3 つの txt ファイルを含むフォルダ"""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("モノシリは社内資料探索AIです。", encoding="utf-8")
    (folder / "b.txt").write_text("ChromaDBをベクトルDBに使用しています。", encoding="utf-8")
    (folder / "c.txt").write_text("Ollamaで完全ローカルLLM推論を行います。", encoding="utf-8")
    return folder


# ─── Ollama / GPU マーカー制御 ───────────────────────────────────────
def pytest_collection_modifyitems(config, items):
    """Ollama 未起動 / GPU 無い環境では該当マーカーを skip"""
    # Ollama 起動状態を一度だけチェック
    ollama_ok = None
    gpu_ok = None

    for item in items:
        if "ollama" in item.keywords:
            if ollama_ok is None:
                try:
                    import urllib.request
                    urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
                    ollama_ok = True
                except Exception:
                    ollama_ok = False
            if not ollama_ok:
                item.add_marker(pytest.mark.skip(reason="Ollama 未起動"))

        if "gpu" in item.keywords:
            if gpu_ok is None:
                try:
                    import torch
                    gpu_ok = torch.cuda.is_available()
                except Exception:
                    gpu_ok = False
            if not gpu_ok:
                item.add_marker(pytest.mark.skip(reason="GPU (CUDA) 未検出"))
