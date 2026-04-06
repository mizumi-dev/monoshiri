"""
モノシリ ローカルLLM管理モジュール
llama-cpp-python + Qwen2.5 GGUF モデルを使った完全ローカル推論。
外部送信なし・インターネット接続不要（モデルDL後）。
"""
from __future__ import annotations
import gc
import json
import logging
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

import psutil

from core.config import (
    MODELS_DIR, LLM_MODELS, LLM_MAX_TOKENS,
    DOWNLOAD_MAX_RETRIES, DOWNLOAD_TIMEOUT,
    HF_MIRROR_URL, GGUF_DIRECT_URLS,
    CHAT_TEMPLATES,
)

logger = logging.getLogger(__name__)

# グローバルLLMキャッシュ（プロセス内で1モデルのみ保持）
_llm = None
_loaded_model_name: str | None = None

# カスタムモデルの永続化ファイル
CUSTOM_MODELS_FILE = MODELS_DIR / "custom_models.json"


# ─── RAM / システム情報 ───────────────────────────────────────

def get_total_ram_gb() -> float:
    """総RAM容量（GB）を返す"""
    return psutil.virtual_memory().total / (1024 ** 3)


def get_available_ram_gb() -> float:
    """現在の空きRAM（GB）を返す"""
    return psutil.virtual_memory().available / (1024 ** 3)


def recommend_model() -> str:
    """
    総RAMに基づいて推奨モデルを返す。
    16GB以上 → Qwen2.5-7B（安定・推奨） / 8GB以上 → Qwen2.5-3B
    """
    total_ram = get_total_ram_gb()
    model_keys = list(LLM_MODELS.keys())

    # 16GB以上なら標準推奨モデル（先頭）、8GB以上なら軽量モデル
    if total_ram >= 16:
        # 標準推奨: Qwen2.5-7B
        for key in model_keys:
            if "標準" in key or "推奨" in key:
                return key
        return model_keys[0]
    else:
        # 軽量: Qwen2.5-3B
        for key in model_keys:
            if "軽量" in key:
                return key
        return model_keys[1] if len(model_keys) > 1 else model_keys[0]


# ─── モデルパス管理 ───────────────────────────────────────────

def get_model_path(model_name: str) -> Path | None:
    """
    モデルのローカルパスを返す。
    ダウンロードされていない場合はNone。
    """
    if model_name not in LLM_MODELS:
        return None
    filename = LLM_MODELS[model_name]["filename"]
    path = MODELS_DIR / "llm" / filename
    return path if path.exists() else None


def is_model_downloaded(model_name: str) -> bool:
    """モデルがダウンロード済みかどうかを返す"""
    return get_model_path(model_name) is not None


def get_installed_models() -> list[str]:
    """ダウンロード済みモデルの表示名一覧を返す"""
    installed = []
    for name in LLM_MODELS:
        if is_model_downloaded(name):
            installed.append(name)
    return installed


# ─── カスタムモデル管理 ───────────────────────────────────────

def load_custom_models() -> None:
    """カスタムモデル定義を読み込んでLLM_MODELSに追加する"""
    if not CUSTOM_MODELS_FILE.exists():
        return
    try:
        with open(CUSTOM_MODELS_FILE, "r", encoding="utf-8") as f:
            custom = json.load(f)
        for name, info in custom.items():
            if name not in LLM_MODELS:
                LLM_MODELS[name] = info
    except Exception as e:
        logger.warning(f"カスタムモデル読み込みエラー: {e}")


def save_custom_models() -> None:
    """カスタムモデル定義を保存する"""
    custom = {
        name: info
        for name, info in LLM_MODELS.items()
        if "（カスタム）" in name
    }
    CUSTOM_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CUSTOM_MODELS_FILE, "w", encoding="utf-8") as f:
        json.dump(custom, f, ensure_ascii=False, indent=2)


def add_local_gguf(gguf_path: Path, display_name: str | None = None) -> str:
    """
    ローカルGGUFファイルをモデルとして追加する（コピーして登録）。

    Returns:
        登録した表示名
    """
    model_dir = MODELS_DIR / "llm"
    model_dir.mkdir(parents=True, exist_ok=True)

    dest = model_dir / gguf_path.name
    if not dest.exists():
        import shutil
        shutil.copy2(str(gguf_path), str(dest))

    name = display_name or gguf_path.stem
    key = f"{name}（カスタム）"

    LLM_MODELS[key] = {
        "repo_id": None,
        "filename": gguf_path.name,
        "size_gb": round(dest.stat().st_size / (1024 ** 3), 1),
        "min_ram_gb": 8,
        "speed": "不明",
        "description": f"カスタムモデル: {gguf_path.name}",
    }
    save_custom_models()
    return key


# ─── モデルダウンロード ───────────────────────────────────────

def _check_network_connectivity() -> tuple[bool, str]:
    """
    ネットワーク接続を簡易チェックする。

    Returns:
        (接続可能か, メッセージ)
    """
    test_urls = [
        "https://huggingface.co",
        "https://hf-mirror.com",
    ]
    for url in test_urls:
        try:
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, timeout=10)
            return True, f"接続OK: {url}"
        except Exception:
            continue
    return False, (
        "HuggingFaceに接続できません。\n"
        "考えられる原因:\n"
        "  1. インターネット接続がない\n"
        "  2. ファイアウォール/セキュリティソフトがブロックしている\n"
        "  3. DNS解決に失敗している\n"
        "  4. プロキシ設定が必要\n"
        "\n対処法:\n"
        "  - ブラウザで https://huggingface.co にアクセスできるか確認\n"
        "  - セキュリティソフトの一時停止を試す\n"
        "  - 設定画面の「手動ダウンロード」からブラウザ経由でモデルを入手\n"
    )


def _download_with_urllib(url: str, dest_path: Path, progress_callback=None) -> bool:
    """
    urllib を使ってファイルを直接ダウンロードする（huggingface_hub不要のフォールバック）。

    Args:
        url: ダウンロード元URL
        dest_path: 保存先パス
        progress_callback: 進捗コールバック（0.0〜1.0）

    Returns:
        成功した場合True
    """
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    try:
        logger.info(f"直接ダウンロード開始: {url}")
        req = urllib.request.Request(url, headers={
            "User-Agent": "monoshiri/1.0"
        })
        response = urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT)

        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB

        with open(tmp_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total_size > 0:
                    progress_callback(downloaded / total_size)

        # ダウンロード完了、リネーム
        tmp_path.rename(dest_path)
        logger.info(f"直接ダウンロード完了: {dest_path}")
        return True

    except Exception as e:
        logger.error(f"直接ダウンロードエラー: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def download_model(
    model_name: str,
    progress_callback=None,
) -> tuple[bool, str]:
    """
    HuggingFaceからGGUFモデルをダウンロードする。
    リトライ・ミラー・直接ダウンロードの3段階フォールバック対応。

    Args:
        model_name: LLM_MODELSのキー
        progress_callback: 進捗コールバック（0.0〜1.0）

    Returns:
        (成功したか, メッセージ)
    """
    if model_name not in LLM_MODELS:
        return False, f"未定義のモデル: {model_name}"

    model_info = LLM_MODELS[model_name]
    repo_id = model_info.get("repo_id")
    filename = model_info["filename"]

    if not repo_id:
        return False, f"カスタムモデルはHFダウンロード非対応: {model_name}"

    model_dir = MODELS_DIR / "llm"
    model_dir.mkdir(parents=True, exist_ok=True)
    dest_path = model_dir / filename

    if dest_path.exists():
        return True, f"モデルは既にダウンロード済みです"

    # ── 方法1: huggingface_hub 経由（リトライ付き） ──
    errors = []
    try:
        from huggingface_hub import hf_hub_download

        for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
            try:
                logger.info(
                    f"[{attempt}/{DOWNLOAD_MAX_RETRIES}] "
                    f"huggingface_hub で {repo_id}/{filename} をダウンロード中..."
                )

                # ミラーが設定されている場合はendpointを切り替え
                kwargs = dict(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                )
                if HF_MIRROR_URL:
                    kwargs["endpoint"] = HF_MIRROR_URL

                hf_hub_download(**kwargs)
                return True, "ダウンロード完了"

            except Exception as e:
                err_msg = str(e)
                errors.append(f"試行{attempt}: {err_msg}")
                logger.warning(f"ダウンロード試行 {attempt} 失敗: {err_msg}")
                if attempt < DOWNLOAD_MAX_RETRIES:
                    wait = 2 ** attempt
                    logger.info(f"{wait}秒後にリトライ...")
                    time.sleep(wait)

    except ImportError:
        errors.append("huggingface_hub がインストールされていません")

    # ── 方法2: urllib で直接ダウンロード ──
    direct_url = GGUF_DIRECT_URLS.get(filename)
    if direct_url:
        logger.info("huggingface_hub 失敗。直接ダウンロードを試みます...")

        # ミラーURLに書き換え
        if HF_MIRROR_URL:
            direct_url = direct_url.replace(
                "https://huggingface.co", HF_MIRROR_URL
            )

        if _download_with_urllib(direct_url, dest_path, progress_callback):
            return True, "直接ダウンロードで完了"
        errors.append("直接ダウンロードも失敗")

    # ── 全て失敗 ──
    # 不完全なファイルを削除
    if dest_path.exists():
        dest_path.unlink()

    error_detail = "\n".join(errors)
    manual_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

    msg = (
        f"自動ダウンロードに失敗しました。\n\n"
        f"【エラー詳細】\n{error_detail}\n\n"
        f"【手動ダウンロード方法】\n"
        f"1. ブラウザで以下のURLを開いてダウンロード：\n"
        f"   {manual_url}\n"
        f"2. ダウンロードしたファイルを以下のフォルダに配置：\n"
        f"   {model_dir}\n"
        f"3. アプリを再起動\n\n"
        f"【その他の対処法】\n"
        f"- セキュリティソフトを一時停止してリトライ\n"
        f"- 設定画面で「HFミラーURL」を設定\n"
        f"  （例: https://hf-mirror.com）"
    )
    return False, msg


def download_hf_model(
    repo_id: str,
    filename: str,
    display_name: str,
) -> str | None:
    """
    任意のHuggingFaceリポジトリからGGUFをダウンロードしてカスタムモデルとして登録する。

    Returns:
        成功時は表示名、失敗時はNone
    """
    model_dir = MODELS_DIR / "llm"
    model_dir.mkdir(parents=True, exist_ok=True)
    dest_path = model_dir / filename

    try:
        from huggingface_hub import hf_hub_download
        logger.info(f"{repo_id}/{filename} をダウンロード中...")

        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )

        key = f"{display_name}（カスタム）"
        LLM_MODELS[key] = {
            "repo_id": repo_id,
            "filename": filename,
            "size_gb": round(dest_path.stat().st_size / (1024 ** 3), 1),
            "min_ram_gb": 8,
            "speed": "不明",
            "description": f"HF: {repo_id}/{filename}",
        }
        save_custom_models()
        return key

    except Exception as e:
        logger.error(f"ダウンロードエラー: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return None


# ─── LLM推論 ──────────────────────────────────────────────────

def get_llm(model_name: str):
    """
    LLMインスタンスを取得する（遅延初期化・シングルトン）。
    モデルが変更された場合は前のモデルを解放して新しいモデルを読み込む。
    """
    global _llm, _loaded_model_name

    if _llm is not None and _loaded_model_name == model_name:
        return _llm

    # 前のモデルを解放
    if _llm is not None:
        logger.info(f"モデルを解放: {_loaded_model_name}")
        del _llm
        _llm = None
        gc.collect()

    model_path = get_model_path(model_name)
    if model_path is None:
        raise FileNotFoundError(
            f"モデルファイルが見つかりません: {model_name}\n"
            f"設定画面からダウンロードしてください。"
        )

    logger.info(f"LLMを読み込み中: {model_path}")

    try:
        from llama_cpp import Llama

        total_ram = get_total_ram_gb()
        # RAMに応じてコンテキスト長を調整
        n_ctx = 4096 if total_ram >= 16 else 2048

        _llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=min(os.cpu_count() or 4, 8),  # 最大8スレッド
            n_gpu_layers=0,  # CPU推論（GPU未対応PCでも安全）
            verbose=False,
        )
        _loaded_model_name = model_name
        logger.info(f"LLM読み込み完了: {model_name} (n_ctx={n_ctx})")
        return _llm

    except ImportError:
        raise ImportError(
            "llama-cpp-python がインストールされていません。\n"
            "pip install llama-cpp-python を実行してください。"
        )
    except Exception as e:
        logger.error(f"LLM読み込みエラー: {e}")
        raise


def _build_prompt(
    model_name: str,
    system_prompt: str,
    user_message: str,
) -> tuple[str, list[str]]:
    """
    モデルに対応したチャットプロンプトを構築する。

    Returns:
        (プロンプト文字列, ストップトークンリスト)
    """
    model_info = LLM_MODELS.get(model_name, {})
    template_name = model_info.get("chat_template", "chatml")
    tmpl = CHAT_TEMPLATES.get(template_name)

    if tmpl is None:
        # デフォルトはChatML
        tmpl = CHAT_TEMPLATES["chatml"]

    prompt = (
        f"{tmpl['system_prefix']}{system_prompt}{tmpl['system_suffix']}"
        f"{tmpl['user_prefix']}{user_message}{tmpl['user_suffix']}"
        f"{tmpl['assistant_prefix']}"
    )
    stop_tokens = tmpl["stop_tokens"]

    return prompt, stop_tokens


def generate_answer(
    model_name: str,
    question: str,
    context: str,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str:
    """
    質問とコンテキストから回答を生成する。
    モデルに対応したチャットテンプレートを自動選択する。

    Args:
        model_name: 使用するモデル名
        question: ユーザーの質問
        context: RAG検索で取得した文書コンテキスト
        max_tokens: 最大生成トークン数

    Returns:
        生成された回答文字列
    """
    llm = get_llm(model_name)

    system_prompt = (
        "あなたは社内文書に基づいて質問に答えるAIアシスタントです。\n"
        "以下のルールを必ず守ってください：\n"
        "1. 提供された参考文書の内容のみに基づいて回答する\n"
        "2. 文書に記載のない情報を推測や補完で答えない\n"
        "3. 文書に記載がない場合は「文書には記載がありません」と明示する\n"
        "4. 回答は正確・簡潔に。日本語で回答する"
    )

    user_message = (
        f"以下の参考文書を元に、質問に答えてください。\n\n"
        f"【参考文書】\n{context}\n\n"
        f"【質問】\n{question}"
    )

    prompt, stop_tokens = _build_prompt(model_name, system_prompt, user_message)

    response = llm(
        prompt,
        max_tokens=max_tokens,
        stop=stop_tokens,
        echo=False,
    )

    answer = response["choices"][0]["text"].strip()
    return answer


def release_llm() -> None:
    """LLMを解放してメモリを開放する"""
    global _llm, _loaded_model_name
    if _llm is not None:
        del _llm
        _llm = None
        _loaded_model_name = None
        gc.collect()
        logger.info("LLMを解放しました")
