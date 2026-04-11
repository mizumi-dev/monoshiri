"""
モノシリ Embeddingモジュール
multilingual-e5-large を使った完全ローカルEmbedding。
外部送信は一切なし。モードに関わらず常にローカル実行。
"""
from __future__ import annotations
import logging
import os
import time
from concurrent.futures import CancelledError
from pathlib import Path

logger = logging.getLogger(__name__)

# グローバルモデルキャッシュ（プロセス内で1回だけ読み込む）
_model = None
# 最後のダウンロードエラーメッセージ
_last_error: str | None = None


def get_last_error() -> str | None:
    """最後のダウンロードエラーを返す"""
    return _last_error


def get_embedding_model():
    """
    Embeddingモデルを取得する（遅延初期化）。
    初回呼び出し時にローカルキャッシュを探し、なければHuggingFaceからダウンロード。
    """
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def _resolve_device() -> str:
    """
    使用するデバイスを決定する。
    設定が "auto" の場合はCUDAの有無を自動判定。
    """
    import core.config as _cfg
    device_pref = getattr(_cfg, "EMBEDDING_DEVICE", "auto")

    if device_pref == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
                logger.info(f"CUDAが使用可能: GPU={torch.cuda.get_device_name(0)}, VRAM={vram_mb}MB → GPU使用")
            else:
                device = "cpu"
                logger.info("CUDAが使用不可 → CPU使用")
        except ImportError:
            device = "cpu"
            logger.info("torchがインポートできません → CPU使用")
    else:
        device = device_pref
        logger.info(f"デバイス設定: {device}（設定値）")

    return device


def _load_model():
    global _last_error
    _last_error = None

    from sentence_transformers import SentenceTransformer
    from core.config import (
        EMBEDDING_MODEL_DIR, EMBEDDING_MODEL_ID,
        DOWNLOAD_MAX_RETRIES, HF_MIRROR_URL, MODELS_DIR,
    )

    logger.info("Embeddingモデルを初期化中...")
    device = _resolve_device()

    # ローカルキャッシュが存在する場合は優先使用
    if EMBEDDING_MODEL_DIR.exists() and any(EMBEDDING_MODEL_DIR.iterdir()):
        try:
            model = SentenceTransformer(str(EMBEDDING_MODEL_DIR), device=device)
            logger.info(f"ローカルキャッシュからEmbeddingモデルを読み込みました (device={device})")
            return model
        except Exception as e:
            logger.warning(f"ローカルキャッシュの読み込み失敗（再ダウンロードします）: {e}")

    # HuggingFaceミラーが設定されている場合は環境変数でオーバーライド
    original_endpoint = os.environ.get("HF_ENDPOINT", "")
    if HF_MIRROR_URL:
        os.environ["HF_ENDPOINT"] = HF_MIRROR_URL
        logger.info(f"HFミラーを使用: {HF_MIRROR_URL}")

    # リトライ付きでHuggingFaceからダウンロード
    errors = []
    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            logger.info(
                f"[{attempt}/{DOWNLOAD_MAX_RETRIES}] "
                f"HuggingFaceから {EMBEDDING_MODEL_ID} をダウンロード中..."
            )
            model = SentenceTransformer(EMBEDDING_MODEL_ID, device=device)

            # ローカルに保存して次回以降はオフラインで動作
            EMBEDDING_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(EMBEDDING_MODEL_DIR))
            logger.info(f"Embeddingモデルをローカルに保存しました: {EMBEDDING_MODEL_DIR} (device={device})")

            # 環境変数を戻す
            if original_endpoint:
                os.environ["HF_ENDPOINT"] = original_endpoint
            elif "HF_ENDPOINT" in os.environ and HF_MIRROR_URL:
                del os.environ["HF_ENDPOINT"]

            return model

        except Exception as e:
            err_msg = str(e)
            errors.append(f"試行{attempt}: {err_msg}")
            logger.warning(f"ダウンロード試行 {attempt} 失敗: {err_msg}")
            if attempt < DOWNLOAD_MAX_RETRIES:
                wait = 2 ** attempt
                logger.info(f"{wait}秒後にリトライ...")
                time.sleep(wait)

    # 環境変数を戻す
    if original_endpoint:
        os.environ["HF_ENDPOINT"] = original_endpoint
    elif "HF_ENDPOINT" in os.environ and HF_MIRROR_URL:
        del os.environ["HF_ENDPOINT"]

    error_detail = "\n".join(errors)
    cache_dir = EMBEDDING_MODEL_DIR
    _last_error = (
        f"Embeddingモデルのダウンロードに失敗しました。\n\n"
        f"【エラー詳細】\n{error_detail}\n\n"
        f"【手動ダウンロード方法】\n"
        f"1. ブラウザで以下のURLを開く：\n"
        f"   https://huggingface.co/{EMBEDDING_MODEL_ID}\n"
        f"2. 「Files and versions」タブから全ファイルをダウンロード\n"
        f"3. 以下のフォルダに配置：\n"
        f"   {cache_dir}\n"
        f"4. アプリを再起動\n\n"
        f"【その他の対処法】\n"
        f"- セキュリティソフトを一時停止してリトライ\n"
        f"- 設定画面で「HFミラーURL」を設定\n"
        f"  （例: https://hf-mirror.com）"
    )
    raise ConnectionError(_last_error)


def embed_texts(
    texts: list[str],
    cancel_check=None,
    batch_size: int = 256,
) -> list[list[float]]:
    """
    テキストリストをEmbeddingベクトルに変換する。
    multilingual-e5-largeは "passage: " プレフィックスを使用（文書側）。

    Args:
        texts: 変換するテキストリスト
        cancel_check: キャンセル確認コールバック（呼び出してTrueならCancelledError）
        batch_size: Embeddingバッチサイズ（大きいほど高速・メモリ使用量増加）
    """
    if not texts:
        return []

    if cancel_check and cancel_check():
        from concurrent.futures import CancelledError
        raise CancelledError()

    model = get_embedding_model()
    prefixed = [f"passage: {t}" for t in texts]

    # バッチ単位でエンコードしてキャンセルチェックを挟む
    all_embeddings: list = []
    for i in range(0, len(prefixed), batch_size):
        if cancel_check and cancel_check():
            raise CancelledError("Embeddingがキャンセルされました")
        batch = prefixed[i:i + batch_size]
        result = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.extend(result.tolist())

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """
    クエリをEmbeddingベクトルに変換する。
    multilingual-e5-largeは "query: " プレフィックスを使用（クエリ側）。
    """
    model = get_embedding_model()
    prefixed = f"query: {query}"
    embedding = model.encode(
        [prefixed],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embedding[0].tolist()


def is_embedding_ready() -> bool:
    """Embeddingモデルが利用可能かチェックする"""
    try:
        from sentence_transformers import SentenceTransformer
        return True
    except ImportError:
        return False


def reset_model() -> None:
    """モデルキャッシュをクリアする（メモリ解放用）"""
    global _model
    if _model is not None:
        del _model
        _model = None
        import gc
        gc.collect()
        logger.info("Embeddingモデルのキャッシュをクリアしました")
