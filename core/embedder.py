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
from typing import Optional

logger = logging.getLogger(__name__)

# グローバルモデルキャッシュ（プロセス内で1回だけ読み込む）
_model = None
# 最後のダウンロードエラーメッセージ
_last_error: str | None = None

# ═══════════════════════════════════════════════════════════════════════════
# VRAM自動調整システム
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveBatchSizeManager:
    """
    VRAMサイズに応じてBATCH_SIZEを自動調整するマネージャー
    - 起動時にVRAM検出して最適値を計算
    - OOM検出時に自動フォールバック
    """

    def __init__(self, initial_batch_size: Optional[int] = None):
        self.total_vram_mb, self.free_vram_mb = self._get_vram_info()

        if initial_batch_size:
            self.current_batch_size = initial_batch_size
        else:
            self.current_batch_size = self._calculate_optimal_batch_size()

        self.min_batch_size = 64
        self.max_batch_size = self.current_batch_size
        self.oom_count = 0
        self.last_adjustment_time = time.time()

        logger.info(f"[VRAM Auto-Tune] 検出VRAM: {self.total_vram_mb}MB, "
                   f"初期BATCH_SIZE: {self.current_batch_size}")

    def _get_vram_info(self) -> tuple[int, int]:
        """VRAM総容量と空き容量をMB単位で返す"""
        try:
            import torch
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
                allocated = torch.cuda.memory_allocated(0) // (1024 ** 2)
                # 予約済みメモリを除いた実質的な空き
                free = total - allocated
                return total, max(free, 512)  # 最低512MBは確保
        except ImportError:
            pass
        return 0, 0

    def _calculate_optimal_batch_size(self) -> int:
        """VRAMサイズに応じた最適BATCH_SIZEを計算"""
        if self.total_vram_mb == 0:
            return 256  # CPUモード時のデフォルト

        # 安全マージン: 1GB（OS/他アプリ用）
        usable_vram = max(self.total_vram_mb - 1024, 512)

        # E5-Largeモデルの推定メモリ使用量（チャンクあたり約2MB）
        # BATCH_SIZE * 2MB ≈ 使用VRAM
        # 最大60%まで使用
        max_batch_for_vram = int((usable_vram * 0.6) / 2)

        # 段階的な値を選択（2の倍数で切り上げ）
        valid_sizes = [128, 256, 384, 512, 768, 1024, 1536, 2048]
        for size in valid_sizes:
            if max_batch_for_vram <= size:
                return size
        return 2048  # 最大値

    def on_oom_error(self) -> int:
        """OOM発生時にBATCH_SIZEを減少"""
        self.oom_count += 1
        new_size = int(self.current_batch_size * 0.7)  # 30%減少

        # 最小値を下回らないように
        if new_size < self.min_batch_size:
            new_size = self.min_batch_size

        # 有効な値に丸める
        valid_sizes = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048]
        for size in valid_sizes:
            if new_size <= size:
                new_size = size
                break

        self.current_batch_size = new_size
        logger.warning(f"[VRAM Auto-Tune] OOM検出 #{self.oom_count}: "
                      f"BATCH_SIZEを{new_size}に減少")
        return new_size

    def try_increase(self) -> bool:
        """VRAMに余裕があればBATCH_SIZEを増加"""
        if self.total_vram_mb == 0:
            return False

        # 前回調整から30秒以上経過しているかチェック
        if time.time() - self.last_adjustment_time < 30:
            return False

        try:
            import torch
            total = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)

            # 使用率が40%未満なら増加を検討
            usage_ratio = allocated / total
            if usage_ratio < 0.4 and self.current_batch_size < self.max_batch_size:
                valid_sizes = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048]
                current_idx = valid_sizes.index(self.current_batch_size) if self.current_batch_size in valid_sizes else 2

                if current_idx < len(valid_sizes) - 1:
                    new_size = valid_sizes[current_idx + 1]
                    if new_size <= self.max_batch_size:
                        self.current_batch_size = new_size
                        self.last_adjustment_time = time.time()
                        logger.info(f"[VRAM Auto-Tune] VRAM余裕検出: "
                                   f"BATCH_SIZEを{new_size}に増加")
                        return True
        except ImportError:
            pass
        return False

    def get_current_batch_size(self) -> int:
        """現在のBATCH_SIZEを返す"""
        return self.current_batch_size


# グローバルマネージャーインスタンス（遅延初期化）
_batch_size_manager: Optional[AdaptiveBatchSizeManager] = None


def get_batch_size_manager() -> AdaptiveBatchSizeManager:
    """グローバルBatchSizeManagerを取得（初回呼び出し時に初期化）"""
    global _batch_size_manager
    if _batch_size_manager is None:
        _batch_size_manager = AdaptiveBatchSizeManager()
    return _batch_size_manager


def get_optimal_batch_size() -> int:
    """VRAMに応じた最適BATCH_SIZEを返す"""
    return get_batch_size_manager().get_current_batch_size()


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
    batch_size: int = None,
) -> list[list[float]]:
    """
    テキストリストをEmbeddingベクトルに変換する。
    multilingual-e5-largeは "passage: " プレフィックスを使用（文書側）。

    Args:
        texts: 変換するテキストリスト
        cancel_check: キャンセル確認コールバック（呼び出してTrueならCancelledError）
        batch_size: Embeddingバッチサイズ（None時はVRAM自動検出）
    """
    if not texts:
        return []

    if cancel_check and cancel_check():
        from concurrent.futures import CancelledError
        raise CancelledError()

    # VRAM自動調整: batch_sizeが未指定なら最適値を取得
    if batch_size is None:
        batch_size = get_optimal_batch_size()

    model = get_embedding_model()
    prefixed = [f"passage: {t}" for t in texts]

    # バッチ単位でエンコードしてキャンセルチェックを挟む
    all_embeddings: list = []

    try:
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

    except RuntimeError as e:
        # OOM検出時の自動フォールバック
        if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
            logger.warning(f"[VRAM Auto-Tune] OOM検出: {e}")

            # BATCH_SIZEを減少してリトライ
            manager = get_batch_size_manager()
            new_batch_size = manager.on_oom_error()

            # GPUキャッシュをクリア
            try:
                import torch
                torch.cuda.empty_cache()
                logger.info("[VRAM Auto-Tune] GPUキャッシュをクリアしました")
            except ImportError:
                pass

            # 処理済み部分を破棄して、減少したBATCH_SIZEでリトライ
            logger.info(f"[VRAM Auto-Tune] BATCH_SIZE={new_batch_size}でリトライします")
            return embed_texts(texts, cancel_check, new_batch_size)
        else:
            raise  # OOM以外のエラーはそのまま再送出

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
