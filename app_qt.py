"""
モノシリ PyQt6 ネイティブUIエントリーポイント
ブラウザ不要・start.bat一発起動
"""
from __future__ import annotations
import logging
import sys
import os
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _setup_logging() -> None:
    # MONOSHIRI_DATA_DIR 環境変数を尊重するため core.config の定数を利用
    from core.config import LOGS_DIR
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "monoshiri.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.WARNING)
    root.addHandler(ch)


def _warmup_model_background(model_name: str) -> None:
    """バックグラウンドでOllama warmupを実行する"""
    if not model_name:
        return
    try:
        from core.llm import warmup_model
        warmup_model(model_name)
    except Exception as e:
        logging.getLogger(__name__).warning(f"warmup失敗（無視）: {e}")


def main() -> None:
    import time
    t_start = time.time()

    _setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("モノシリ PyQt6 UI 起動")

    # ── まず Qt とスプラッシュを最速で立ち上げる ─────────────
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont
    from PyQt6.QtCore import QTimer

    app = QApplication(sys.argv)
    app.setApplicationName("モノシリ")
    app.setApplicationDisplayName("モノシリ — 社内資料探索AI")
    app.setStyle("Fusion")

    # デフォルトフォント（日本語対応）
    font = QFont("Meiryo UI", 9)
    if not font.exactMatch():
        font = QFont("Yu Gothic UI", 9)
    if not font.exactMatch():
        font = QFont("MS UI Gothic", 9)
    app.setFont(font)

    # スプラッシュを即座に表示（ユーザー体感は 1 秒以内）
    from ui_qt.splash import SplashScreen
    splash = SplashScreen()
    splash.show()
    app.processEvents()
    elapsed_ui = time.time() - t_start
    logger.info(f"スプラッシュ表示完了（{elapsed_ui:.2f}秒）")
    splash.set_status("起動中...", 5)

    # ── 重いロード処理をバックグラウンドスレッドで実行 ──────
    state = {"config": {}, "error": None, "done": False}

    def _heavy_load():
        try:
            # カスタムモデル読み込み
            splash.set_status_safe("カスタムモデル読み込み中...", 10)
            try:
                from core.llm import load_custom_models
                load_custom_models()
            except Exception:
                pass

            # 設定読み込み
            splash.set_status_safe("設定読み込み中...", 20)
            try:
                from core.config import load_config
                state["config"] = load_config()
            except Exception:
                state["config"] = {}

            # ChromaDB（重い）
            splash.set_status_safe("ベクトルDB初期化中...", 35)
            try:
                import chromadb  # noqa: F401
                from core.chromadb_store import get_chroma_client
                get_chroma_client()
            except Exception as e:
                logger.warning(f"ChromaDB 事前初期化スキップ: {e}")

            # sentence-transformers（torch を連鎖 import・最重量）
            splash.set_status_safe("Embedding エンジン読み込み中...", 60)
            try:
                import sentence_transformers  # noqa: F401
            except Exception:
                pass

            # IndexManager
            splash.set_status_safe("インデックスマネージャ初期化中...", 80)
            try:
                from core.indexer import IndexManager  # noqa: F401
            except Exception:
                pass

            # DownloadManager
            splash.set_status_safe("モデルダウンローダ初期化中...", 90)
            try:
                from core.downloader import DownloadManager  # noqa: F401
            except Exception:
                pass

            splash.set_status_safe("UI 構築中...", 98)
        except Exception as e:
            state["error"] = str(e)
            logger.exception("起動時ロード失敗")
        finally:
            state["done"] = True

    load_thread = threading.Thread(target=_heavy_load, daemon=True, name="startup-load")
    load_thread.start()

    # 定期的にロード完了を確認してメインウィンドウへ切替
    def _check_load():
        if not state["done"]:
            return
        timer.stop()
        elapsed_total = time.time() - t_start
        logger.info(f"重いロード完了（合計 {elapsed_total:.2f}秒）")

        if state["error"]:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "起動エラー", f"起動時にエラーが発生しました:\n{state['error']}")
            app.quit()
            return

        splash.set_status("準備完了", 100)
        config = state["config"]

        # メインウィンドウ表示
        from ui_qt.main_window import MainWindow
        window = MainWindow(config=config)
        window.show()
        splash.finish(window)

        # Ollama warmup をバックグラウンドで実行
        selected_model = config.get("selected_model", "")
        if selected_model:
            t = threading.Thread(
                target=_warmup_model_background,
                args=(selected_model,),
                daemon=True,
                name="warmup",
            )
            t.start()

        # ウィンドウへの参照を保持（GC防止）
        app._main_window = window  # type: ignore[attr-defined]

    timer = QTimer()
    timer.timeout.connect(_check_load)
    timer.start(50)  # 50ms 間隔でポーリング

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
