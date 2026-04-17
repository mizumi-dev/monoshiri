"""
ui_qt/splash.py の結合テスト（pytest-qt）
"""
from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.ui


class TestSplashScreen:
    def test_construction_fast(self, qtbot):
        """スプラッシュのオブジェクト生成は 200ms 以内

        注: show() は Windows の描画に 1 秒ほど要するが、実アプリでは
        app.exec() 内で非同期処理されるためユーザー体感は異なる。
        実ユーザー体感での 1 秒以内は Phase D E2E で検証する。
        """
        from ui_qt.splash import SplashScreen  # 事前 import でコスト除外
        t_start = time.time()
        splash = SplashScreen()
        elapsed = time.time() - t_start
        assert elapsed < 0.2, f"スプラッシュ生成が遅すぎる: {elapsed*1000:.0f}ms"
        splash.show()
        qtbot.waitExposed(splash, timeout=3000)
        splash.close()

    def test_set_status_updates_label(self, qtbot):
        from ui_qt.splash import SplashScreen
        splash = SplashScreen()
        qtbot.addWidget(splash)
        splash.show()
        splash.set_status("テストステータス", 42)
        assert splash._status_label.text() == "テストステータス"
        assert splash._progress.value() == 42

    def test_set_status_safe_thread_safe(self, qtbot):
        """バックグラウンドスレッドから呼んでも例外にならない"""
        import threading
        from ui_qt.splash import SplashScreen

        splash = SplashScreen()
        qtbot.addWidget(splash)
        splash.show()

        def bg():
            splash.set_status_safe("bg status", 77)

        t = threading.Thread(target=bg, daemon=True)
        t.start()
        t.join(timeout=3)

        # QueuedConnection のイベントを処理させる
        qtbot.wait(200)

        assert splash._status_label.text() == "bg status"
        assert splash._progress.value() == 77

    def test_elapsed_timer_updates(self, qtbot):
        from ui_qt.splash import SplashScreen
        splash = SplashScreen()
        qtbot.addWidget(splash)
        splash.show()
        qtbot.wait(300)  # 100ms * 3
        text = splash._elapsed_label.text()
        assert "経過時間" in text
        assert "秒" in text
