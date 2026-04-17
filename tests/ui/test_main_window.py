"""
ui_qt/main_window.py の結合テスト（pytest-qt）
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.ui


@pytest.fixture
def config():
    return {
        "selected_model": "",
        "folders": [],
        "similarity_threshold": 0.40,
        "plan": "free",
        "llm_n_gpu_layers": 0,
        "embedding_device": "auto",
    }


class TestMainWindow:
    def test_creates_window(self, qtbot, config):
        from ui_qt.main_window import MainWindow
        w = MainWindow(config=config)
        qtbot.addWidget(w)
        assert w.windowTitle() == "モノシリ — 社内資料探索AI"

    def test_three_nav_buttons(self, qtbot, config):
        from ui_qt.main_window import MainWindow
        w = MainWindow(config=config)
        qtbot.addWidget(w)
        assert len(w._nav_buttons) == 3

    def test_default_page_is_chat(self, qtbot, config):
        from ui_qt.main_window import MainWindow
        w = MainWindow(config=config)
        qtbot.addWidget(w)
        assert w._stack.currentIndex() == 0

    def test_switch_page(self, qtbot, config):
        from ui_qt.main_window import MainWindow
        w = MainWindow(config=config)
        qtbot.addWidget(w)
        w._switch_page(1)
        assert w._stack.currentIndex() == 1
        w._switch_page(2)
        assert w._stack.currentIndex() == 2
        w._switch_page(0)
        assert w._stack.currentIndex() == 0

    def test_nav_button_click_switches_page(self, qtbot, config):
        from PyQt6.QtCore import Qt
        from ui_qt.main_window import MainWindow
        w = MainWindow(config=config)
        qtbot.addWidget(w)
        qtbot.mouseClick(w._nav_buttons[1], Qt.MouseButton.LeftButton)
        assert w._stack.currentIndex() == 1

    def test_update_stats_does_not_crash(self, qtbot, config):
        from ui_qt.main_window import MainWindow
        w = MainWindow(config=config)
        qtbot.addWidget(w)
        w._update_stats()  # 例外にならない

    def test_apply_ollama_status(self, qtbot, config):
        from ui_qt.main_window import MainWindow
        w = MainWindow(config=config)
        qtbot.addWidget(w)
        w._apply_ollama_status(True)
        assert "起動中" in w._ollama_label.text()
        w._apply_ollama_status(False)
        assert "未起動" in w._ollama_label.text()
