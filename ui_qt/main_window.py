"""
モノシリ メインウィンドウ
左サイドバー（ナビ） + QStackedWidget（3ページ）構成
"""
from __future__ import annotations
import logging

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QStackedWidget, QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QFont, QIcon

from ui_qt.workers import StatsWorker, OllamaStatusWorker

logger = logging.getLogger(__name__)

# サイドバー幅
SIDEBAR_WIDTH = 190

# サイドバーCSS
SIDEBAR_STYLE = """
QWidget#sidebar {
    background-color: #1e2130;
    border-right: 1px solid #2d3250;
}
QPushButton#navBtn {
    background-color: transparent;
    color: #c8cfe8;
    border: none;
    text-align: left;
    padding: 10px 16px;
    font-size: 13px;
    border-radius: 6px;
    margin: 2px 8px;
}
QPushButton#navBtn:hover {
    background-color: #2d3250;
    color: #ffffff;
}
QPushButton#navBtn[active="true"] {
    background-color: #3b4a82;
    color: #ffffff;
    font-weight: bold;
}
QLabel#appTitle {
    color: #ffffff;
    font-size: 16px;
    font-weight: bold;
    padding: 16px 16px 4px 16px;
}
QLabel#appSubtitle {
    color: #7a8aaa;
    font-size: 10px;
    padding: 0px 16px 12px 16px;
}
QLabel#statsLabel {
    color: #7a8aaa;
    font-size: 10px;
    padding: 4px 16px;
}
QFrame#divider {
    background-color: #2d3250;
    max-height: 1px;
    margin: 4px 12px;
}
"""


class MainWindow(QMainWindow):
    """
    モノシリ メインウィンドウ。
    左サイドバー + 右コンテンツ（QStackedWidget）。
    """

    def __init__(self, config: dict, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._pages: list[QWidget] = []
        self._nav_buttons: list[QPushButton] = []

        self.setWindowTitle("モノシリ — 社内資料探索AI")
        self.setMinimumSize(920, 640)
        self.resize(1200, 760)

        self._build_ui()

        # 起動時はチャットページを表示
        self._switch_page(0)

        # インデックス統計の定期更新（5秒ごと）
        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(5000)
        self._stats_timer.timeout.connect(self._update_stats)
        self._stats_timer.start()
        QTimer.singleShot(1000, self._update_stats)  # ChromaDB初期化を遅延

        # Ollamaステータスの定期更新（10秒ごと）
        self._ollama_timer = QTimer(self)
        self._ollama_timer.setInterval(10000)
        self._ollama_timer.timeout.connect(self._update_ollama_status)
        self._ollama_timer.start()
        QTimer.singleShot(1500, self._update_ollama_status)  # Ollama確認を遅延

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── サイドバー ──────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        sidebar.setStyleSheet(SIDEBAR_STYLE)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # ロゴ・タイトル
        title_lbl = QLabel("🔍 モノシリ")
        title_lbl.setObjectName("appTitle")
        sub_lbl = QLabel("社内資料は、一切外に出ません")
        sub_lbl.setObjectName("appSubtitle")
        sub_lbl.setWordWrap(True)
        sidebar_layout.addWidget(title_lbl)
        sidebar_layout.addWidget(sub_lbl)

        # 区切り線
        div1 = QFrame()
        div1.setObjectName("divider")
        sidebar_layout.addWidget(div1)

        # ナビボタン
        nav_items = [
            ("💬  チャット", "chat"),
            ("📁  インデックス管理", "index"),
            ("⚙️  設定", "settings"),
        ]
        for i, (label, _key) in enumerate(nav_items):
            btn = QPushButton(label)
            btn.setObjectName("navBtn")
            btn.setCheckable(False)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, idx=i: self._switch_page(idx))
            sidebar_layout.addWidget(btn)
            self._nav_buttons.append(btn)

        # 区切り線
        div2 = QFrame()
        div2.setObjectName("divider")
        sidebar_layout.addWidget(div2)

        # 統計情報ラベル
        self._stats_label = QLabel("読み込み中...")
        self._stats_label.setObjectName("statsLabel")
        self._stats_label.setWordWrap(True)
        sidebar_layout.addWidget(self._stats_label)

        # Ollamaステータス
        self._ollama_label = QLabel("Ollama: 確認中...")
        self._ollama_label.setObjectName("statsLabel")
        self._ollama_label.setWordWrap(True)
        sidebar_layout.addWidget(self._ollama_label)

        sidebar_layout.addStretch()

        # バージョン表示
        ver_lbl = QLabel("v4.0")
        ver_lbl.setObjectName("statsLabel")
        ver_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        sidebar_layout.addWidget(ver_lbl)

        root_layout.addWidget(sidebar)

        # ── コンテンツエリア ──────────────────────────────────
        self._stack = QStackedWidget()
        self._stack.setStyleSheet("background-color: #f5f6fa;")
        root_layout.addWidget(self._stack, 1)

        # 3ページを遅延ロード（初回アクセス時に初期化）
        from ui_qt.chat_widget import ChatWidget
        from ui_qt.index_widget import IndexWidget
        from ui_qt.settings_widget import SettingsWidget

        self._chat_widget = ChatWidget(config=self._config, parent=self)
        self._index_widget = IndexWidget(config=self._config, parent=self)
        self._settings_widget = SettingsWidget(config=self._config, parent=self)

        self._stack.addWidget(self._chat_widget)
        self._stack.addWidget(self._index_widget)
        self._stack.addWidget(self._settings_widget)

    @pyqtSlot(int)
    def _switch_page(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        for i, btn in enumerate(self._nav_buttons):
            btn.setProperty("active", "true" if i == index else "false")
            btn.setStyleSheet("")  # プロパティ変更後にスタイル再適用

    @pyqtSlot()
    def _update_stats(self) -> None:
        w = StatsWorker(dict(self._config))
        w._callback = self._stats_label.setText
        w.start()

    @pyqtSlot()
    def _update_ollama_status(self) -> None:
        w = OllamaStatusWorker()
        w._callback = self._apply_ollama_status
        w.start()

    @pyqtSlot(bool)
    def _apply_ollama_status(self, running: bool) -> None:
        if running:
            self._ollama_label.setText("🟢 Ollama 起動中")
            self._ollama_label.setStyleSheet("color: #4ade80; font-size: 10px; padding: 4px 16px;")
        else:
            self._ollama_label.setText("🔴 Ollama 未起動")
            self._ollama_label.setStyleSheet("color: #f87171; font-size: 10px; padding: 4px 16px;")

    def closeEvent(self, event) -> None:
        """ウィンドウ閉じ時に全タイマー停止"""
        self._stats_timer.stop()
        self._ollama_timer.stop()
        super().closeEvent(event)

    def update_config(self, config: dict) -> None:
        """設定変更時に各ページへ通知する"""
        self._config = config
        if hasattr(self._chat_widget, "update_config"):
            self._chat_widget.update_config(config)
        self._update_stats()
