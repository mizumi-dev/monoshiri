"""
モノシリ 起動スプラッシュスクリーン
QApplication 起動直後に最速表示し、重いロードの進捗をユーザーへ伝える。
ユーザーの不安を軽減するため 1 秒以内に必ず表示される。
"""
from __future__ import annotations

import time

from PyQt6.QtCore import Qt, QTimer, QMetaObject, Q_ARG, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QLabel,
    QProgressBar,
    QSplashScreen,
    QVBoxLayout,
    QWidget,
)


class SplashScreen(QSplashScreen):
    """
    ロード進捗を表示するスプラッシュ。
    バックグラウンドスレッドから `set_status_safe()` で安全に更新できる。
    """

    def __init__(self) -> None:
        # ベースとなる空のピックスマップ（背景を描画）
        pixmap = QPixmap(520, 280)
        pixmap.fill(QColor("#1e3a5f"))  # 紺系の落ち着いた色

        # タイトルを直接描画
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # タイトル
        painter.setPen(QColor("#ffffff"))
        title_font = QFont("Meiryo UI", 28, QFont.Weight.Bold)
        if not title_font.exactMatch():
            title_font = QFont("Yu Gothic UI", 28, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.drawText(
            pixmap.rect().adjusted(0, 60, 0, 0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "モノシリ",
        )

        # サブタイトル
        painter.setPen(QColor("#c0d4e8"))
        sub_font = QFont("Meiryo UI", 11)
        painter.setFont(sub_font)
        painter.drawText(
            pixmap.rect().adjusted(0, 120, 0, 0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "社内資料探索AI — 完全ローカル動作",
        )

        painter.end()

        super().__init__(pixmap, Qt.WindowType.WindowStaysOnTopHint)

        # 進捗バー（スプラッシュ上に子ウィジェットとして追加）
        self._container = QWidget(self)
        self._container.setGeometry(40, 190, 440, 70)
        self._container.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(self._container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._status_label = QLabel("起動中...", self._container)
        self._status_label.setStyleSheet(
            "color: #ffffff; font-size: 11px; background: transparent;"
        )
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        self._progress = QProgressBar(self._container)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(8)
        self._progress.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #4a6a8a;
                border-radius: 4px;
                background: #0f2540;
            }
            QProgressBar::chunk {
                background: #5aa0e0;
                border-radius: 3px;
            }
            """
        )
        layout.addWidget(self._progress)

        # 経過時間ラベル（ユーザーが「止まっていない」と確認できるように）
        self._elapsed_label = QLabel("", self._container)
        self._elapsed_label.setStyleSheet(
            "color: #a0b8d0; font-size: 10px; background: transparent;"
        )
        self._elapsed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._elapsed_label)

        self._t_start = time.time()
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._elapsed_timer.start(100)

    # ──────────────────────────────────────────────────────────
    # UIスレッドから直接呼ぶメソッド
    # ──────────────────────────────────────────────────────────
    @pyqtSlot(str, int)
    def set_status(self, message: str, progress: int) -> None:
        """UIスレッドから安全に呼ぶ版"""
        self._status_label.setText(message)
        self._progress.setValue(progress)

    # ──────────────────────────────────────────────────────────
    # バックグラウンドスレッドから呼べる版（QMetaObject 経由で UI スレッドへ dispatch）
    # ──────────────────────────────────────────────────────────
    def set_status_safe(self, message: str, progress: int) -> None:
        """別スレッドから呼んでも安全な版。UIスレッドで set_status を実行する。"""
        try:
            QMetaObject.invokeMethod(
                self,
                "set_status",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, message),
                Q_ARG(int, progress),
            )
        except Exception:
            # 失敗してもロード本体は止めない
            pass

    def _update_elapsed(self) -> None:
        elapsed = time.time() - self._t_start
        self._elapsed_label.setText(f"経過時間: {elapsed:.1f}秒")

    def finish(self, window) -> None:  # type: ignore[override]
        self._elapsed_timer.stop()
        super().finish(window)
