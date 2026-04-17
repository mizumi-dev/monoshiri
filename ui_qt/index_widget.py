"""
モノシリ インデックス管理ウィジェット
フォルダ追加/削除・インデックス作成・進捗表示
"""
from __future__ import annotations
import logging
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QProgressBar, QTextEdit,
    QFrame, QFileDialog, QMessageBox, QGroupBox, QSplitter,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QFont

from ui_qt.workers import IndexStatsWorker, _run_in_background

logger = logging.getLogger(__name__)


class IndexWidget(QWidget):
    """インデックス管理画面"""

    def __init__(self, config: dict, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._index_mgr = None  # バックグラウンドで初期化
        self._build_ui()
        self._refresh_folder_list()

        # IndexManagerのポーリングタイマー（まだ開始しない）
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(2000)
        self._poll_timer.timeout.connect(self._poll_index_status)

        # IndexManagerをバックグラウンドでimport + 初期化（3秒後）
        QTimer.singleShot(3000, self._init_index_mgr_background)

        # 初期統計更新（3秒後にバックグラウンドで実行）
        QTimer.singleShot(3000, self._update_stats)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        # ── タイトル ─────────────────────────────────────────
        title = QLabel("📁 インデックス管理")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #1a1a2e;")
        root.addWidget(title)

        # ── 説明 ──────────────────────────────────────────────
        desc = QLabel("社内資料が入ったフォルダを登録してインデックスを作成します。")
        desc.setStyleSheet("font-size: 12px; color: #666;")
        desc.setWordWrap(True)
        root.addWidget(desc)

        # ── 上下スプリッター（フォルダ管理 | 進捗・ログ） ────
        splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(splitter, 1)

        # ── フォルダ管理セクション ────────────────────────────
        folder_group = QGroupBox("対象フォルダ")
        folder_group.setStyleSheet("""
            QGroupBox {
                font-size: 13px; font-weight: bold; color: #1a1a2e;
                border: 1px solid #dde0f0; border-radius: 8px;
                margin-top: 8px; padding-top: 8px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
        """)
        folder_layout = QVBoxLayout(folder_group)
        folder_layout.setContentsMargins(12, 16, 12, 12)
        folder_layout.setSpacing(8)

        # フォルダリスト
        self._folder_list = QListWidget()
        self._folder_list.setMinimumHeight(120)
        self._folder_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #dde0f0; border-radius: 6px;
                background: white; font-size: 12px;
            }
            QListWidget::item { padding: 8px 12px; border-bottom: 1px solid #f0f2fa; }
            QListWidget::item:selected { background: #e8eaf4; color: #1a1a2e; }
        """)
        folder_layout.addWidget(self._folder_list)

        # フォルダ操作ボタン行
        folder_btn_row = QHBoxLayout()
        folder_btn_row.setSpacing(8)

        add_btn = QPushButton("＋ フォルダを追加")
        add_btn.setStyleSheet(self._btn_style("#3b4a82"))
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.clicked.connect(self._add_folder)

        self._remove_btn = QPushButton("－ 選択フォルダを削除")
        self._remove_btn.setStyleSheet(self._btn_style("#888"))
        self._remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._remove_btn.clicked.connect(self._remove_folder)

        folder_btn_row.addWidget(add_btn)
        folder_btn_row.addWidget(self._remove_btn)
        folder_btn_row.addStretch()
        folder_layout.addLayout(folder_btn_row)

        splitter.addWidget(folder_group)

        # ── インデックス進捗セクション ────────────────────────
        progress_group = QGroupBox("インデックス処理")
        progress_group.setStyleSheet(folder_group.styleSheet())
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setContentsMargins(12, 16, 12, 12)
        progress_layout.setSpacing(8)

        # 統計情報
        self._stats_label = QLabel("インデックス済み: 読み込み中...")
        self._stats_label.setStyleSheet("font-size: 12px; color: #4a5080;")
        progress_layout.addWidget(self._stats_label)

        # ステータス行
        status_row = QHBoxLayout()
        self._phase_label = QLabel("待機中")
        self._phase_label.setStyleSheet("font-size: 12px; color: #666;")
        self._elapsed_label = QLabel("")
        self._elapsed_label.setStyleSheet("font-size: 11px; color: #888;")
        status_row.addWidget(self._phase_label)
        status_row.addStretch()
        status_row.addWidget(self._elapsed_label)
        progress_layout.addLayout(status_row)

        # プログレスバー
        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(18)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #dde0f0; border-radius: 6px;
                background: #f0f2fa; text-align: center; font-size: 11px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b4a82, stop:1 #5a70c8);
                border-radius: 5px;
            }
        """)
        progress_layout.addWidget(self._progress_bar)

        # ファイル名表示
        self._current_file_label = QLabel("")
        self._current_file_label.setStyleSheet("font-size: 10px; color: #888;")
        self._current_file_label.setWordWrap(True)
        progress_layout.addWidget(self._current_file_label)

        # インデックスボタン行
        index_btn_row = QHBoxLayout()
        index_btn_row.setSpacing(8)

        self._index_btn = QPushButton("▶ インデックス作成")
        self._index_btn.setStyleSheet(self._btn_style("#27ae60"))
        self._index_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._index_btn.clicked.connect(self._start_index)

        self._reindex_btn = QPushButton("🔄 全件再インデックス")
        self._reindex_btn.setStyleSheet(self._btn_style("#e67e22"))
        self._reindex_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reindex_btn.clicked.connect(self._start_reindex)

        self._cancel_btn = QPushButton("⏹ キャンセル")
        self._cancel_btn.setStyleSheet(self._btn_style("#e74c3c"))
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_btn.clicked.connect(self._cancel_index)

        index_btn_row.addWidget(self._index_btn)
        index_btn_row.addWidget(self._reindex_btn)
        index_btn_row.addWidget(self._cancel_btn)
        index_btn_row.addStretch()
        progress_layout.addLayout(index_btn_row)

        splitter.addWidget(progress_group)

        # ── スキップログセクション ────────────────────────────
        skip_group = QGroupBox("スキップログ（処理できなかったファイル）")
        skip_group.setStyleSheet(folder_group.styleSheet())
        skip_layout = QVBoxLayout(skip_group)
        skip_layout.setContentsMargins(12, 16, 12, 12)

        self._skip_log_view = QTextEdit()
        self._skip_log_view.setReadOnly(True)
        self._skip_log_view.setMaximumHeight(120)
        self._skip_log_view.setStyleSheet("""
            QTextEdit {
                border: 1px solid #dde0f0; border-radius: 6px;
                background: #fffdf5; font-size: 11px; color: #666;
                font-family: 'Consolas', monospace;
            }
        """)
        self._skip_log_view.setPlaceholderText("スキップされたファイルがあれば、ここに表示されます。")
        skip_layout.addWidget(self._skip_log_view)

        splitter.addWidget(skip_group)
        splitter.setSizes([200, 220, 120])

    def _btn_style(self, color: str) -> str:
        c = color.lstrip("#")
        if len(c) == 3:
            c = c[0]*2 + c[1]*2 + c[2]*2
        r = min(255, int(c[0:2], 16) + 20)
        g = min(255, int(c[2:4], 16) + 20)
        b = min(255, int(c[4:6], 16) + 20)
        hover = f"#{r:02x}{g:02x}{b:02x}"
        return f"""
            QPushButton {{
                background-color: {color};
                color: white; border: none; border-radius: 6px;
                padding: 6px 14px; font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {hover}; }}
            QPushButton:disabled {{ background-color: #aaa; }}
        """

    # ── フォルダ管理 ─────────────────────────────────────────

    @pyqtSlot()
    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "フォルダを選択", "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if not folder:
            return
        folder_path = str(Path(folder))
        folders: list[str] = self._config.get("folders", [])
        if folder_path in folders:
            QMessageBox.information(self, "確認", "このフォルダはすでに登録されています。")
            return
        folders.append(folder_path)
        self._config["folders"] = folders
        self._save_config()
        self._refresh_folder_list()

    @pyqtSlot()
    def _remove_folder(self) -> None:
        items = self._folder_list.selectedItems()
        if not items:
            QMessageBox.information(self, "確認", "削除するフォルダを選択してください。")
            return
        folder_path = items[0].data(Qt.ItemDataRole.UserRole)
        reply = QMessageBox.question(
            self, "フォルダを削除",
            f"以下のフォルダを登録から削除しますか？\n{folder_path}\n\n"
            "（ファイル自体は削除されません）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        folders: list[str] = self._config.get("folders", [])
        if folder_path in folders:
            folders.remove(folder_path)
        self._config["folders"] = folders
        self._save_config()
        self._refresh_folder_list()

    def _refresh_folder_list(self) -> None:
        self._folder_list.clear()
        folders: list[str] = self._config.get("folders", [])
        for f in folders:
            p = Path(f)
            try:
                file_count = sum(1 for _ in p.rglob("*") if _.is_file()) if p.exists() else 0
                label = f"{p.name}  ({file_count} ファイル)\n{f}"
            except Exception:
                label = f"{p.name}\n{f}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, f)
            self._folder_list.addItem(item)

        if not folders:
            placeholder = QListWidgetItem("フォルダが登録されていません。「＋ フォルダを追加」から追加してください。")
            placeholder.setFlags(placeholder.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            placeholder.setForeground(Qt.GlobalColor.gray)
            self._folder_list.addItem(placeholder)

    def _save_config(self) -> None:
        try:
            from core.config import save_config
            save_config(self._config)
        except Exception as e:
            logger.warning(f"設定保存失敗: {e}")

    # ── インデックス処理 ─────────────────────────────────────

    @pyqtSlot()
    def _start_index(self) -> None:
        folders = [Path(f) for f in self._config.get("folders", []) if Path(f).exists()]
        if not folders:
            QMessageBox.warning(self, "フォルダ未登録", "インデックス対象のフォルダを先に追加してください。")
            return
        mgr = self._index_mgr
        if mgr is None:
            QMessageBox.warning(self, "初期化中", "インデックスエンジンがまだ初期化中です。しばらくお待ちください。")
            return
        try:
            if mgr.is_running:
                QMessageBox.information(self, "処理中", "インデックス処理がすでに実行中です。")
                return
            mgr.start(folders)
            self._on_index_started()
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"インデックス開始エラー:\n{e}")

    @pyqtSlot()
    def _start_reindex(self) -> None:
        reply = QMessageBox.question(
            self, "全件再インデックス",
            "既存のインデックスを削除して全ファイルを再処理します。\n"
            "大量のファイルがある場合は時間がかかります。続けますか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        folders = [Path(f) for f in self._config.get("folders", []) if Path(f).exists()]
        if not folders:
            QMessageBox.warning(self, "フォルダ未登録", "インデックス対象のフォルダを先に追加してください。")
            return
        mgr = self._index_mgr
        if mgr is None:
            QMessageBox.warning(self, "初期化中", "インデックスエンジンがまだ初期化中です。しばらくお待ちください。")
            return
        try:
            if mgr.is_running:
                QMessageBox.information(self, "処理中", "インデックス処理がすでに実行中です。")
                return
            mgr.start_force_reindex(folders)
            self._on_index_started()
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"再インデックス開始エラー:\n{e}")

    @pyqtSlot()
    def _cancel_index(self) -> None:
        if self._index_mgr:
            try:
                self._index_mgr.cancel()
            except Exception:
                pass

    def _on_index_started(self) -> None:
        self._index_btn.setEnabled(False)
        self._reindex_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)

    # ── IndexManager ポーリング ──────────────────────────────

    def _init_index_mgr_background(self) -> None:
        """IndexManagerをバックグラウンドでimport・初期化し、完了後にポーリングを開始する"""
        def _task():
            from core.indexer import IndexManager
            return IndexManager.get()
        def _on_done(mgr):
            self._index_mgr = mgr
            self._poll_timer.start()
        _run_in_background(_task, _on_done)

    @pyqtSlot()
    def _poll_index_status(self) -> None:
        mgr = self._index_mgr
        if mgr is None:
            return

        status = mgr.status
        phase = mgr.phase_label if hasattr(mgr, "phase_label") else status
        progress = mgr.progress
        done = mgr.done_count
        total = mgr.total_count
        current_file = getattr(mgr, "current_file", "")
        elapsed_str = mgr.get_elapsed_str() if mgr.start_time else ""

        self._phase_label.setText(phase)
        self._elapsed_label.setText(elapsed_str)

        if total > 0:
            self._progress_bar.setMaximum(total)
            self._progress_bar.setValue(done)
            self._progress_bar.setFormat(f"{done} / {total}")
        else:
            self._progress_bar.setMaximum(100)
            self._progress_bar.setValue(int(progress * 100))
            self._progress_bar.setFormat(f"{int(progress * 100)}%")

        if current_file:
            fname = Path(current_file).name if current_file else ""
            self._current_file_label.setText(f"処理中: {fname}")
        else:
            self._current_file_label.setText("")

        if status in ("done", "cancelled", "error", "idle"):
            self._index_btn.setEnabled(True)
            self._reindex_btn.setEnabled(True)
            self._cancel_btn.setVisible(False)
            if status == "done":
                self._update_stats()
                self._load_skip_log()
        elif mgr.is_running:
            self._index_btn.setEnabled(False)
            self._reindex_btn.setEnabled(False)
            self._cancel_btn.setVisible(True)

    def _update_stats(self) -> None:
        w = IndexStatsWorker()
        w._callback = self._stats_label.setText
        w.start()

    def _load_skip_log(self) -> None:
        try:
            from core.config import SKIP_LOG_FILE
            if SKIP_LOG_FILE.exists():
                content = SKIP_LOG_FILE.read_text(encoding="utf-8")
                self._skip_log_view.setPlainText(content or "（スキップされたファイルはありません）")
            else:
                self._skip_log_view.setPlainText("（スキップログなし）")
        except Exception:
            pass
