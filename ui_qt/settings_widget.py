"""
モノシリ 設定ウィジェット
4タブ構成: AIモデル / GPU・Ollama / ネットワーク / システム情報
"""
from __future__ import annotations
import logging
import os
import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTabWidget, QGroupBox, QScrollArea, QFrame, QLineEdit,
    QComboBox, QProgressBar, QTextEdit, QMessageBox,
    QFileDialog, QSizePolicy, QGridLayout, QSpacerItem,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QFont

from ui_qt.workers import (
    InstalledModelsWorker, RecommendedModelsWorker,
    MemInfoWorker, CudaInfoWorker, OllamaCheckWorker,
    OllamaRegisterWorker, NetworkCheckWorker, IndexStatsWorker,
)

logger = logging.getLogger(__name__)


# ─── 汎用ボタンスタイル ───────────────────────────────────────────

def _lighten(color: str) -> str:
    c = color.lstrip("#")
    if len(c) == 3:
        c = c[0]*2 + c[1]*2 + c[2]*2
    r = min(255, int(c[0:2], 16) + 25)
    g = min(255, int(c[2:4], 16) + 25)
    b = min(255, int(c[4:6], 16) + 25)
    return f"#{r:02x}{g:02x}{b:02x}"

def _btn(label: str, color: str = "#3b4a82", small: bool = False) -> QPushButton:
    btn = QPushButton(label)
    h = "28px" if small else "32px"
    hover_bg = _lighten(color)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {color}; color: white; border: none;
            border-radius: 5px; padding: 0 12px; height: {h};
            font-size: {"11px" if small else "12px"};
        }}
        QPushButton:hover {{ background-color: {hover_bg}; }}
        QPushButton:disabled {{ background-color: #aaa; }}
    """)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    return btn


def _group(title: str) -> QGroupBox:
    g = QGroupBox(title)
    g.setStyleSheet("""
        QGroupBox {
            font-size: 13px; font-weight: bold; color: #1a1a2e;
            border: 1px solid #dde0f0; border-radius: 8px;
            margin-top: 8px; padding-top: 8px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
    """)
    return g


# ─── メイン設定ウィジェット ──────────────────────────────────────

class SettingsWidget(QWidget):
    """設定画面（4タブ）"""

    def __init__(self, config: dict, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._build_ui()

        # ダウンロードキュー・Ollama状態の定期更新（起動5秒後から開始）
        self._dl_timer = QTimer(self)
        self._dl_timer.setInterval(1000)
        self._dl_timer.timeout.connect(self._refresh_download_queue)
        QTimer.singleShot(5000, self._dl_timer.start)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 12)
        root.setSpacing(12)

        title = QLabel("⚙️ 設定")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #1a1a2e;")
        root.addWidget(title)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #dde0f0; border-radius: 6px; background: white; }
            QTabBar::tab {
                background: #e8eaf4; color: #555; padding: 8px 18px;
                border-top-left-radius: 6px; border-top-right-radius: 6px;
                margin-right: 2px; font-size: 12px;
            }
            QTabBar::tab:selected { background: white; color: #1a1a2e; font-weight: bold; }
        """)
        root.addWidget(self._tabs, 1)

        self._tabs.addTab(self._build_model_tab(), "🤖 AIモデル")
        self._tabs.addTab(self._build_gpu_tab(), "⚡ GPU・Ollama")
        self._tabs.addTab(self._build_network_tab(), "🌐 ネットワーク")
        self._tabs.addTab(self._build_system_tab(), "ℹ️ システム情報")

    # ════════════════════════════════════════════════════════════
    # タブ1: AIモデル
    # ════════════════════════════════════════════════════════════

    def _build_model_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # 説明
        info_lbl = QLabel(
            "💬 <b>回答AI（LLM）</b> — 質問に対して回答文を生成します。<br>"
            "🔍 <b>検索AI（Embedding）</b> — 社内資料をベクトル化して関連文書を検索します（自動管理）。"
        )
        info_lbl.setStyleSheet("font-size: 12px; color: #4a5080; background: #f0f4ff; "
                               "border: 1px solid #c8d0ee; border-radius: 6px; padding: 10px;")
        info_lbl.setWordWrap(True)
        layout.addWidget(info_lbl)

        # ダウンロードキュー
        self._dl_queue_group = _group("📥 ダウンロードキュー")
        self._dl_queue_layout = QVBoxLayout()
        self._dl_queue_layout.setContentsMargins(12, 16, 12, 12)
        self._dl_queue_layout.setSpacing(6)
        self._dl_queue_group.setLayout(self._dl_queue_layout)
        self._dl_queue_group.setVisible(False)
        layout.addWidget(self._dl_queue_group)

        # インストール済みモデル
        installed_group = _group("✅ インストール済みモデル")
        installed_v = QVBoxLayout()
        installed_v.setContentsMargins(12, 16, 12, 12)
        installed_v.setSpacing(6)
        self._installed_models_layout = installed_v
        installed_group.setLayout(installed_v)
        layout.addWidget(installed_group)
        # インストール済みモデルは遅延ロード（バックグラウンドスレッドで実行）
        QTimer.singleShot(800, self._refresh_installed_models_async)

        # モデル追加
        add_group = _group("➕ 回答AIモデルを追加")
        add_v = QVBoxLayout()
        add_v.setContentsMargins(12, 16, 12, 12)
        add_v.setSpacing(12)
        add_group.setLayout(add_v)

        # おすすめモデルセクションのコンテナ（遅延ロード用）
        self._recommended_container = QWidget()
        self._recommended_container_layout = QVBoxLayout(self._recommended_container)
        self._recommended_container_layout.setContentsMargins(0, 0, 0, 0)
        self._recommended_container_layout.addWidget(QLabel("⭐ おすすめモデル（読み込み中...）"))
        add_v.addWidget(self._recommended_container)
        QTimer.singleShot(1000, self._load_recommended_section_async)

        add_v.addWidget(self._build_hf_download_section())
        add_v.addWidget(self._build_local_add_section())
        layout.addWidget(add_group)

        # Embeddingモデル状態
        emb_group = _group("🔍 検索AI（Embedding）")
        emb_v = QVBoxLayout()
        emb_v.setContentsMargins(12, 16, 12, 12)
        emb_v.setSpacing(4)
        emb_group.setLayout(emb_v)
        self._build_embedding_status(emb_v)
        layout.addWidget(emb_group)

        layout.addStretch()
        scroll.setWidget(inner)
        return scroll

    @pyqtSlot()
    def _refresh_installed_models_async(self) -> None:
        w = InstalledModelsWorker(self._config)
        w._callback = self._on_installed_models_loaded
        w.start()

    @pyqtSlot(list)
    def _on_installed_models_loaded(self, models: list) -> None:
        while self._installed_models_layout.count():
            item = self._installed_models_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not models:
            lbl = QLabel("📥 インストール済みのモデルがありません。下の「回答AIモデルを追加」からダウンロードしてください。")
            lbl.setStyleSheet("color: #888; font-size: 12px; padding: 8px;")
            lbl.setWordWrap(True)
            self._installed_models_layout.addWidget(lbl)
            return
        for model_name, model_info, is_selected in models:
            row = self._build_model_row(model_name, model_info, is_selected)
            self._installed_models_layout.addWidget(row)

    @pyqtSlot()
    def _load_recommended_section_async(self) -> None:
        w = RecommendedModelsWorker(self._config)
        w._callback = self._on_recommended_loaded
        w.start()

    @pyqtSlot(list)
    def _on_recommended_loaded(self, models: list) -> None:
        while self._recommended_container_layout.count():
            item = self._recommended_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        header = QLabel("⭐ おすすめモデル")
        header.setStyleSheet("font-weight: bold; color: #1a1a2e; font-size: 12px;")
        self._recommended_container_layout.addWidget(header)
        if not models:
            self._recommended_container_layout.addWidget(QLabel("（追加可能なモデルはありません）"))
            return
        try:
            from core.llm import get_total_ram_gb
            total_ram = get_total_ram_gb()
        except Exception:
            total_ram = 99
        for model_name, model_info, is_rec, is_queued in models:
            frame = QFrame()
            frame.setStyleSheet("QFrame { background: white; border: 1px solid #dde0f0; border-radius: 6px; }")
            row = QHBoxLayout(frame)
            row.setContentsMargins(10, 6, 10, 6)
            badge = "⭐ " if is_rec else ""
            name_lbl = QLabel(f"{badge}<b>{model_name}</b>")
            name_lbl.setStyleSheet("font-size: 12px;")
            size = model_info.get("size_gb", "?")
            min_ram = model_info.get("min_ram_gb", "?")
            warn = " ⚠️ RAM不足の可能性" if isinstance(min_ram, (int, float)) and total_ram < min_ram else ""
            detail = QLabel(f"サイズ:{size}GB / RAM:{min_ram}GB以上{warn}")
            detail.setStyleSheet("font-size: 10px; color: #888;")
            info_col = QVBoxLayout()
            info_col.setSpacing(1)
            info_col.addWidget(name_lbl)
            info_col.addWidget(detail)
            row.addLayout(info_col, 1)
            if is_queued:
                q_btn = _btn("待機中", "#888", small=True)
                q_btn.setEnabled(False)
                row.addWidget(q_btn)
            else:
                dl_btn = _btn("＋ DL", "#3b4a82", small=True)
                dl_btn.clicked.connect(lambda _, n=model_name: self._queue_download(n))
                row.addWidget(dl_btn)
            self._recommended_container_layout.addWidget(frame)

    def _build_model_row(self, model_name: str, model_info: dict, is_selected: bool) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { background: #f0f4ff; border: 1px solid #c8d0ee; border-radius: 6px; }"
            if is_selected else
            "QFrame { background: white; border: 1px solid #dde0f0; border-radius: 6px; }"
        )
        row = QHBoxLayout(frame)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)

        icon = "🟢" if is_selected else "✅"
        name_lbl = QLabel(f"{icon} <b>{model_name}</b>")
        name_lbl.setStyleSheet("font-size: 12px;")
        size = model_info.get("size_gb", "?")
        ram = model_info.get("min_ram_gb", "?")
        speed = model_info.get("speed", "?")
        desc = model_info.get("description", "")
        detail_lbl = QLabel(f"{desc}<br><span style='color:#888;font-size:10px;'>サイズ:{size}GB / RAM:{ram}GB以上 / 速度:{speed}</span>")
        detail_lbl.setStyleSheet("font-size: 12px; color: #4a5080;")
        detail_lbl.setWordWrap(True)

        info_col = QVBoxLayout()
        info_col.setSpacing(2)
        info_col.addWidget(name_lbl)
        info_col.addWidget(detail_lbl)
        row.addLayout(info_col, 1)

        if not is_selected:
            select_btn = _btn("選択", "#3b4a82", small=True)
            select_btn.clicked.connect(lambda _, n=model_name: self._select_model(n))
            row.addWidget(select_btn)

            del_btn = _btn("🗑️", "#e74c3c", small=True)
            del_btn.setFixedWidth(36)
            del_btn.setToolTip("アンインストール")
            del_btn.clicked.connect(lambda _, n=model_name: self._delete_model(n))
            row.addWidget(del_btn)
        else:
            using_btn = _btn("使用中", "#27ae60", small=True)
            using_btn.setEnabled(False)
            row.addWidget(using_btn)

        return frame

    def _build_hf_download_section(self) -> QWidget:
        """HuggingFaceからダウンロードフォーム"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QLabel("🤗 HuggingFaceからダウンロード")
        header.setStyleSheet("font-weight: bold; color: #1a1a2e; font-size: 12px;")
        layout.addWidget(header)

        form = QFrame()
        form.setStyleSheet("QFrame { background: white; border: 1px solid #dde0f0; border-radius: 6px; }")
        form_v = QVBoxLayout(form)
        form_v.setContentsMargins(10, 10, 10, 10)
        form_v.setSpacing(6)

        self._hf_repo = QLineEdit()
        self._hf_repo.setPlaceholderText("リポジトリID  例: mmnga/Llama-3-ELYZA-JP-8B-GGUF")
        self._hf_repo.setStyleSheet("border: 1px solid #dde0f0; border-radius: 4px; padding: 5px 8px;")
        form_v.addWidget(QLabel("リポジトリID:"))
        form_v.addWidget(self._hf_repo)

        self._hf_filename = QLineEdit()
        self._hf_filename.setPlaceholderText("ファイル名  例: Llama-3-ELYZA-JP-8B-q4_k_m.gguf")
        self._hf_filename.setStyleSheet("border: 1px solid #dde0f0; border-radius: 4px; padding: 5px 8px;")
        form_v.addWidget(QLabel("ファイル名（.gguf）:"))
        form_v.addWidget(self._hf_filename)

        self._hf_display = QLineEdit()
        self._hf_display.setPlaceholderText("表示名（任意）  例: ELYZA-JP-8B")
        self._hf_display.setStyleSheet("border: 1px solid #dde0f0; border-radius: 4px; padding: 5px 8px;")
        form_v.addWidget(QLabel("表示名:"))
        form_v.addWidget(self._hf_display)

        hf_btn = _btn("＋ キューに追加", "#3b4a82")
        hf_btn.clicked.connect(self._queue_hf_download)
        form_v.addWidget(hf_btn)

        layout.addWidget(form)
        return widget

    def _build_local_add_section(self) -> QWidget:
        """ローカルGGUFファイル追加フォーム"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QLabel("📂 ローカルファイルから追加")
        header.setStyleSheet("font-weight: bold; color: #1a1a2e; font-size: 12px;")
        layout.addWidget(header)

        form = QFrame()
        form.setStyleSheet("QFrame { background: white; border: 1px solid #dde0f0; border-radius: 6px; }")
        form_v = QVBoxLayout(form)
        form_v.setContentsMargins(10, 10, 10, 10)
        form_v.setSpacing(6)

        path_row = QHBoxLayout()
        self._local_path = QLineEdit()
        self._local_path.setPlaceholderText("GGUFファイルのパス  例: C:\\Users\\田中\\Downloads\\model.gguf")
        self._local_path.setStyleSheet("border: 1px solid #dde0f0; border-radius: 4px; padding: 5px 8px;")
        browse_btn = _btn("参照", "#666", small=True)
        browse_btn.clicked.connect(self._browse_gguf)
        path_row.addWidget(self._local_path, 1)
        path_row.addWidget(browse_btn)

        self._local_display = QLineEdit()
        self._local_display.setPlaceholderText("表示名（任意）")
        self._local_display.setStyleSheet("border: 1px solid #dde0f0; border-radius: 4px; padding: 5px 8px;")

        form_v.addWidget(QLabel("GGUFファイルのパス:"))
        form_v.addLayout(path_row)
        form_v.addWidget(QLabel("表示名:"))
        form_v.addWidget(self._local_display)

        add_btn = _btn("➕ 追加", "#3b4a82")
        add_btn.clicked.connect(self._add_local_gguf)
        form_v.addWidget(add_btn)

        layout.addWidget(form)
        return widget

    def _build_embedding_status(self, layout: QVBoxLayout) -> None:
        try:
            from core.config import EMBEDDING_MODEL_DIR, EMBEDDING_MODEL_ID
            ready = EMBEDDING_MODEL_DIR.exists() and any(EMBEDDING_MODEL_DIR.iterdir())
            if ready:
                size_mb = sum(f.stat().st_size for f in EMBEDDING_MODEL_DIR.rglob("*") if f.is_file()) / (1024 ** 2)
                lbl = QLabel(f"✅ multilingual-e5-large（{size_mb:.0f}MB）— 使用中")
                lbl.setStyleSheet("color: #27ae60; font-size: 12px;")
            else:
                lbl = QLabel("⚠️ 未インストール（インデックス作成時に自動ダウンロードされます）")
                lbl.setStyleSheet("color: #e67e22; font-size: 12px;")
            layout.addWidget(lbl)
        except Exception:
            layout.addWidget(QLabel("Embedding状態の取得エラー"))

    # ════════════════════════════════════════════════════════════
    # タブ2: GPU・Ollama
    # ════════════════════════════════════════════════════════════

    def _build_gpu_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Ollama ステータス
        ollama_group = _group("🦙 Ollama ステータス")
        ollama_v = QVBoxLayout()
        ollama_v.setContentsMargins(12, 16, 12, 12)
        ollama_v.setSpacing(8)
        ollama_group.setLayout(ollama_v)

        status_row = QHBoxLayout()
        self._ollama_status_lbl = QLabel("確認中...")
        self._ollama_status_lbl.setStyleSheet("font-size: 13px;")
        refresh_btn = _btn("🔄 再確認", "#666", small=True)
        refresh_btn.clicked.connect(self._check_ollama)
        status_row.addWidget(self._ollama_status_lbl)
        status_row.addStretch()
        status_row.addWidget(refresh_btn)
        ollama_v.addLayout(status_row)

        hint_lbl = QLabel(
            "Ollamaが未起動の場合: タスクトレイのOllamaアイコンをクリックして起動してください。\n"
            "未インストールの場合: https://ollama.com からダウンロードしてください。"
        )
        hint_lbl.setStyleSheet("font-size: 11px; color: #888;")
        hint_lbl.setWordWrap(True)
        ollama_v.addWidget(hint_lbl)

        # モデル登録状態
        self._ollama_models_layout = QVBoxLayout()
        self._ollama_models_layout.setSpacing(4)
        ollama_v.addLayout(self._ollama_models_layout)
        layout.addWidget(ollama_group)

        # Embedding デバイス設定
        emb_group = _group("🔍 検索AI（Embedding）デバイス設定")
        emb_v = QVBoxLayout()
        emb_v.setContentsMargins(12, 16, 12, 12)
        emb_v.setSpacing(8)
        emb_group.setLayout(emb_v)

        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("デバイス:"))
        self._emb_device_combo = QComboBox()
        self._emb_device_combo.addItems(["auto (自動)", "cuda (GPU強制)", "cpu (CPU強制)"])
        self._emb_device_combo.setStyleSheet("border: 1px solid #dde0f0; border-radius: 4px; padding: 4px;")
        current_dev = self._config.get("embedding_device", "auto")
        idx = {"auto": 0, "cuda": 1, "cpu": 2}.get(current_dev, 0)
        self._emb_device_combo.setCurrentIndex(idx)
        save_emb_btn = _btn("保存", "#3b4a82", small=True)
        save_emb_btn.clicked.connect(self._save_emb_device)
        dev_row.addWidget(self._emb_device_combo)
        dev_row.addWidget(save_emb_btn)
        dev_row.addStretch()
        emb_v.addLayout(dev_row)

        # GPU情報（遅延ロード）
        self._cuda_info_lbl = QLabel("GPU情報を確認中...")
        self._cuda_info_lbl.setStyleSheet("font-size: 11px; color: #888;")
        emb_v.addWidget(self._cuda_info_lbl)
        layout.addWidget(emb_group)

        layout.addStretch()
        scroll.setWidget(inner)

        # 初期チェック（遅延実行）
        QTimer.singleShot(1500, self._check_ollama)
        QTimer.singleShot(2000, self._load_cuda_info)
        return scroll

    @pyqtSlot()
    def _load_mem_info(self) -> None:
        w = MemInfoWorker()
        w._callback = self._mem_info_lbl.setText
        w.start()

    @pyqtSlot()
    def _load_cuda_info(self) -> None:
        w = CudaInfoWorker()
        w._callback = self._apply_cuda_info
        w.start()

    @pyqtSlot(str, str)
    def _apply_cuda_info(self, text: str, style: str) -> None:
        self._cuda_info_lbl.setText(text)
        self._cuda_info_lbl.setStyleSheet(style)

    @pyqtSlot()
    def _check_ollama(self) -> None:
        self._ollama_status_lbl.setText("確認中...")
        w = OllamaCheckWorker()
        w._callback = self._on_ollama_checked
        w.start()

    @pyqtSlot(bool, list)
    def _on_ollama_checked(self, running: bool, reg_list: list) -> None:
        if running:
            self._ollama_status_lbl.setText("🟢 Ollama 起動中")
            self._ollama_status_lbl.setStyleSheet("font-size: 13px; color: #27ae60;")
        else:
            self._ollama_status_lbl.setText("🔴 Ollama 未起動")
            self._ollama_status_lbl.setStyleSheet("font-size: 13px; color: #e74c3c;")

        while self._ollama_models_layout.count():
            item = self._ollama_models_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for model_name, registered in reg_list:
            short = model_name.split("（")[0]
            row = QHBoxLayout()
            icon = "✅" if registered else "⬜"
            row.addWidget(QLabel(f"{icon} {short}"))
            row_w = QWidget()
            row_w.setLayout(row)
            if not registered:
                reg_btn = _btn("Ollamaに登録", "#3b4a82", small=True)
                reg_btn.clicked.connect(lambda _, n=model_name: self._register_model(n))
                row.addWidget(reg_btn)
            else:
                rereg_btn = _btn("🔄 再登録", "#666", small=True)
                rereg_btn.clicked.connect(lambda _, n=model_name: self._register_model(n))
                row.addWidget(rereg_btn)
            row.addStretch()
            self._ollama_models_layout.addWidget(row_w)

    @pyqtSlot(str)
    def _register_model(self, model_name: str) -> None:
        worker = OllamaRegisterWorker(model_name)
        worker._callback = self._on_register_result
        worker.start()

    @pyqtSlot(str, bool, str)
    def _on_register_result(self, model_name: str, success: bool, msg: str) -> None:
        short = model_name.split("（")[0]
        if success:
            QMessageBox.information(self, "登録完了", f"✅ {short} を Ollama に登録しました。")
        else:
            QMessageBox.warning(self, "登録失敗", f"❌ 登録失敗: {short}\n{msg}")
        QTimer.singleShot(500, self._check_ollama)

    @pyqtSlot()
    def _save_emb_device(self) -> None:
        device_map = {0: "auto", 1: "cuda", 2: "cpu"}
        device = device_map.get(self._emb_device_combo.currentIndex(), "auto")
        self._config["embedding_device"] = device
        try:
            from core.config import save_config
            import core.config as app_config
            save_config(self._config)
            app_config.EMBEDDING_DEVICE = device
            try:
                from core.embedder import reset_model
                reset_model()
            except Exception:
                pass
            QMessageBox.information(self, "保存完了", f"デバイス設定を保存しました: {device}")
        except Exception as e:
            QMessageBox.warning(self, "保存エラー", str(e))

    # ════════════════════════════════════════════════════════════
    # タブ3: ネットワーク
    # ════════════════════════════════════════════════════════════

    def _build_network_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # 接続テスト
        conn_group = _group("🌐 ネットワーク診断")
        conn_v = QVBoxLayout()
        conn_v.setContentsMargins(12, 16, 12, 12)
        conn_v.setSpacing(8)
        conn_group.setLayout(conn_v)

        test_btn = _btn("接続テスト実行", "#3b4a82")
        test_btn.setFixedWidth(160)
        test_btn.clicked.connect(self._test_network)
        conn_v.addWidget(test_btn)

        self._net_result_lbl = QLabel("")
        self._net_result_lbl.setWordWrap(True)
        self._net_result_lbl.setStyleSheet("font-size: 12px;")
        conn_v.addWidget(self._net_result_lbl)
        layout.addWidget(conn_group)

        # HF ミラー設定
        mirror_group = _group("🔗 HuggingFaceミラー設定")
        mirror_v = QVBoxLayout()
        mirror_v.setContentsMargins(12, 16, 12, 12)
        mirror_v.setSpacing(8)
        mirror_group.setLayout(mirror_v)

        mirror_desc = QLabel(
            "HuggingFace本体に接続できない場合、ミラーサイトを利用できます。\n"
            "空欄にすると通常の https://huggingface.co を使用します。"
        )
        mirror_desc.setStyleSheet("font-size: 11px; color: #666;")
        mirror_desc.setWordWrap(True)
        mirror_v.addWidget(mirror_desc)

        mirror_row = QHBoxLayout()
        self._mirror_input = QLineEdit()
        try:
            import core.config as app_config
            self._mirror_input.setText(app_config.HF_MIRROR_URL or "")
        except Exception:
            pass
        self._mirror_input.setPlaceholderText("例: https://hf-mirror.com")
        self._mirror_input.setStyleSheet("border: 1px solid #dde0f0; border-radius: 4px; padding: 5px 8px;")
        save_mirror_btn = _btn("保存", "#3b4a82", small=True)
        save_mirror_btn.clicked.connect(self._save_mirror)
        mirror_row.addWidget(self._mirror_input, 1)
        mirror_row.addWidget(save_mirror_btn)
        mirror_v.addLayout(mirror_row)
        layout.addWidget(mirror_group)

        layout.addStretch()
        scroll.setWidget(inner)
        return scroll

    @pyqtSlot()
    def _test_network(self) -> None:
        self._net_result_lbl.setText("⏳ 接続確認中...")
        worker = NetworkCheckWorker()
        worker._callback = self._on_net_result
        worker.start()

    @pyqtSlot(bool, str)
    def _on_net_result(self, ok: bool, msg: str) -> None:
        if ok:
            self._net_result_lbl.setText(f"✅ {msg}")
            self._net_result_lbl.setStyleSheet("font-size: 12px; color: #27ae60;")
        else:
            self._net_result_lbl.setText(f"❌ 接続失敗\n{msg[:300]}")
            self._net_result_lbl.setStyleSheet("font-size: 12px; color: #e74c3c;")

    @pyqtSlot()
    def _save_mirror(self) -> None:
        mirror_url = self._mirror_input.text().strip()
        try:
            import core.config as app_config
            from core.config import save_config
            app_config.HF_MIRROR_URL = mirror_url
            self._config["hf_mirror_url"] = mirror_url
            save_config(self._config)
            QMessageBox.information(self, "保存完了", f"ミラーURL を保存しました: {mirror_url or '（デフォルト）'}")
        except Exception as e:
            QMessageBox.warning(self, "保存エラー", str(e))

    # ════════════════════════════════════════════════════════════
    # タブ4: システム情報
    # ════════════════════════════════════════════════════════════

    def _build_system_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # メモリ情報
        mem_group = _group("💻 メモリ情報")
        mem_v = QVBoxLayout()
        mem_v.setContentsMargins(12, 16, 12, 12)
        mem_group.setLayout(mem_v)

        self._mem_info_lbl = QLabel("読み込み中...")
        self._mem_info_lbl.setStyleSheet("font-size: 12px;")
        mem_v.addWidget(self._mem_info_lbl)
        layout.addWidget(mem_group)
        QTimer.singleShot(1500, self._load_mem_info)

        # データパス
        path_group = _group("📂 データ保存場所")
        path_v = QVBoxLayout()
        path_v.setContentsMargins(12, 16, 12, 12)
        path_v.setSpacing(6)
        path_group.setLayout(path_v)

        try:
            from core.config import DATA_DIR, MODELS_DIR, LOGS_DIR
            for label, path in [
                ("データフォルダ:", DATA_DIR),
                ("モデルフォルダ:", MODELS_DIR),
                ("ログフォルダ:", LOGS_DIR),
            ]:
                row = QHBoxLayout()
                row.addWidget(QLabel(label))
                path_lbl = QLabel(str(path))
                path_lbl.setStyleSheet("font-size: 11px; color: #4a5080;")
                path_lbl.setWordWrap(True)
                open_btn = _btn("開く", "#666", small=True)
                open_btn.setFixedWidth(50)
                open_btn.clicked.connect(lambda _, p=path: self._open_folder(p))
                row.addWidget(path_lbl, 1)
                row.addWidget(open_btn)
                path_v.addLayout(row)
        except Exception as e:
            path_v.addWidget(QLabel(f"パス取得エラー: {e}"))
        layout.addWidget(path_group)

        # インデックス統計
        stats_group = _group("📊 インデックス統計")
        stats_v = QVBoxLayout()
        stats_v.setContentsMargins(12, 16, 12, 12)
        stats_group.setLayout(stats_v)

        self._stats_chunks_label = QLabel("読み込み中...")
        self._stats_chunks_label.setStyleSheet("font-size: 12px;")
        stats_v.addWidget(self._stats_chunks_label)
        QTimer.singleShot(1200, self._load_index_stats)

        delete_index_btn = _btn("🗑️ インデックスを全削除", "#e74c3c", small=True)
        delete_index_btn.setFixedWidth(200)
        delete_index_btn.clicked.connect(self._delete_all_index)
        stats_v.addWidget(delete_index_btn)
        layout.addWidget(stats_group)

        layout.addStretch()
        scroll.setWidget(inner)
        return scroll

    @pyqtSlot()
    def _load_index_stats(self) -> None:
        w = IndexStatsWorker(fmt="総チャンク数: <b>{count:,}</b>")
        w._callback = self._stats_chunks_label.setText
        w.start()

    def _open_folder(self, path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
            os.startfile(str(path))
        except Exception as e:
            QMessageBox.warning(self, "エラー", f"フォルダを開けませんでした:\n{e}")

    @pyqtSlot()
    def _delete_all_index(self) -> None:
        reply = QMessageBox.question(
            self, "インデックスを全削除",
            "すべてのインデックスデータを削除しますか？\n"
            "削除後は再インデックスが必要になります。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            from core.indexer import IndexManager
            from core.chromadb_store import get_chroma_client
            from core.config import CHROMA_COLLECTION
            from core.chromadb_store import reset_chroma_client
            client = get_chroma_client()
            client.delete_collection(CHROMA_COLLECTION)
            reset_chroma_client()
            QMessageBox.information(self, "削除完了", "インデックスを全削除しました。")
        except Exception as e:
            QMessageBox.warning(self, "削除エラー", str(e))

    # ════════════════════════════════════════════════════════════
    # ダウンロードキュー（1秒ポーリング）
    # ════════════════════════════════════════════════════════════

    @pyqtSlot()
    def _refresh_download_queue(self) -> None:
        try:
            from core.downloader import DownloadManager, JobStatus, STATUS_LABEL
            dm = DownloadManager.get()
            jobs = dm.get_jobs()

            active = [j for j in jobs if j.status in (JobStatus.PENDING, JobStatus.DOWNLOADING)]

            if not jobs:
                self._dl_queue_group.setVisible(False)
                return

            self._dl_queue_group.setVisible(True)

            # キューウィジェットを再描画
            while self._dl_queue_layout.count():
                item = self._dl_queue_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            for job in jobs:
                frame = QFrame()
                frame.setStyleSheet("QFrame { background: white; border: 1px solid #dde0f0; border-radius: 6px; }")
                row = QHBoxLayout(frame)
                row.setContentsMargins(10, 6, 10, 6)
                row.setSpacing(8)

                short = job.model_name.split("（")[0]
                name_lbl = QLabel(f"<b>{short}</b>")
                name_lbl.setFixedWidth(160)
                row.addWidget(name_lbl)

                status_lbl = QLabel(STATUS_LABEL.get(job.status, job.status))
                status_lbl.setFixedWidth(120)
                row.addWidget(status_lbl)

                if job.status == JobStatus.DOWNLOADING:
                    pb = QProgressBar()
                    pb.setMinimum(0)
                    pb.setMaximum(100)
                    pb.setValue(int(job.progress * 100))
                    pb.setFormat(job.progress_label())
                    pb.setFixedHeight(16)
                    pb.setStyleSheet("""
                        QProgressBar { border: 1px solid #dde0f0; border-radius: 4px; background: #f0f2fa; font-size: 10px; }
                        QProgressBar::chunk { background: #3b4a82; border-radius: 3px; }
                    """)
                    row.addWidget(pb, 1)
                else:
                    row.addStretch(1)

                if job.status == JobStatus.PENDING:
                    cancel_btn = _btn("🚫", "#e74c3c", small=True)
                    cancel_btn.setFixedWidth(32)
                    cancel_btn.setToolTip("キャンセル")
                    cancel_btn.clicked.connect(lambda _, n=job.model_name: self._cancel_download(n))
                    row.addWidget(cancel_btn)
                elif job.status in (JobStatus.FAILED, JobStatus.CANCELLED):
                    retry_btn = _btn("🔄", "#e67e22", small=True)
                    retry_btn.setFixedWidth(32)
                    retry_btn.setToolTip("リトライ")
                    retry_btn.clicked.connect(lambda _, n=job.model_name: self._retry_download(n))
                    row.addWidget(retry_btn)

                self._dl_queue_layout.addWidget(frame)

            # 全完了時に「履歴を消去」ボタン
            if not active and jobs:
                clear_btn = _btn("履歴を消去", "#888", small=True)
                clear_btn.setFixedWidth(120)
                clear_btn.clicked.connect(self._clear_dl_history)
                self._dl_queue_layout.addWidget(clear_btn)

            # DL完了時にインストール済みリストを更新
            if not active and hasattr(self, "_installed_models_layout"):
                self._refresh_installed_models_async()

        except Exception:
            pass

    def _cancel_download(self, model_name: str) -> None:
        try:
            from core.downloader import DownloadManager
            DownloadManager.get().cancel(model_name)
        except Exception:
            pass

    def _retry_download(self, model_name: str) -> None:
        try:
            from core.downloader import DownloadManager
            DownloadManager.get().retry(model_name)
        except Exception:
            pass

    def _clear_dl_history(self) -> None:
        try:
            from core.downloader import DownloadManager
            DownloadManager.get().clear_finished()
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════
    # モデル操作
    # ════════════════════════════════════════════════════════════

    def _select_model(self, model_name: str) -> None:
        self._config["selected_model"] = model_name
        try:
            from core.config import save_config
            save_config(self._config)
        except Exception as e:
            QMessageBox.warning(self, "保存エラー", str(e))
            return
        self._refresh_installed_models_async()
        # メインウィンドウのモデルラベルも更新
        mw = self.window()
        if hasattr(mw, "update_config"):
            mw.update_config(self._config)
        QMessageBox.information(self, "モデル選択", f"✅ {model_name} を選択しました。")

    def _delete_model(self, model_name: str) -> None:
        reply = QMessageBox.question(
            self, "モデルを削除",
            f"以下のモデルをアンインストールしますか？\n{model_name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            from core.llm import get_model_path
            path = get_model_path(model_name)
            if path and path.exists():
                path.unlink()
                if self._config.get("selected_model") == model_name:
                    self._config["selected_model"] = ""
                    from core.config import save_config
                    save_config(self._config)
                self._refresh_installed_models_async()
                QMessageBox.information(self, "削除完了", f"削除しました: {model_name}")
            else:
                QMessageBox.warning(self, "エラー", "モデルファイルが見つかりません。")
        except Exception as e:
            QMessageBox.warning(self, "削除エラー", str(e))

    def _queue_download(self, model_name: str) -> None:
        try:
            from core.downloader import DownloadManager
            added = DownloadManager.get().add(model_name)
            if added:
                QMessageBox.information(self, "DLキュー", f"✅ キューに追加しました:\n{model_name}")
            else:
                QMessageBox.information(self, "DLキュー", "既にキューに入っています。")
        except Exception as e:
            QMessageBox.warning(self, "エラー", str(e))

    @pyqtSlot()
    def _queue_hf_download(self) -> None:
        repo_id = self._hf_repo.text().strip()
        filename = self._hf_filename.text().strip()
        display_name = self._hf_display.text().strip()
        if not repo_id or not filename:
            QMessageBox.warning(self, "入力不足", "リポジトリIDとファイル名を入力してください。")
            return
        if not filename.endswith(".gguf"):
            QMessageBox.warning(self, "形式エラー", "ファイル名は .gguf で終わる必要があります。")
            return
        try:
            from core.config import LLM_MODELS
            from core.downloader import DownloadManager
            name = display_name or filename.replace(".gguf", "")
            key = f"{name}（カスタム）"
            if key not in LLM_MODELS:
                LLM_MODELS[key] = {
                    "repo_id": repo_id, "filename": filename,
                    "size_gb": 0, "min_ram_gb": 8, "speed": "不明",
                    "description": f"HF: {repo_id}/{filename}",
                    "chat_template": "chatml",
                }
            added = DownloadManager.get().add(key)
            if added:
                QMessageBox.information(self, "DLキュー", f"✅ キューに追加しました:\n{key}")
                self._hf_repo.clear()
                self._hf_filename.clear()
                self._hf_display.clear()
            else:
                QMessageBox.information(self, "DLキュー", "既にキューに入っています。")
        except Exception as e:
            QMessageBox.warning(self, "エラー", str(e))

    @pyqtSlot()
    def _browse_gguf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "GGUFファイルを選択", "", "GGUFファイル (*.gguf)"
        )
        if path:
            self._local_path.setText(path)

    @pyqtSlot()
    def _add_local_gguf(self) -> None:
        path_str = self._local_path.text().strip()
        display_name = self._local_display.text().strip() or None
        if not path_str:
            QMessageBox.warning(self, "入力不足", "GGUFファイルのパスを入力してください。")
            return
        gguf_path = Path(path_str)
        if not gguf_path.exists():
            QMessageBox.warning(self, "ファイルなし", f"ファイルが見つかりません:\n{path_str}")
            return
        if gguf_path.suffix.lower() != ".gguf":
            QMessageBox.warning(self, "形式エラー", "GGUFファイル（.gguf）のみ対応しています。")
            return
        try:
            from core.llm import add_local_gguf
            from core.config import save_config
            key = add_local_gguf(gguf_path, display_name)
            self._config["selected_model"] = key
            save_config(self._config)
            self._refresh_installed_models_async()
            QMessageBox.information(self, "追加完了", f"✅ 追加しました: {key}")
            self._local_path.clear()
            self._local_display.clear()
        except Exception as e:
            QMessageBox.warning(self, "追加エラー", str(e))
