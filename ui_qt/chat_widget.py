"""
モノシリ チャットウィジェット
QTextBrowser による HTML メッセージ表示 + QThread ストリーミング
"""
from __future__ import annotations
import json
import logging
import os
import html as _html
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextBrowser,
    QPushButton, QTextEdit, QLabel, QSlider, QSplitter,
    QTabWidget, QListWidget, QListWidgetItem, QLineEdit,
    QFrame, QScrollArea, QSizePolicy, QGroupBox, QToolButton,
    QMessageBox,
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QUrl, QSize,
)
from PyQt6.QtGui import QFont, QTextCursor, QColor, QPalette

logger = logging.getLogger(__name__)

from core.config import HISTORY_FILE  # MONOSHIRI_DATA_DIR 対応のため config 経由で取得


# ─── CSS ───────────────────────────────────────────────────────────

CHAT_HTML_CSS = """
<style>
body {
    font-family: 'Meiryo UI', 'Yu Gothic UI', 'MS UI Gothic', sans-serif;
    font-size: 13px;
    background-color: #f5f6fa;
    margin: 8px 12px;
    line-height: 1.6;
}
.user-row { text-align: right; margin: 8px 0; }
.user-bubble {
    display: inline-block;
    background-color: #3b4a82;
    color: #ffffff;
    border-radius: 14px 14px 2px 14px;
    padding: 10px 14px;
    max-width: 75%;
    text-align: left;
    word-wrap: break-word;
}
.assistant-row { text-align: left; margin: 8px 0; }
.assistant-bubble {
    display: inline-block;
    background-color: #ffffff;
    color: #1a1a2e;
    border-radius: 14px 14px 14px 2px;
    padding: 10px 14px;
    max-width: 80%;
    text-align: left;
    border: 1px solid #dde0f0;
    word-wrap: break-word;
}
.assistant-bubble.streaming {
    border-left: 3px solid #3b4a82;
}
.evidence-box {
    background-color: #f0f4ff;
    border: 1px solid #c8d0ee;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0 8px 0;
    max-width: 80%;
    font-size: 11px;
    color: #4a5080;
}
.evidence-title { font-weight: bold; color: #3b4a82; margin-bottom: 4px; }
.ev-link { color: #3b4a82; text-decoration: none; }
.ev-link:hover { text-decoration: underline; }
.sys-msg {
    text-align: center;
    color: #888;
    font-size: 11px;
    margin: 12px 0;
    font-style: italic;
}
</style>
"""


# ─── バックグラウンドRAGワーカー ──────────────────────────────────

class RagWorker(QThread):
    evidence_ready = pyqtSignal(list)
    token_received = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        question: str,
        model_name: str,
        similarity_threshold: float,
        chat_history: list[dict],
    ) -> None:
        super().__init__()
        self.question = question
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.chat_history = chat_history

    def run(self) -> None:
        try:
            from core.rag import stream_question
            for event in stream_question(
                question=self.question,
                model_name=self.model_name,
                similarity_threshold=self.similarity_threshold,
                chat_history=self.chat_history,
            ):
                if event["type"] == "evidence":
                    self.evidence_ready.emit(event.get("data", []))
                elif event["type"] == "token":
                    token = event.get("data", "")
                    if token:
                        self.token_received.emit(token)
                elif event["type"] == "error":
                    self.error_occurred.emit(event.get("data", "不明なエラー"))
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.finished_signal.emit()


# ─── チャットウィジェット ─────────────────────────────────────────

class ChatWidget(QWidget):
    """メインチャット画面"""

    def __init__(self, config: dict, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._messages: list[dict] = []
        self._pending_answer = ""
        self._pending_evidence: list[dict] = []
        self._streaming = False
        self._worker: Optional[RagWorker] = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(120)
        self._refresh_timer.timeout.connect(self._on_refresh_timer)
        self._needs_refresh = False

        self._build_ui()
        self._load_history()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── タブ（チャット / 履歴） ──────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabWidget::pane { border: none; background: #f5f6fa; }
            QTabBar::tab {
                background: #e8eaf4; color: #555; padding: 8px 20px;
                border-top-left-radius: 6px; border-top-right-radius: 6px;
                margin-right: 2px; font-size: 12px;
            }
            QTabBar::tab:selected { background: #f5f6fa; color: #1a1a2e; font-weight: bold; }
        """)
        root.addWidget(self._tabs)

        # ── チャットタブ ─────────────────────────────────────
        chat_tab = QWidget()
        chat_layout = QVBoxLayout(chat_tab)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)

        # モデル・フォルダ情報バー
        info_bar = QFrame()
        info_bar.setFixedHeight(32)
        info_bar.setStyleSheet("background: #e8eaf4; border-bottom: 1px solid #dde0f0;")
        info_layout = QHBoxLayout(info_bar)
        info_layout.setContentsMargins(12, 0, 12, 0)
        self._model_label = QLabel("🤖 モデル: 未設定")
        self._model_label.setStyleSheet("font-size: 11px; color: #4a5080;")
        self._folder_label = QLabel("📂 フォルダ: —")
        self._folder_label.setStyleSheet("font-size: 11px; color: #4a5080;")
        info_layout.addWidget(self._model_label)
        info_layout.addStretch()
        info_layout.addWidget(self._folder_label)
        chat_layout.addWidget(info_bar)

        # メッセージブラウザ
        self._chat_browser = QTextBrowser()
        self._chat_browser.setOpenLinks(False)
        self._chat_browser.anchorClicked.connect(self._on_link_clicked)
        self._chat_browser.setStyleSheet("""
            QTextBrowser {
                background-color: #f5f6fa;
                border: none;
                padding: 4px;
            }
        """)
        chat_layout.addWidget(self._chat_browser, 1)

        # コントロールバー（類似度スライダー）
        ctrl_bar = QFrame()
        ctrl_bar.setFixedHeight(36)
        ctrl_bar.setStyleSheet("background: #eef0f8; border-top: 1px solid #dde0f0;")
        ctrl_layout = QHBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(12, 0, 12, 0)
        ctrl_layout.setSpacing(8)

        sim_lbl = QLabel("類似度閾値:")
        sim_lbl.setStyleSheet("font-size: 11px; color: #4a5080;")
        ctrl_layout.addWidget(sim_lbl)

        self._sim_slider = QSlider(Qt.Orientation.Horizontal)
        self._sim_slider.setMinimum(10)
        self._sim_slider.setMaximum(90)
        self._sim_slider.setValue(40)
        self._sim_slider.setFixedWidth(120)
        self._sim_slider.setStyleSheet("QSlider::groove:horizontal { height: 4px; background: #c0ccee; border-radius: 2px; } QSlider::handle:horizontal { width: 12px; height: 12px; background: #3b4a82; border-radius: 6px; margin: -4px 0; }")
        self._sim_slider.valueChanged.connect(self._on_sim_changed)
        ctrl_layout.addWidget(self._sim_slider)

        self._sim_value_lbl = QLabel("0.40")
        self._sim_value_lbl.setFixedWidth(32)
        self._sim_value_lbl.setStyleSheet("font-size: 11px; color: #3b4a82;")
        ctrl_layout.addWidget(self._sim_value_lbl)

        ctrl_layout.addStretch()

        clear_btn = QPushButton("🗑️ チャット消去")
        clear_btn.setStyleSheet("background: transparent; color: #888; border: none; font-size: 11px;")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.clicked.connect(self._clear_chat)
        ctrl_layout.addWidget(clear_btn)

        chat_layout.addWidget(ctrl_bar)

        # 入力エリア
        input_frame = QFrame()
        input_frame.setStyleSheet("background: #ffffff; border-top: 1px solid #dde0f0;")
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(12, 8, 12, 8)
        input_layout.setSpacing(6)

        self._input_box = QTextEdit()
        self._input_box.setPlaceholderText("質問を入力してください（Ctrl+Enter で送信）")
        self._input_box.setMaximumHeight(90)
        self._input_box.setMinimumHeight(60)
        self._input_box.setStyleSheet("""
            QTextEdit {
                border: 1px solid #dde0f0;
                border-radius: 8px;
                padding: 8px;
                font-size: 13px;
                background: #fafbff;
            }
            QTextEdit:focus { border: 1px solid #3b4a82; }
        """)
        self._input_box.installEventFilter(self)
        input_layout.addWidget(self._input_box)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._send_btn = QPushButton("送信  ⌨ Ctrl+Enter")
        self._send_btn.setFixedHeight(34)
        self._send_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b4a82;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 0 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #4a5a9e; }
            QPushButton:disabled { background-color: #a0a8c8; }
        """)
        self._send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send_btn.clicked.connect(self._on_send)

        self._stop_btn = QPushButton("⏹ 停止")
        self._stop_btn.setFixedHeight(34)
        self._stop_btn.setVisible(False)
        self._stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 0 16px;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #c0392b; }
        """)
        self._stop_btn.clicked.connect(self._on_stop)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("font-size: 11px; color: #888;")

        btn_row.addStretch()
        btn_row.addWidget(self._status_lbl)
        btn_row.addWidget(self._stop_btn)
        btn_row.addWidget(self._send_btn)
        input_layout.addLayout(btn_row)

        chat_layout.addWidget(input_frame)

        self._tabs.addTab(chat_tab, "💬 チャット")

        # ── 履歴タブ ─────────────────────────────────────────
        history_tab = QWidget()
        hist_layout = QVBoxLayout(history_tab)
        hist_layout.setContentsMargins(12, 12, 12, 12)
        hist_layout.setSpacing(8)

        hist_search_row = QHBoxLayout()
        self._hist_search = QLineEdit()
        self._hist_search.setPlaceholderText("🔍 履歴を検索...")
        self._hist_search.setStyleSheet("border: 1px solid #dde0f0; border-radius: 6px; padding: 6px 10px;")
        self._hist_search.textChanged.connect(self._filter_history)
        hist_search_row.addWidget(self._hist_search)
        hist_clear_btn = QPushButton("全削除")
        hist_clear_btn.setStyleSheet("background: #e8eaf4; border: 1px solid #c0ccee; border-radius: 4px; padding: 4px 10px;")
        hist_clear_btn.clicked.connect(self._delete_all_history)
        hist_search_row.addWidget(hist_clear_btn)
        hist_layout.addLayout(hist_search_row)

        self._history_list = QListWidget()
        self._history_list.setStyleSheet("""
            QListWidget { border: 1px solid #dde0f0; border-radius: 8px; background: white; }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #f0f0f8; }
            QListWidget::item:selected { background: #e8eaf4; color: #1a1a2e; }
        """)
        self._history_list.itemClicked.connect(self._load_history_item)
        hist_layout.addWidget(self._history_list)

        self._tabs.addTab(history_tab, "📜 履歴")

        self._update_info_bar()

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent
        if obj is self._input_box and event.type() == QEvent.Type.KeyPress:
            ke = event
            if ke.key() == Qt.Key.Key_Return and ke.modifiers() == Qt.KeyboardModifier.ControlModifier:
                self._on_send()
                return True
        return super().eventFilter(obj, event)

    # ── モデル・設定更新 ─────────────────────────────────────

    def update_config(self, config: dict) -> None:
        self._config = config
        self._update_info_bar()

    def _update_info_bar(self) -> None:
        model = self._config.get("selected_model", "未設定")
        short = model.split("（")[0] if model else "未設定"
        self._model_label.setText(f"🤖 {short}")
        folders = self._config.get("folders", [])
        self._folder_label.setText(f"📂 {len(folders)} フォルダ")

    # ── 類似度スライダー ─────────────────────────────────────

    @pyqtSlot(int)
    def _on_sim_changed(self, value: int) -> None:
        threshold = value / 100.0
        self._sim_value_lbl.setText(f"{threshold:.2f}")

    def _get_similarity(self) -> float:
        return self._sim_slider.value() / 100.0

    # ── 送信・停止 ──────────────────────────────────────────

    @pyqtSlot()
    def _on_send(self) -> None:
        if self._streaming:
            return
        question = self._input_box.toPlainText().strip()
        if not question:
            return

        model_name = self._config.get("selected_model", "")
        if not model_name:
            QMessageBox.warning(self, "モデル未設定", "設定画面でLLMモデルを選択してください。")
            return

        self._input_box.clear()
        self._append_user_message(question)
        self._start_streaming(question, model_name)

    @pyqtSlot()
    def _on_stop(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(1000)
        self._finish_streaming()

    # ── ストリーミング ────────────────────────────────────────

    def _start_streaming(self, question: str, model_name: str) -> None:
        self._streaming = True
        self._pending_answer = ""
        self._pending_evidence = []
        self._send_btn.setEnabled(False)
        self._stop_btn.setVisible(True)
        self._status_lbl.setText("⏳ 検索中...")
        self._refresh_timer.start()

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in self._messages[-10:]
        ]

        self._worker = RagWorker(
            question=question,
            model_name=model_name,
            similarity_threshold=self._get_similarity(),
            chat_history=history,
        )
        self._worker.evidence_ready.connect(self._on_evidence_ready)
        self._worker.token_received.connect(self._on_token)
        self._worker.finished_signal.connect(self._on_stream_finished)
        self._worker.error_occurred.connect(self._on_stream_error)
        self._worker.start()

    @pyqtSlot(list)
    def _on_evidence_ready(self, evidence: list) -> None:
        self._pending_evidence = evidence
        count = len(evidence)
        self._status_lbl.setText(f"📄 {count} 件の文書を参照中...")
        self._needs_refresh = True

    @pyqtSlot(str)
    def _on_token(self, token: str) -> None:
        self._pending_answer += token
        self._needs_refresh = True

    @pyqtSlot()
    def _on_stream_finished(self) -> None:
        if self._pending_answer or self._pending_evidence:
            self._messages.append({
                "role": "assistant",
                "content": self._pending_answer,
                "evidence": self._pending_evidence,
                "timestamp": datetime.now().isoformat(),
            })
            self._save_history()
        self._finish_streaming()
        self._refresh_chat(force=True)

    @pyqtSlot(str)
    def _on_stream_error(self, error_msg: str) -> None:
        self._messages.append({
            "role": "assistant",
            "content": f"⚠️ エラー: {error_msg}",
            "evidence": [],
            "timestamp": datetime.now().isoformat(),
        })
        self._finish_streaming()
        self._refresh_chat(force=True)

    def _finish_streaming(self) -> None:
        self._streaming = False
        self._pending_answer = ""
        self._pending_evidence = []
        self._refresh_timer.stop()
        self._send_btn.setEnabled(True)
        self._stop_btn.setVisible(False)
        self._status_lbl.setText("")
        self._update_info_bar()

    @pyqtSlot()
    def _on_refresh_timer(self) -> None:
        if self._needs_refresh:
            self._needs_refresh = False
            self._refresh_chat()

    # ── チャット表示 ─────────────────────────────────────────

    def _append_user_message(self, question: str) -> None:
        self._messages.append({
            "role": "user",
            "content": question,
            "evidence": [],
            "timestamp": datetime.now().isoformat(),
        })
        self._refresh_chat(force=True)

    def _refresh_chat(self, force: bool = False) -> None:
        sb = self._chat_browser.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 20

        html = self._build_html()
        self._chat_browser.setHtml(html)

        if at_bottom:
            sb.setValue(sb.maximum())

    def _build_html(self) -> str:
        parts = [f"<html><head>{CHAT_HTML_CSS}</head><body>"]

        if not self._messages and not self._streaming:
            parts.append(
                '<div class="sys-msg">質問を入力してください。<br>'
                'インデックス済みの社内資料から回答します。</div>'
            )

        for msg in self._messages:
            role = msg["role"]
            content = _html.escape(msg["content"]).replace("\n", "<br>")
            evidence = msg.get("evidence", [])

            if role == "user":
                parts.append(
                    f'<div class="user-row">'
                    f'<span class="user-bubble">{content}</span>'
                    f'</div>'
                )
            else:
                parts.append(
                    f'<div class="assistant-row">'
                    f'<span class="assistant-bubble">{content}</span>'
                    f'</div>'
                )
                if evidence:
                    parts.append(self._build_evidence_html(evidence))

        # ストリーミング中の仮メッセージ
        if self._streaming:
            streaming_content = _html.escape(self._pending_answer).replace("\n", "<br>") if self._pending_answer else "▋"
            parts.append(
                f'<div class="assistant-row">'
                f'<span class="assistant-bubble streaming">{streaming_content}</span>'
                f'</div>'
            )
            if self._pending_evidence:
                parts.append(self._build_evidence_html(self._pending_evidence))

        parts.append("</body></html>")
        return "".join(parts)

    def _build_evidence_html(self, evidence: list[dict]) -> str:
        if not evidence:
            return ""
        items = []
        for ev in evidence:
            fname = _html.escape(ev.get("file_name", "不明"))
            fpath = ev.get("file_path", "")
            page = ev.get("page")
            slide = ev.get("slide")
            sim = ev.get("similarity", 0.0)
            sim_pct = int(sim * 100)

            location = ""
            if page:
                location = f" p.{page}"
            elif slide:
                location = f" スライド{slide}"

            if fpath:
                safe_path = _html.escape(fpath).replace("\\", "/")
                link = f'<a href="openfile:///{safe_path}" class="ev-link">📄 {fname}{location}</a>'
            else:
                link = f'📄 {fname}{location}'

            items.append(f'{link} <span style="color:#888;">({sim_pct}%)</span>')

        body = " &nbsp;｜&nbsp; ".join(items)
        return (
            f'<div class="evidence-box">'
            f'<div class="evidence-title">📎 参照文書</div>'
            f'{body}'
            f'</div>'
        )

    # ── リンククリック（ファイルを開く） ─────────────────────

    @pyqtSlot(QUrl)
    def _on_link_clicked(self, url: QUrl) -> None:
        try:
            if url.scheme() == "openfile":
                path_str = url.toString().replace("openfile:///", "", 1)
                path = Path(path_str.replace("/", os.sep))
                if path.exists():
                    os.startfile(str(path))
                else:
                    QMessageBox.warning(self, "ファイルが見つかりません", f"ファイルが見つかりません:\n{path}")
        except Exception as e:
            logger.warning(f"ファイルを開けませんでした: {e}")

    # ── チャット消去 ─────────────────────────────────────────

    @pyqtSlot()
    def _clear_chat(self) -> None:
        self._messages.clear()
        self._refresh_chat(force=True)

    # ── 類似度設定をconfigから読み込む ──────────────────────

    def _load_sim_from_config(self) -> None:
        threshold = self._config.get("similarity_threshold", 0.40)
        val = int(threshold * 100)
        self._sim_slider.setValue(max(10, min(90, val)))

    # ── 履歴保存・読み込み ──────────────────────────────────

    def _save_history(self) -> None:
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            existing = []
            if HISTORY_FILE.exists():
                try:
                    existing = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
                except Exception:
                    existing = []

            # 現在の会話セッションを追加
            if self._messages:
                session = {
                    "timestamp": datetime.now().isoformat(),
                    "messages": self._messages[-20:],  # 最大20件
                }
                existing.append(session)
                # 直近100セッションのみ保持
                existing = existing[-100:]

            HISTORY_FILE.write_text(
                json.dumps(existing, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"履歴保存失敗: {e}")

    def _load_history(self) -> None:
        try:
            if not HISTORY_FILE.exists():
                return
            sessions = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            self._all_history_sessions = sessions
            self._populate_history_list(sessions)
        except Exception:
            self._all_history_sessions = []

    def _populate_history_list(self, sessions: list) -> None:
        self._history_list.clear()
        for session in reversed(sessions):
            ts = session.get("timestamp", "")
            msgs = session.get("messages", [])
            # 旧Streamlit版フォーマット（question/answer）からの後方互換変換
            if not msgs and session.get("question"):
                msgs = [
                    {"role": "user", "content": session.get("question", "")},
                    {"role": "assistant", "content": session.get("answer", ""),
                     "evidence": session.get("evidence", [])},
                ]
                session["messages"] = msgs  # 後続のクリック処理で使えるよう更新
            if not msgs:
                continue
            first_q = next((m["content"] for m in msgs if m.get("role") == "user"), "（不明）")
            short_q = first_q[:50] + "..." if len(first_q) > 50 else first_q
            try:
                dt = datetime.fromisoformat(ts)
                date_str = dt.strftime("%Y/%m/%d %H:%M")
            except Exception:
                date_str = ts[:16]
            item = QListWidgetItem(f"[{date_str}] {short_q}")
            item.setData(Qt.ItemDataRole.UserRole, session)
            self._history_list.addItem(item)

    @pyqtSlot()
    def _filter_history(self) -> None:
        keyword = self._hist_search.text().lower()
        sessions = getattr(self, "_all_history_sessions", [])
        if keyword:
            def matches(s: dict) -> bool:
                # 新フォーマット: messages 内を検索
                if any(keyword in m.get("content", "").lower() for m in s.get("messages", [])):
                    return True
                # 旧フォーマット: question / answer を検索
                if keyword in s.get("question", "").lower():
                    return True
                if keyword in s.get("answer", "").lower():
                    return True
                return False
            filtered = [s for s in sessions if matches(s)]
        else:
            filtered = sessions
        self._populate_history_list(filtered)

    @pyqtSlot(QListWidgetItem)
    def _load_history_item(self, item: QListWidgetItem) -> None:
        session = item.data(Qt.ItemDataRole.UserRole)
        if not session:
            return
        self._messages = list(session.get("messages", []))
        self._refresh_chat(force=True)
        self._tabs.setCurrentIndex(0)

    @pyqtSlot()
    def _delete_all_history(self) -> None:
        reply = QMessageBox.question(
            self, "履歴を全削除",
            "チャット履歴をすべて削除しますか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                HISTORY_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            self._all_history_sessions = []
            self._history_list.clear()
