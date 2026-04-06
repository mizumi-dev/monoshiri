"""
モノシリ チャット画面
質問入力・回答表示・エビデンス表示・履歴タブ・類似度スライダーを提供する。
"""
from __future__ import annotations
import json
import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st

from core.config import (
    load_config, save_config,
    HISTORY_FILE, DEFAULT_SIMILARITY, TOP_K,
)

logger = logging.getLogger(__name__)


# ─── 履歴管理 ─────────────────────────────────────────────────

def load_history() -> list[dict]:
    """チャット履歴を読み込む"""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_to_history(question: str, answer: str, evidence: list[dict]) -> None:
    """1件の質問・回答を履歴に追記する"""
    history = load_history()
    history.append({
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "evidence": evidence,
    })
    # 最大1000件保持
    if len(history) > 1000:
        history = history[-1000:]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def clear_history() -> None:
    """履歴を全削除する"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)


# ─── エビデンス表示コンポーネント ─────────────────────────────

def render_evidence(evidence: list[dict]) -> None:
    """エビデンス（出典情報）を表示する。クリックでファイルを開く。"""
    if not evidence:
        return

    with st.expander(f"📎 参照元 ({len(evidence)}件)", expanded=True):
        for i, ev in enumerate(evidence):
            file_path_str = ev.get("file_path", "")
            file_name = ev.get("file_name", "不明")
            page = ev.get("page")
            slide = ev.get("slide")
            similarity = ev.get("similarity", 0.0)

            # 表示ラベル
            label = file_name
            if page:
                label += f"  ▶  p.{page}"
            elif slide:
                label += f"  ▶  スライド{slide}"

            col_file, col_sim = st.columns([4, 1])

            with col_file:
                file_path = Path(file_path_str) if file_path_str else None
                file_exists = file_path is not None and file_path.exists()

                if file_exists:
                    if st.button(
                        f"📄 {label}",
                        key=f"ev_{i}_{hash(file_path_str)}",
                        use_container_width=True,
                        help=f"クリックしてファイルを開く: {file_path_str}",
                    ):
                        _open_file(file_path_str)
                else:
                    st.text(f"📄 {label}")
                    if file_path_str:
                        st.caption(f"  {file_path_str}")

            with col_sim:
                sim_pct = int(similarity * 100)
                st.metric("類似度", f"{sim_pct}%", label_visibility="collapsed")
                st.progress(similarity)


def _open_file(file_path: str) -> None:
    """OSのデフォルトアプリでファイルを開く"""
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", file_path])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", file_path])
        else:
            subprocess.Popen(["xdg-open", file_path])
    except Exception as e:
        st.warning(f"ファイルを開けませんでした: {e}")


# ─── チャットタブ ──────────────────────────────────────────────

def _render_chat_tab(config: dict) -> None:
    selected_model = config.get("selected_model", "")
    folders = config.get("folders", [])
    similarity = config.get("similarity_threshold", DEFAULT_SIMILARITY)

    # メイン / サイドコントロール
    col_chat, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown("#### 検索設定")

        new_sim = st.slider(
            "類似度の閾値",
            min_value=0.0,
            max_value=1.0,
            value=float(similarity),
            step=0.05,
            help="値を下げると広く検索。上げると絞り込み検索になります（デフォルト: 0.5）",
        )
        if abs(new_sim - similarity) > 0.001:
            config["similarity_threshold"] = new_sim
            save_config(config)
            st.session_state.config = config

        st.divider()

        if selected_model:
            # 表示名を短縮
            short_name = selected_model.split("（")[0]
            st.caption(f"🤖 {short_name}")
        else:
            st.warning("モデル未設定")

        if not folders:
            st.warning("フォルダ未登録")

        # インデックス情報
        try:
            from core.indexer import get_index_stats
            stats = get_index_stats()
            if stats["total_chunks"] > 0:
                st.metric("インデックス済み", f"{stats['total_chunks']:,} チャンク")
        except Exception:
            pass

    with col_chat:
        # チャット履歴表示（セッション内）
        for msg in st.session_state.get("chat_messages", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("evidence"):
                    render_evidence(msg["evidence"])

        # 入力が無効な場合の警告
        disabled = not selected_model or not folders or st.session_state.get("indexing", False)

        if not selected_model:
            st.info("💡 まず **設定** 画面でAIモデルをダウンロードしてください。")
        elif not folders:
            st.info("💡 まず **インデックス管理** 画面でフォルダを追加してください。")

        # チャット入力
        if prompt := st.chat_input(
            "社内資料について質問してください...",
            disabled=disabled,
        ):
            # ユーザーメッセージを追加
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # 回答生成
            with st.chat_message("assistant"):
                with st.spinner("🔍 文書を検索中..."):
                    try:
                        from core.rag import answer_question
                        result = answer_question(
                            question=prompt,
                            model_name=selected_model,
                            similarity_threshold=new_sim,
                            top_k=TOP_K,
                        )

                        st.markdown(result["answer"])
                        render_evidence(result["evidence"])

                        # セッション履歴に追加
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "evidence": result["evidence"],
                        })

                        # 永続履歴に保存
                        save_to_history(prompt, result["answer"], result["evidence"])

                    except Exception as e:
                        error_msg = f"エラーが発生しました:\n\n```\n{str(e)[:400]}\n```"
                        st.error(error_msg)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "evidence": [],
                        })

        # 会話クリアボタン
        if st.session_state.get("chat_messages"):
            if st.button("🗑️ 会話をクリア", key="clear_chat_btn"):
                st.session_state.chat_messages = []
                st.rerun()


# ─── 履歴タブ ──────────────────────────────────────────────────

def _render_history_tab() -> None:
    st.subheader("📜 過去の質問履歴")

    history = load_history()

    if not history:
        st.info("まだ質問履歴がありません。チャットで質問すると、ここに記録されます。")
        return

    # キーワード検索
    keyword = st.text_input(
        "🔎 キーワードで絞り込み",
        placeholder="例：品質基準 / 作業手順",
        key="history_search",
    )

    filtered = history
    if keyword:
        filtered = [
            h for h in history
            if keyword in h.get("question", "") or keyword in h.get("answer", "")
        ]

    st.caption(f"{len(filtered):,}件の履歴（最新50件を表示）")

    if not filtered:
        st.warning(f"「{keyword}」に該当する履歴がありません")
        return

    # 新しい順に最大50件表示
    for item in reversed(filtered[-50:]):
        ts = item.get("timestamp", "")
        q = item.get("question", "")
        a = item.get("answer", "")
        ev = item.get("evidence", [])

        # タイムスタンプを読みやすく整形
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(ts)
            ts_str = dt.strftime("%Y/%m/%d %H:%M")
        except Exception:
            ts_str = ts[:16].replace("T", " ")

        # 質問を短縮してタイトルに
        q_short = q[:60] + "..." if len(q) > 60 else q
        title = f"🕐 {ts_str}  |  {q_short}"

        with st.expander(title):
            st.markdown(f"**❓ 質問：** {q}")
            st.divider()
            st.markdown(f"**💬 回答：**\n\n{a}")
            if ev:
                render_evidence(ev)

    st.divider()
    if st.button("🗑️ 履歴をすべて削除", key="clear_all_history"):
        clear_history()
        st.success("履歴を削除しました")
        st.rerun()


# ─── メイン ───────────────────────────────────────────────────

def render_chat_page() -> None:
    """チャット画面をレンダリングする"""
    config = st.session_state.get("config", load_config())

    tab_chat, tab_history = st.tabs(["💬 チャット", "📜 履歴"])

    with tab_chat:
        _render_chat_tab(config)

    with tab_history:
        _render_history_tab()
