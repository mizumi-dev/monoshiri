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
    SAMPLE_QUESTIONS,
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

def render_evidence(evidence: list[dict], key_prefix: str = "ev") -> None:
    """エビデンス（出典情報）を表示する。クリックでファイルを開く。

    Args:
        evidence: 出典情報のリスト
        key_prefix: ボタンキーの接頭辞。複数箇所から呼ぶ場合は一意の値を渡す
    """
    if not evidence:
        return

    # key重複を防ぐため、呼び出し回数カウンターを使用
    counter_key = f"_ev_counter_{key_prefix}"
    counter = st.session_state.get(counter_key, 0)
    st.session_state[counter_key] = counter + 1

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

                # key重複防止: prefix + インデックス + 呼び出し回数 + file_path hash
                btn_key = f"{key_prefix}_{i}_{counter}_{hash(file_path_str)}"

                if file_exists:
                    if st.button(
                        f"📄 {label}",
                        key=btn_key,
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
            help="値を下げると広く検索。上げると絞り込み検索になります（デフォルト: 0.4）",
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
        # ─── Free層 使用量バナー ────────────────────────────────
        try:
            from core.usage_tracker import get_usage_summary
            usage = get_usage_summary()
            q_remaining = usage["questions_remaining"]
            q_max       = usage["questions_max"]
            month_str   = usage["month"]

            if usage["questions_limit_reached"]:
                st.warning(
                    f"⚠️ **{month_str}の質問数が上限（{q_max}回）に達しました。**  \n"
                    "来月1日にリセットされます。（β期間中は引き続きご利用いただけます）",
                    icon=None,
                )
            elif q_remaining <= 30:
                st.info(
                    f"📊 {month_str}の残り質問数: **{q_remaining}回** / {q_max}回",
                    icon=None,
                )
        except Exception:
            pass  # 使用量取得失敗はサイレント

        # ─── チャット履歴表示（セッション内）────────────────────
        chat_messages = st.session_state.get("chat_messages", [])
        # BUG-010修正: render_evidence に msg インデックスを含む一意のキーを渡す。
        #   複数ターンで同じファイルが参照元に現れると key が衝突する。
        for i_msg, msg in enumerate(chat_messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("evidence"):
                    render_evidence(msg["evidence"], key_prefix=f"ev_{i_msg}")

        # ─── 入力が無効な場合の警告 ──────────────────────────
        disabled = not selected_model or not folders or st.session_state.get("indexing", False)

        if not selected_model:
            st.info("💡 まず **設定** 画面でAIモデルをダウンロードしてください。")
        elif not folders:
            st.info("💡 まず **インデックス管理** 画面でフォルダを追加してください。")
        elif not chat_messages:
            # ─── 初回表示: 固定サンプル質問3件（v3.4確定文言）──
            st.markdown("#### 💬 何でも聞いてください")
            st.caption("例えば、こんな質問から始めてみてください：")
            for i, sq in enumerate(SAMPLE_QUESTIONS):
                if st.button(f"📄 {sq}", key=f"sample_q_{i}", use_container_width=True):
                    # サンプル質問をクリックした場合、promptとして処理
                    st.session_state["_pending_prompt"] = sq
                    st.rerun()

        # ─── サンプル質問クリック後の処理 ───────────────────────
        pending_prompt = st.session_state.pop("_pending_prompt", None)

        # ─── チャット入力 ─────────────────────────────────────
        prompt = st.chat_input(
            "社内資料について質問してください（2,000文字以内）...",
            disabled=disabled,
        ) or pending_prompt

        if prompt:
            # 2,000文字制限（v3.4: プロンプトインジェクション対策）
            prompt = prompt[:2000]

            # 送信時点で質問数を1カウント（再試行はカウントしない）
            try:
                from core.usage_tracker import increment_question_count, is_question_limit_reached
                if is_question_limit_reached():
                    st.warning("今月の質問数上限に達しています。（β期間中は引き続き利用可）")
                else:
                    increment_question_count()
            except Exception:
                pass

            # ユーザーメッセージをセッション状態に追加
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            # BUG-001修正: LLMに渡す会話履歴を取得（現在の質問の直前まで）
            # 最後の要素（今追加したユーザーメッセージ）は stream_question 内で追加されるため除く
            history_for_llm = st.session_state.chat_messages[:-1]

            # ── ユーザーメッセージを画面に表示（BUG-002修正で削除してしまっていた）──
            with st.chat_message("user"):
                st.markdown(prompt)

            # 回答生成（ストリーミング + BUG-002修正）
            # ストリーミング表示は維持しつつ、完了後にst.rerun()で
            # 履歴ループから一貫したレンダリングに切り替える
            full_answer = ""
            error_msg = None
            evidence_list = []

            # ─── ストリーミング表示エリア ───────────────────────────
            # v3.4仕様: LLM回答生成失敗時に自動再試行（最大3回）
            _MAX_RETRIES = 3
            with st.chat_message("assistant"):
                for _attempt in range(1, _MAX_RETRIES + 1):
                    try:
                        from core.rag import stream_question
                        answer_parts = []
                        error_msg = None

                        # エビデンスと回答のプレースホルダを用意
                        evidence_placeholder = st.empty()
                        answer_placeholder = st.empty()

                        with st.spinner("🔍 文書を検索中..." if _attempt == 1 else f"🔄 再試行中... ({_attempt}/{_MAX_RETRIES})"):
                            stream = stream_question(
                                question=prompt,
                                model_name=selected_model,
                                similarity_threshold=new_sim,
                                top_k=TOP_K,
                                chat_history=history_for_llm,  # BUG-001修正
                            )
                            # 最初のイベントはエビデンス
                            first = next(stream, None)
                            if first and first["type"] == "evidence":
                                evidence_list = first["data"]

                        # エビデンスを即座に表示
                        with evidence_placeholder.container():
                            render_evidence(evidence_list, key_prefix="cur")

                        # トークンをストリーミング表示しながらバッファに蓄積
                        for event in stream:
                            if event["type"] == "token":
                                answer_parts.append(event["data"])
                                answer_placeholder.markdown("".join(answer_parts))
                            elif event["type"] == "error":
                                error_msg = event["data"]
                                break

                        full_answer = "".join(answer_parts)
                        if error_msg:
                            if _attempt < _MAX_RETRIES:
                                logger.warning(f"LLM回答生成エラー（試行{_attempt}/{_MAX_RETRIES}）: {error_msg[:200]}")
                                import time; time.sleep(1)
                                continue  # リトライ
                            answer_placeholder.error(
                                f"エラーが発生しました:\n\n```\n{error_msg[:400]}\n```"
                            )
                        break  # 成功またはエラー表示済み → ループ終了

                    except Exception as e:
                        import traceback as _tb
                        tb_str = _tb.format_exc()
                        logger.error(f"チャットエラー（試行{_attempt}/{_MAX_RETRIES}）:\n{tb_str}")
                        if _attempt < _MAX_RETRIES:
                            import time; time.sleep(1)
                            continue  # リトライ
                        error_msg = str(e)[:400]
                        st.error(f"エラーが発生しました: {error_msg}")

            # BUG-002修正: セッション状態に保存してから st.rerun() で再描画
            # → 履歴ループからの一貫したレンダリングに切り替え、消える問題を解消
            final_content = full_answer if full_answer else (
                f"エラーが発生しました:\n\n```\n{error_msg}\n```" if error_msg
                else "回答を生成できませんでした。"
            )
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": final_content,
                "evidence": evidence_list,
            })

            # 永続履歴に保存
            if full_answer:
                save_to_history(prompt, full_answer, evidence_list)

            # BUG-002修正: rerun で上部の履歴ループから一貫してレンダリング
            st.rerun()

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
    for idx, item in enumerate(reversed(filtered[-50:])):
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
                render_evidence(ev, key_prefix=f"hist_{idx}")

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
