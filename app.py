"""
モノシリ - 社内資料探索AI
メインアプリケーションエントリーポイント

「社内資料は、一切外に出ません。」
- 完全ローカルモード（Phase 1 MVP）
- Streamlit + ChromaDB + multilingual-e5-large + llama-cpp-python
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st

# プロジェクトルートをPythonパスに追加
_app_dir = Path(__file__).parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

# ─── ページ設定（最初に呼ぶ必要がある）─────────────────────────
st.set_page_config(
    page_title="モノシリ",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "モノシリ - 社内資料探索AI（完全ローカルモード）",
    },
)

# ─── セッション状態初期化 ─────────────────────────────────────
def _init_session_state() -> None:
    """セッション状態を初期化する（初回のみ実行）"""
    if "initialized" not in st.session_state:
        from core.config import load_config
        from core.llm import load_custom_models

        # 設定を読み込む
        config = load_config()

        # カスタムモデルを読み込む（llm.pyのCUSTOM_MODELS_FILEから）
        load_custom_models()

        st.session_state.config = config
        st.session_state.page = "chat"
        st.session_state.chat_messages = []
        st.session_state.initialized = True


_init_session_state()


# ─── モデルプリウォーム（初回ロードを起動時に前倒し）─────────────
def _warmup_model_if_needed() -> None:
    """
    選択済みモデルをバックグラウンドでOllamaにロードする。
    モデルがVRAMにロードされていない場合のみ実行する。
    SPEED修正: warmup_done フラグではなく /api/ps で実際のロード状態を確認し、
    アンロード後のコールドスタートを防ぐ。
    """
    selected_model = st.session_state.config.get("selected_model", "")
    if not selected_model:
        return

    import threading
    from core.llm import warmup_model, is_model_downloaded, is_model_loaded_in_ollama

    if not is_model_downloaded(selected_model):
        return

    # すでにVRAMにロード済みなら何もしない
    if is_model_loaded_in_ollama(selected_model):
        return

    def _run():
        try:
            warmup_model(selected_model)
        except Exception:
            pass  # プリウォームの失敗はサイレントに無視

    threading.Thread(target=_run, daemon=True).start()


_warmup_model_if_needed()

# ─── サイドバー ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 モノシリ")
    st.caption("社内資料は、一切外に出ません。")

    st.divider()

    # ナビゲーションボタン
    pages = [
        ("chat",     "💬 チャット"),
        ("index",    "📁 インデックス管理"),
        ("settings", "⚙️ 設定"),
    ]

    for page_id, page_label in pages:
        is_active = st.session_state.page == page_id
        if st.button(
            page_label,
            key=f"nav_{page_id}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            if st.session_state.page != page_id:
                st.session_state.page = page_id
                st.rerun()

    st.divider()

    # サイドバー下部: インデックス統計
    try:
        from core.indexer import get_index_stats
        stats = get_index_stats()
        if stats["total_chunks"] > 0:
            st.metric("インデックス済み", f"{stats['total_chunks']:,} チャンク")
    except Exception:
        pass

    # 登録フォルダ数
    folders = st.session_state.config.get("folders", [])
    if folders:
        st.caption(f"📂 {len(folders)}フォルダ登録済み")

    # モデル情報
    selected_model = st.session_state.config.get("selected_model", "")
    if selected_model:
        short = selected_model.split("（")[0]
        st.caption(f"🤖 {short}")
    else:
        st.caption("🤖 モデル未設定")

    st.divider()
    st.caption("Phase 1 — 完全ローカルモード")


# ─── ページルーティング ───────────────────────────────────────
page = st.session_state.get("page", "chat")

if page == "chat":
    from ui.chat_page import render_chat_page
    render_chat_page()

elif page == "index":
    from ui.index_page import render_index_page
    render_index_page()

elif page == "settings":
    from ui.settings_page import render_settings_page
    render_settings_page()

else:
    st.error(f"不明なページ: {page}")
    st.session_state.page = "chat"
    st.rerun()
# reload-trigger-4
