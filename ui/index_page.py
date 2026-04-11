"""
モノシリ インデックス管理画面
フォルダ追加・インデックス作成・進捗表示・スキップログを提供する。
SHA-256差分検知により変更ファイルのみ再処理する。

疎結合設計:
  - UI は IndexManager のみを通じてインデックス処理を制御する
  - IndexManager はバックグラウンドスレッドで実行し、UIスレッドをブロックしない
  - ファイル処理の詳細ロジックはすべて indexer.py に閉じている
"""
from __future__ import annotations
import logging
from pathlib import Path

import streamlit as st

from core.config import load_config, save_config, SKIP_LOG_FILE
from core.extractor import scan_folder, estimate_index_time

logger = logging.getLogger(__name__)


# ─── フォルダ管理 ─────────────────────────────────────────────

def _render_folder_management(config: dict) -> None:
    """フォルダ追加・削除UIを表示する"""
    st.subheader("📂 対象フォルダ")

    folders: list[str] = config.get("folders", [])

    # フォルダ追加フォーム
    with st.form("add_folder_form", clear_on_submit=True):
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            new_folder = st.text_input(
                "フォルダパスを追加",
                placeholder="例: C:\\Users\\田中\\Documents\\社内資料",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("➕ 追加", use_container_width=True)

    if submitted:
        folder_path = Path(new_folder.strip()) if new_folder.strip() else None
        if not folder_path:
            st.warning("フォルダパスを入力してください")
        elif not folder_path.exists():
            st.error(f"フォルダが見つかりません: {new_folder}")
        elif str(folder_path) in folders:
            st.warning("このフォルダはすでに登録済みです")
        elif not folder_path.is_dir():
            st.error("フォルダではなくファイルが指定されています")
        else:
            folders.append(str(folder_path.resolve()))
            config["folders"] = folders
            save_config(config)
            st.session_state.config = config
            st.success(f"✅ 追加しました: {folder_path.name}")
            st.rerun()

    # 登録済みフォルダ一覧
    if not folders:
        st.info("フォルダが登録されていません。上のフォームからフォルダを追加してください。")
        return

    for i, folder_str in enumerate(folders):
        folder_path = Path(folder_str)
        exists = folder_path.exists()

        col_name, col_count, col_del = st.columns([4, 2, 1])

        with col_name:
            icon = "📂" if exists else "⚠️"
            st.text(f"{icon} {folder_str}")
            if not exists:
                st.caption("⚠️ フォルダが見つかりません（移動または削除された可能性があります）")

        with col_count:
            if exists:
                try:
                    files = scan_folder(folder_path)
                    st.caption(f"{len(files):,}件のファイル")
                except Exception:
                    st.caption("ファイル数を取得できません")

        with col_del:
            if st.button("削除", key=f"del_folder_{i}", use_container_width=True):
                folders.pop(i)
                config["folders"] = folders
                save_config(config)
                st.session_state.config = config
                try:
                    from core.indexer import delete_folder_from_index
                    delete_folder_from_index(folder_path)
                    st.success(f"削除しました: {folder_str}")
                except Exception as e:
                    logger.warning(f"インデックス削除エラー: {e}")
                st.rerun()


# ─── インデックス進捗表示（自動リフレッシュ） ─────────────────

@st.fragment(run_every=1)
def _render_indexing_progress() -> None:
    """
    IndexManagerの進捗をリアルタイム表示する。
    1秒ごとに自動リフレッシュして進捗バーを更新する。
    settings_page.py の _render_download_queue と同じパターン。
    """
    from core.indexer import IndexManager
    mgr = IndexManager.get()

    if mgr.status == "idle":
        return

    # フェーズラベル表示
    st.markdown(f"**{mgr.phase_label}**")

    # 進捗バー（ファイル処理フェーズのみ）
    if mgr.total_count > 0:
        st.progress(mgr.progress)
        st.caption(
            f"[{mgr.done_count}/{mgr.total_count}]  {mgr.current_file}"
            f"  |  経過: {mgr.get_elapsed_str()}"
        )
    elif mgr.status in ("scanning", "deleting"):
        st.caption(f"経過: {mgr.get_elapsed_str()}")

    # キャンセルボタン（処理中のみ）
    if mgr.is_running:
        if st.button("⏹️ キャンセル", key="cancel_indexing_btn"):
            mgr.cancel()

    # 完了・エラー時の結果表示
    if mgr.is_finished:
        if mgr.status == "done":
            result = mgr.result
            if result.get("no_change"):
                st.success("✅ インデックスは最新の状態です（変更なし）")
            else:
                st.success(
                    f"✅ 完了！  処理: {result.get('processed', 0)}件  "
                    f"スキップ: {result.get('skipped', 0)}件  "
                    f"（所要時間: {mgr.get_elapsed_str()}）"
                )
        elif mgr.status == "cancelled":
            st.warning("⏹️ キャンセルされました")
        elif mgr.status == "error":
            st.error(f"❌ エラー: {mgr.error_msg}")

        if st.button("閉じる", key="close_indexing_result"):
            mgr.reset()
            st.rerun()


# ─── スキップログ表示 ──────────────────────────────────────────

def _render_skip_log() -> None:
    """スキップされたファイルのログを表示する"""
    st.subheader("📋 スキップログ")

    from core.indexer import load_skip_log
    skip_log = load_skip_log()

    if not skip_log:
        st.info("スキップされたファイルはありません")
        return

    st.warning(f"{len(skip_log)}件のファイルが処理できませんでした")

    for item in skip_log[:50]:  # 最大50件
        with st.container():
            col_file, col_reason = st.columns([3, 2])
            with col_file:
                st.text(f"⚠️ {item['file_name']}")
                st.caption(item.get("file_path", ""))
            with col_reason:
                st.caption(item.get("reason", "不明"))

    if len(skip_log) > 50:
        st.caption(f"...他{len(skip_log) - 50}件（ログファイルを参照: {SKIP_LOG_FILE}）")


# ─── メイン ───────────────────────────────────────────────────

def render_index_page() -> None:
    """インデックス管理画面をレンダリングする"""
    st.header("📁 インデックス管理")

    config = st.session_state.get("config", load_config())
    from core.indexer import IndexManager
    from core.chromadb_store import get_index_stats
    mgr = IndexManager.get()

    # フォルダ管理セクション
    _render_folder_management(config)

    st.divider()

    folders = config.get("folders", [])

    if folders:
        # インデックス統計
        try:
            stats = get_index_stats()
            if stats["total_chunks"] > 0:
                st.metric("現在のインデックス", f"{stats['total_chunks']:,} チャンク")
        except Exception:
            pass

        col_btn, col_info = st.columns([2, 3])

        with col_btn:
            if not mgr.is_running:
                if st.button(
                    "🔄 インデックスを作成・更新",
                    type="primary",
                    use_container_width=True,
                    help="変更されたファイルのみ差分で更新します（SHA-256検知）",
                ):
                    folder_paths = [Path(f) for f in folders if Path(f).exists()]
                    if folder_paths:
                        mgr.reset()
                        mgr.start(folder_paths)
                    else:
                        st.error("有効なフォルダがありません")
                    st.rerun()
            else:
                st.button(
                    "⏳ インデックス作成中...",
                    disabled=True,
                    use_container_width=True,
                )

        with col_info:
            st.caption(
                "💡 差分インデックス機能により、変更されたファイルのみ再処理します。\n"
                "初回は時間がかかりますが、2回目以降は高速です。"
            )

    # インデックス進捗（処理中または完了直後のみ表示）
    if mgr.status != "idle":
        st.divider()
        _render_indexing_progress()

    st.divider()

    # スキップログ
    _render_skip_log()
