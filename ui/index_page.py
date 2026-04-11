"""
モノシリ インデックス管理画面
フォルダ追加・インデックス作成・進捗表示・スキップログを提供する。

設計方針:
- インデックス処理は IndexManager (core/indexer.py) のバックグラウンドスレッドで実行
- @st.fragment(run_every=1) で進捗を1秒ごとに自動更新
- キャンセルボタンは fragment 内に配置するため、処理中でも確実に反応する
"""
from __future__ import annotations
import logging
from pathlib import Path

import streamlit as st

from core.config import load_config, save_config, SKIP_LOG_FILE
from core.extractor import scan_folder, estimate_index_time


# PERF修正: scan_folder は再帰スキャンのため毎 rerun の呼び出しをキャッシュ。
# フォルダのファイル数はほとんど変わらないので 30 秒 TTL で十分。
@st.cache_data(ttl=30, show_spinner=False)
def _cached_scan_folder(folder_str: str) -> int:
    """フォルダのファイル数をキャッシュ付きで返す"""
    p = __import__("pathlib").Path(folder_str)
    if not p.exists():
        return 0
    try:
        return len(scan_folder(p))
    except Exception:
        return 0


@st.cache_data(ttl=60, show_spinner=False)
def _cached_index_stats() -> dict:
    """インデックス統計をキャッシュ付きで返す"""
    from core.indexer import get_index_stats
    return get_index_stats()

logger = logging.getLogger(__name__)


# ─── フォルダ管理 ─────────────────────────────────────────────

def _render_folder_management(config: dict) -> None:
    """フォルダ追加・削除UIを表示する"""
    st.subheader("📂 対象フォルダ")

    folders: list[str] = config.get("folders", [])

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
                    count = _cached_scan_folder(folder_str)
                    st.caption(f"{count:,}件のファイル")
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


# ─── インデックス進捗フラグメント ────────────────────────────────
#
# @st.fragment(run_every=1) により1秒ごとに自動更新。
# キャンセルボタンはこのフラグメント内にあるため、
# バックグラウンドスレッドをブロックすることなく確実に反応する。

@st.fragment(run_every=1)
def _render_indexing_progress() -> None:
    """インデックス進捗を自動更新表示する（フラグメント）"""
    from core.indexer import IndexManager

    im = IndexManager.get()

    # アイドル状態なら何も表示しない
    if im.status == "idle":
        return

    with st.container(border=True):

        # ── 実行中 ──────────────────────────────────────────
        if im.is_running:
            col_label, col_cancel = st.columns([3, 1])
            with col_label:
                st.markdown(f"**{im.phase_label}**")
            with col_cancel:
                if st.button("⏹️ キャンセル", key="cancel_idx", use_container_width=True):
                    im.cancel()

            # 進捗バー
            if im.status == "indexing" and im.total_count > 0:
                st.progress(im.progress)
                done = im.done_count
                total = im.total_count
                elapsed_str = im.get_elapsed_str()

                status_line = f"[{done}/{total}]"
                if im.current_file:
                    status_line += f"  {im.current_file}"
                st.caption(status_line)
                st.caption(f"経過: {elapsed_str}")

                # 残り時間の推定
                if done > 0 and im.elapsed > 0:
                    per_file = im.elapsed / done
                    remaining = per_file * max(total - done, 0)
                    if remaining > 5:
                        rem = f"{int(remaining // 60)}分{int(remaining % 60)}秒" if remaining >= 60 else f"{int(remaining)}秒"
                        st.caption(f"残り約 {rem}")
            else:
                st.caption(im.get_elapsed_str() + " 経過")

        # ── 完了 ────────────────────────────────────────────
        elif im.status == "done":
            result = im.result
            elapsed = im.get_elapsed_str()

            if result.get("no_change"):
                st.success("✅ インデックスは最新の状態です（変更なし）")
            else:
                processed = result.get("processed", 0)
                skipped = result.get("skipped", 0)
                st.success(
                    f"✅ 完了！  処理: {processed}件  スキップ: {skipped}件  "
                    f"（所要時間: {elapsed}）"
                )

            if st.button("閉じる", key="close_idx_done"):
                im.reset()
                st.rerun()

        # ── キャンセル ────────────────────────────────────────
        elif im.status == "cancelled":
            elapsed = im.get_elapsed_str()
            st.warning(f"⏹️ キャンセルしました（処理済みのデータは保持されます）（{elapsed}）")

            if st.button("閉じる", key="close_idx_cancelled"):
                im.reset()
                st.rerun()

        # ── エラー ────────────────────────────────────────────
        elif im.status == "error":
            st.error(f"❌ エラーが発生しました")
            with st.expander("エラー詳細"):
                st.code(im.error_msg[:2000], language=None)

            if st.button("閉じる", key="close_idx_error"):
                im.reset()
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

    for item in skip_log[:50]:
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
    from core.indexer import IndexManager

    st.header("📁 インデックス管理")

    config = st.session_state.get("config", load_config())
    im = IndexManager.get()

    # フォルダ管理セクション
    _render_folder_management(config)

    st.divider()

    folders = config.get("folders", [])

    if folders:
        # インデックス統計（キャッシュ付き）
        try:
            stats = _cached_index_stats()
            if stats["total_chunks"] > 0:
                st.metric("現在のインデックス", f"{stats['total_chunks']:,} チャンク")
        except Exception:
            pass

        col_btn, col_info = st.columns([2, 3])

        with col_btn:
            if im.is_running:
                st.button(
                    "⏳ インデックス作成中...",
                    disabled=True,
                    use_container_width=True,
                )
            else:
                # ファイル数を取得して目安時間を表示
                try:
                    total_files = sum(
                        _cached_scan_folder(f)
                        for f in folders if Path(f).exists()
                    )
                    btn_help = f"変更ファイルのみ差分更新（SHA-256検知）"
                except Exception:
                    btn_help = "変更ファイルのみ差分で更新します"

                if st.button(
                    "🔄 インデックスを作成・更新",
                    type="primary",
                    use_container_width=True,
                    help=btn_help,
                ):
                    folder_paths = [Path(f) for f in folders if Path(f).exists()]
                    if folder_paths:
                        im.start(folder_paths)
                        st.rerun()
                    else:
                        st.error("有効なフォルダがありません")

                # 全件再インデックスボタン（抽出ロジック変更後などに使用）
                if st.button(
                    "⚡ 全件再インデックス（強制）",
                    use_container_width=True,
                    help="ハッシュキャッシュをクリアして全ファイルを再処理します。抽出ロジック変更後に使用してください。",
                ):
                    folder_paths = [Path(f) for f in folders if Path(f).exists()]
                    if folder_paths:
                        im.start_force_reindex(folder_paths)
                        st.rerun()
                    else:
                        st.error("有効なフォルダがありません")

        with col_info:
            st.caption(
                "💡 差分インデックス機能により、変更されたファイルのみ再処理します。\n"
                "初回は時間がかかりますが、2回目以降は高速です。"
            )

    # インデックス進捗（フラグメント：1秒ごと自動更新、キャンセル可能）
    # ※ 無条件でフラグメントを追加する。条件付きにするとst.rerun()のたびに
    #   fragmentが削除→再追加されrun_every=1のタイマーがリセットされるため。
    #   fragment内部でidle時は何も描画しないことで同等の制御を行う。
    st.divider()
    _render_indexing_progress()

    st.divider()

    # スキップログ
    _render_skip_log()
