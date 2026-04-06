"""
モノシリ インデックス管理画面
フォルダ追加・インデックス作成・進捗表示・スキップログを提供する。
SHA-256差分検知により変更ファイルのみ再処理する。
"""
from __future__ import annotations
import time
import uuid
import json
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
                # インデックスからも削除
                try:
                    from core.indexer import delete_folder_from_index
                    delete_folder_from_index(folder_path)
                    st.success(f"削除しました: {folder_str}")
                except Exception as e:
                    logger.warning(f"インデックス削除エラー: {e}")
                st.rerun()


# ─── インデックス作成 ──────────────────────────────────────────

def _run_indexing(config: dict) -> None:
    """
    インデックス作成を実行する。
    Streamlitの進捗表示（progress bar + テキスト）でリアルタイム更新する。
    """
    from core.indexer import (
        get_collection, scan_and_diff, delete_from_chroma,
        process_single_file, flush_batch, load_skip_log, save_skip_log,
        BATCH_SIZE,
    )

    folders = [Path(f) for f in config.get("folders", []) if Path(f).exists()]

    if not folders:
        st.error("有効なフォルダがありません。フォルダを追加してください。")
        st.session_state.indexing = False
        return

    # 差分スキャン
    with st.spinner("📂 フォルダをスキャン中..."):
        files_to_process, deleted_paths = scan_and_diff(folders)

    # 削除されたファイルをChromaDBから除去
    if deleted_paths:
        with st.spinner(f"🗑️ 削除されたファイル（{len(deleted_paths)}件）を処理中..."):
            delete_from_chroma(deleted_paths)

    total = len(files_to_process)

    if total == 0:
        st.success("✅ インデックスは最新の状態です（変更なし）")
        st.session_state.indexing = False
        return

    # 時間見積もり表示
    estimated = estimate_index_time(total)
    st.info(f"📝 {total:,}件のファイルを処理します（目安: {estimated}）")

    # 進捗UI
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    time_text = st.empty()
    cancel_placeholder = st.empty()

    cancel_flag: list[bool] = [False]

    with cancel_placeholder:
        if st.button("⏹️ キャンセル", key="cancel_indexing"):
            cancel_flag[0] = True

    # スキップログの準備
    skip_log = load_skip_log()
    processing_paths = {str(f) for f, _ in files_to_process}
    skip_log = [s for s in skip_log if s["file_path"] not in processing_paths]

    # ChromaDBコレクション取得
    collection = get_collection()

    # 変更ファイルのChromaDBエントリを削除（再インデックス前）
    with st.spinner("🔄 変更ファイルの旧データを削除中..."):
        delete_from_chroma([str(f) for f, _ in files_to_process])

    # バッチバッファ
    batch_texts: list[str] = []
    batch_ids: list[str] = []
    batch_metadatas: list[dict] = []

    skipped = 0
    processed = 0
    start_time = time.time()

    for i, (file_path, folder_id) in enumerate(files_to_process):
        # キャンセルチェック
        if cancel_flag[0]:
            status_text.warning("⏹️ キャンセルされました")
            break

        # 進捗表示
        progress = (i + 1) / total
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"🔄 [{i+1}/{total}]  {file_path.name}")

        elapsed = time.time() - start_time
        if i > 0:
            per_file = elapsed / i
            remaining = per_file * (total - i)
            rem_str = f"  |  残り約{int(remaining)}秒" if remaining > 5 else ""
        else:
            rem_str = ""
        time_text.caption(f"経過: {int(elapsed)}秒{rem_str}")

        # ファイル処理
        chunks_added, skip_reason = process_single_file(
            file_path=file_path,
            folder_id=folder_id,
            collection=collection,
            batch_texts=batch_texts,
            batch_ids=batch_ids,
            batch_metadatas=batch_metadatas,
            skip_log=skip_log,
        )

        if skip_reason:
            skipped += 1
        else:
            processed += 1

        # バッチが溜まったらフラッシュ
        if len(batch_texts) >= BATCH_SIZE:
            try:
                flush_batch(collection, batch_texts, batch_ids, batch_metadatas)
            except Exception as e:
                logger.error(f"バッチ格納エラー: {e}")
                st.error(f"Embeddingエラーが発生しました: {e}")
                st.session_state.indexing = False
                return
            batch_texts.clear()
            batch_ids.clear()
            batch_metadatas.clear()

    # 残りのバッチをフラッシュ
    if batch_texts:
        try:
            flush_batch(collection, batch_texts, batch_ids, batch_metadatas)
        except Exception as e:
            logger.error(f"最終バッチ格納エラー: {e}")

    # スキップログ保存
    save_skip_log(skip_log)

    # 完了表示
    total_time = int(time.time() - start_time)
    progress_bar.progress(1.0)

    if not cancel_flag[0]:
        status_text.success(
            f"✅ 完了！  処理: {processed}件  スキップ: {skipped}件  "
            f"（所要時間: {total_time}秒）"
        )
    time_text.empty()
    cancel_placeholder.empty()

    st.session_state.indexing = False
    time.sleep(1.5)
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

    # テーブル形式で表示
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

    # フォルダ管理セクション
    _render_folder_management(config)

    st.divider()

    # インデックス作成ボタン
    folders = config.get("folders", [])

    if folders:
        # インデックス統計
        try:
            from core.indexer import get_index_stats
            stats = get_index_stats()
            if stats["total_chunks"] > 0:
                st.metric("現在のインデックス", f"{stats['total_chunks']:,} チャンク")
        except Exception:
            pass

        col_btn, col_info = st.columns([2, 3])

        with col_btn:
            if not st.session_state.get("indexing", False):
                if st.button(
                    "🔄 インデックスを作成・更新",
                    type="primary",
                    use_container_width=True,
                    help="変更されたファイルのみ差分で更新します（SHA-256検知）",
                ):
                    st.session_state.indexing = True
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

    # インデックス実行
    if st.session_state.get("indexing", False):
        st.divider()
        _run_indexing(config)

    st.divider()

    # スキップログ
    _render_skip_log()
