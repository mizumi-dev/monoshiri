"""
モノシリ 設定画面
AIモデル管理（ダウンロードキュー・進捗・切り替え・カスタム追加）とシステム情報を表示する。
"""
from __future__ import annotations
import logging
from pathlib import Path

import streamlit as st

from core.config import (
    load_config, save_config, LLM_MODELS, MODELS_DIR,
    GGUF_DIRECT_URLS, CONFIG_FILE,
)
import core.config as app_config

logger = logging.getLogger(__name__)


# ─── ダウンロードキューパネル（自動リフレッシュ） ─────────────

@st.fragment(run_every=1)
def _render_download_queue() -> None:
    """
    ダウンロードキューパネル。
    1秒ごとに自動リフレッシュして進捗バーをリアルタイム更新する。
    """
    from core.downloader import DownloadManager, JobStatus, STATUS_LABEL

    dm = DownloadManager.get()
    jobs = dm.get_jobs()

    if not jobs:
        return

    st.subheader("📥 ダウンロードキュー")

    active_count = sum(
        1 for j in jobs if j.status in (JobStatus.PENDING, JobStatus.DOWNLOADING)
    )
    if active_count > 0:
        st.caption(f"⬇️ {active_count}件処理中 / 合計{len(jobs)}件")
    else:
        st.caption(f"✅ 全{len(jobs)}件完了")
        if st.button("履歴を消去", key="clear_queue_history"):
            dm.clear_finished()
            st.rerun()

    for job in jobs:
        with st.container(border=True):
            col_name, col_status, col_action = st.columns([3, 2, 1])

            with col_name:
                short = job.model_name.split("（")[0]
                st.markdown(f"**{short}**")
                st.caption(job.model_name)

            with col_status:
                label = STATUS_LABEL.get(job.status, job.status)
                if job.status == JobStatus.DOWNLOADING:
                    st.markdown(f"**{label}**")
                elif job.status == JobStatus.DONE:
                    st.success(label, icon=None)
                elif job.status == JobStatus.FAILED:
                    st.error(label, icon=None)
                elif job.status == JobStatus.CANCELLED:
                    st.warning(label, icon=None)
                else:
                    pos = dm.queue_position(job.model_name)
                    pos_str = f"（{pos}番目）" if pos else ""
                    st.markdown(f"{label}{pos_str}")

            with col_action:
                if job.status == JobStatus.PENDING:
                    if st.button("🚫", key=f"cancel_{job.model_name}",
                                 help="キャンセル", use_container_width=True):
                        dm.cancel(job.model_name)
                        st.rerun()
                elif job.status in (JobStatus.FAILED, JobStatus.CANCELLED):
                    if st.button("🔄", key=f"retry_{job.model_name}",
                                 help="リトライ", use_container_width=True):
                        dm.retry(job.model_name)
                        st.rerun()

            # 進捗バー（ダウンロード中 or 完了時）
            if job.status == JobStatus.DOWNLOADING:
                st.progress(job.progress, text=job.progress_label())

            elif job.status == JobStatus.DONE:
                elapsed = job.elapsed_seconds()
                elapsed_str = (
                    f"{elapsed / 60:.1f}分" if elapsed >= 60
                    else f"{elapsed:.0f}秒"
                )
                st.progress(1.0, text=f"完了（{elapsed_str}）")

            elif job.status == JobStatus.FAILED:
                with st.expander("エラー詳細"):
                    st.code(job.error_msg[:1000], language=None)

    st.divider()


# ─── モデル管理 ───────────────────────────────────────────────

def _render_model_management(config: dict) -> None:
    """AIモデル管理セクションをレンダリングする"""
    from core.llm import (
        is_model_downloaded, get_total_ram_gb, get_available_ram_gb,
        recommend_model, get_model_path, load_custom_models,
    )
    from core.downloader import DownloadManager, JobStatus

    dm = DownloadManager.get()

    # カスタムモデルを読み込む
    load_custom_models()

    selected_model: str = config.get("selected_model", "")
    total_ram = get_total_ram_gb()
    available_ram = get_available_ram_gb()
    recommended = recommend_model()

    # ── ダウンロードキューパネル（自動リフレッシュ） ──
    _render_download_queue()

    # ── RAM情報 ──
    st.subheader("💻 PC情報")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("総RAM", f"{total_ram:.1f} GB")
    with col2:
        st.metric("空きRAM", f"{available_ram:.1f} GB")
    with col3:
        st.metric("推奨モデル", recommended.split("（")[0])

    st.divider()

    # ── 現在使用中のモデル ──
    st.subheader("🤖 使用中のモデル")
    if selected_model and is_model_downloaded(selected_model):
        info = LLM_MODELS.get(selected_model, {})
        st.success(f"● {selected_model}")
        c1, c2, c3 = st.columns(3)
        with c1:
            size = info.get("size_gb", "?")
            st.caption(f"モデルサイズ: {size}GB")
        with c2:
            ram_use = (
                round(info.get("size_gb", 0) * 1.25, 1)
                if isinstance(info.get("size_gb"), (int, float))
                else "?"
            )
            st.caption(f"推定RAM使用: {ram_use}GB")
        with c3:
            st.caption(f"速度: {info.get('speed', '不明')}")
    elif selected_model:
        st.warning(f"⚠️ 選択中のモデルがダウンロードされていません: {selected_model}")
    else:
        st.warning("モデルが選択されていません。下からダウンロードして選択してください。")

    st.divider()

    # ── モデル一覧 ──
    st.subheader("📦 インストール済み・利用可能なモデル")

    for model_name, model_info in LLM_MODELS.items():
        downloaded = is_model_downloaded(model_name)
        is_selected = selected_model == model_name
        job = dm.get_job(model_name)
        is_queued = job is not None and job.status in (
            JobStatus.PENDING, JobStatus.DOWNLOADING
        )

        with st.container(border=True):
            col_info, col_action = st.columns([3, 1])

            with col_info:
                if is_selected:
                    status_icon = "🟢"
                elif downloaded:
                    status_icon = "⚪"
                elif is_queued:
                    status_icon = "⬇️"
                else:
                    status_icon = "💤"

                st.markdown(f"**{status_icon} {model_name}**")

                desc = model_info.get("description", "")
                size = model_info.get("size_gb", "?")
                min_ram = model_info.get("min_ram_gb", "?")
                speed = model_info.get("speed", "?")
                if desc:
                    st.caption(desc)
                st.caption(
                    f"サイズ: {size}GB  |  必要RAM: {min_ram}GB以上  |  速度: {speed}"
                )
                if (
                    not downloaded
                    and isinstance(min_ram, (int, float))
                    and total_ram < min_ram
                ):
                    st.caption(
                        f"⚠️ RAM不足の可能性（現在{total_ram:.0f}GB / 推奨{min_ram}GB以上）"
                    )

            with col_action:
                if downloaded:
                    if not is_selected:
                        if st.button(
                            "選択",
                            key=f"select_{hash(model_name)}",
                            use_container_width=True,
                            type="primary",
                        ):
                            config["selected_model"] = model_name
                            save_config(config)
                            st.session_state.config = config
                            st.success(f"✅ {model_name} を選択しました")
                            st.rerun()
                    else:
                        st.button(
                            "使用中",
                            key=f"using_{hash(model_name)}",
                            disabled=True,
                            use_container_width=True,
                        )
                    if not is_selected:
                        if st.button(
                            "🗑️",
                            key=f"del_{hash(model_name)}",
                            use_container_width=True,
                            help="このモデルをアンインストール",
                        ):
                            path = get_model_path(model_name)
                            if path and path.exists():
                                path.unlink()
                                st.success(f"削除しました: {model_name}")
                                if selected_model == model_name:
                                    config["selected_model"] = ""
                                    save_config(config)
                                    st.session_state.config = config
                                st.rerun()

                elif is_queued:
                    st.button(
                        "待機中",
                        key=f"queued_{hash(model_name)}",
                        disabled=True,
                        use_container_width=True,
                    )

                else:
                    repo_id = model_info.get("repo_id")
                    if repo_id:
                        if st.button(
                            "＋ キューに追加",
                            key=f"dl_{hash(model_name)}",
                            use_container_width=True,
                            type="primary",
                            help=f"ダウンロードキューに追加（約{model_info.get('size_gb', '?')}GB）",
                        ):
                            added = dm.add(model_name)
                            if added:
                                st.success(f"✅ キューに追加しました: {model_name}")
                            else:
                                st.info("既にキューに入っています")
                            st.rerun()
                    else:
                        st.caption("（カスタム）")

    st.divider()

    # ── ダウンロード完了モデルを自動選択 ──
    _auto_select_downloaded(config, dm)

    # ── カスタムモデル追加 ──
    st.subheader("➕ モデルを追加")
    tab_hf, tab_local = st.tabs(["HuggingFaceからダウンロード", "ローカルファイルから追加"])
    with tab_hf:
        _render_hf_download(dm)
    with tab_local:
        _render_local_add()


def _auto_select_downloaded(config: dict, dm) -> None:
    """
    ダウンロード完了したモデルを自動的に選択状態にする。
    既にモデルが選択されている場合は上書きしない。
    """
    from core.downloader import JobStatus
    from core.llm import is_model_downloaded

    if config.get("selected_model"):
        return

    jobs = dm.get_jobs()
    for job in jobs:
        if job.status == JobStatus.DONE and is_model_downloaded(job.model_name):
            config["selected_model"] = job.model_name
            save_config(config)
            st.session_state.config = config
            st.success(f"✅ {job.model_name} を自動選択しました")
            break


def _render_hf_download(dm) -> None:
    """HuggingFaceダウンロードフォームを表示する"""
    st.caption(
        "HuggingFaceのリポジトリIDとGGUFファイル名を指定してキューに追加します。\n"
        "例: `mmnga/Llama-3-ELYZA-JP-8B-GGUF` / `Llama-3-ELYZA-JP-8B-q4_k_m.gguf`"
    )

    with st.form("hf_download_form"):
        repo_id = st.text_input(
            "リポジトリID",
            placeholder="例: mmnga/Llama-3-ELYZA-JP-8B-GGUF",
        )
        filename = st.text_input(
            "ファイル名（.gguf）",
            placeholder="例: Llama-3-ELYZA-JP-8B-q4_k_m.gguf",
        )
        display_name = st.text_input(
            "表示名（任意）",
            placeholder="例: ELYZA-JP-8B",
        )
        submitted = st.form_submit_button("＋ キューに追加", type="primary")

    if submitted:
        if not repo_id or not filename:
            st.warning("リポジトリIDとファイル名を入力してください")
        elif not filename.endswith(".gguf"):
            st.warning("ファイル名は .gguf で終わる必要があります")
        else:
            name = display_name or filename.replace(".gguf", "")
            key = f"{name}（カスタム）"

            # LLM_MODELSに一時登録
            if key not in LLM_MODELS:
                LLM_MODELS[key] = {
                    "repo_id": repo_id,
                    "filename": filename,
                    "size_gb": 0,
                    "min_ram_gb": 8,
                    "speed": "不明",
                    "description": f"HF: {repo_id}/{filename}",
                    "chat_template": "chatml",
                }
            added = dm.add(key)
            if added:
                st.success(f"✅ キューに追加しました: {key}")
                st.rerun()
            else:
                st.info("既にキューに入っています")


def _render_local_add() -> None:
    """ローカルGGUFファイル追加フォームを表示する"""
    from core.llm import add_local_gguf

    st.caption(
        "PC上のGGUFファイルを直接追加します。"
        "ファイルはモノシリのモデルフォルダにコピーされます。"
    )

    with st.form("local_add_form"):
        file_path_str = st.text_input(
            "GGUFファイルのパス",
            placeholder="例: C:\\Users\\田中\\Downloads\\mymodel.gguf",
        )
        display_name = st.text_input(
            "表示名（任意）",
            placeholder="例: MyCustomModel",
            key="local_display_name",
        )
        submitted = st.form_submit_button("➕ 追加", type="primary")

    if submitted:
        gguf_path = Path(file_path_str.strip()) if file_path_str.strip() else None
        if not gguf_path:
            st.warning("ファイルパスを入力してください")
        elif not gguf_path.exists():
            st.error(f"ファイルが見つかりません: {file_path_str}")
        elif gguf_path.suffix.lower() != ".gguf":
            st.error("GGUFファイル（.gguf）のみ対応しています")
        else:
            try:
                key = add_local_gguf(gguf_path, display_name or None)
                config = st.session_state.get("config", load_config())
                config["selected_model"] = key
                save_config(config)
                st.session_state.config = config
                st.success(f"✅ 追加しました: {key}")
                st.rerun()
            except Exception as e:
                st.error(f"追加エラー: {e}")


# ─── ネットワーク設定 ──────────────────────────────────────────

def _render_network_settings() -> None:
    """ネットワーク・ダウンロード設定タブを表示する"""
    from core.llm import _check_network_connectivity

    st.subheader("🌐 ネットワーク診断")

    if st.button("接続テスト実行", type="primary"):
        with st.spinner("HuggingFaceへの接続を確認中..."):
            ok, msg = _check_network_connectivity()
        if ok:
            st.success(msg)
        else:
            st.error("接続に失敗しました")
            st.code(msg, language=None)

    st.divider()

    # HFミラーURL設定
    st.subheader("🔗 HuggingFaceミラー設定")
    st.caption(
        "HuggingFace本体に接続できない場合、ミラーサイトを利用できます。\n"
        "空欄にすると通常のHuggingFace（https://huggingface.co）を使用します。"
    )

    current_mirror = app_config.HF_MIRROR_URL
    new_mirror = st.text_input(
        "ミラーURL",
        value=current_mirror,
        placeholder="例: https://hf-mirror.com",
        key="hf_mirror_input",
    )
    if st.button("ミラーURLを保存"):
        app_config.HF_MIRROR_URL = new_mirror.strip()
        config = st.session_state.get("config", load_config())
        config["hf_mirror_url"] = new_mirror.strip()
        save_config(config)
        st.success(f"保存しました: {new_mirror.strip() or '（デフォルト）'}")

    st.divider()

    # 手動ダウンロードガイド
    st.subheader("📥 手動ダウンロードガイド")
    st.caption("自動ダウンロードに失敗する場合、ブラウザで直接ダウンロードできます。")

    st.markdown("**LLMモデル（GGUF）の配置先フォルダ**")
    model_dir = MODELS_DIR / "llm"
    st.code(str(model_dir), language=None)

    st.markdown("**各モデルの直接ダウンロードURL**")
    for model_name, model_info in LLM_MODELS.items():
        repo_id = model_info.get("repo_id")
        filename = model_info["filename"]
        if repo_id:
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            st.markdown(f"**{model_name}**")
            st.code(url, language=None)

    st.divider()

    st.markdown("**Embeddingモデル**")
    from core.config import EMBEDDING_MODEL_DIR, EMBEDDING_MODEL_ID
    st.text(f"配置先フォルダ: {EMBEDDING_MODEL_DIR}")
    st.code(f"https://huggingface.co/{EMBEDDING_MODEL_ID}", language=None)
    st.caption("「Files and versions」タブから全ファイルをダウンロードし、上記フォルダに配置してください。")

    st.divider()

    st.subheader("🧠 Embeddingモデル状態")
    if EMBEDDING_MODEL_DIR.exists() and any(EMBEDDING_MODEL_DIR.iterdir()):
        st.success(f"ローカルにキャッシュ済み: {EMBEDDING_MODEL_DIR}")
    else:
        st.warning(
            "Embeddingモデルがまだダウンロードされていません。\n"
            "インデックス作成時に自動ダウンロードを試みます。"
        )


# ─── システム情報 ──────────────────────────────────────────────

def _render_system_info() -> None:
    """システム情報タブを表示する"""
    import platform
    import psutil
    from core.config import DATA_DIR, MODELS_DIR, CHROMA_DIR
    from core.indexer import get_index_stats

    st.subheader("💻 システム情報")

    col_sys, col_data = st.columns(2)

    with col_sys:
        st.markdown("**OS / ハードウェア**")
        st.text(f"OS: {platform.system()} {platform.release()}")
        st.text(f"CPU: {psutil.cpu_count(logical=True)}コア")
        mem = psutil.virtual_memory()
        st.text(f"RAM: {mem.total / (1024**3):.1f}GB（使用中: {mem.percent}%）")
        disk = psutil.disk_usage("/")
        st.text(f"ディスク空き: {disk.free / (1024**3):.1f}GB")

    with col_data:
        st.markdown("**データ保存先**")
        st.text(f"データ: {DATA_DIR}")
        st.text(f"モデル: {MODELS_DIR}")
        st.text(f"DB: {CHROMA_DIR}")

        st.markdown("**インデックス情報**")
        try:
            stats = get_index_stats()
            st.text(f"総チャンク数: {stats['total_chunks']:,}")
        except Exception:
            st.text("取得できません")

        try:
            total_size = sum(
                f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file()
            )
            st.text(f"ディスク使用量: {total_size / (1024**2):.1f}MB")
        except Exception:
            st.text("計算できません")

    st.divider()

    st.subheader("⚠️ データリセット")
    st.warning(
        "インデックスをすべて削除します。\n"
        "フォルダの登録・設定は削除されません。"
    )
    if st.button("🗑️ インデックスを全削除（再構築）", type="secondary"):
        try:
            import shutil
            from core.config import CHROMA_DIR, HASH_DIR
            shutil.rmtree(str(CHROMA_DIR), ignore_errors=True)
            shutil.rmtree(str(HASH_DIR), ignore_errors=True)
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            HASH_DIR.mkdir(parents=True, exist_ok=True)
            st.success("✅ インデックスを全削除しました。次回インデックス作成時に再構築されます。")
            st.rerun()
        except Exception as e:
            st.error(f"削除エラー: {e}")


# ─── メイン ───────────────────────────────────────────────────

def render_settings_page() -> None:
    """設定画面をレンダリングする"""
    st.header("⚙️ 設定")

    config = st.session_state.get("config", load_config())

    # 保存済みミラーURLをモジュール変数に反映
    saved_mirror = config.get("hf_mirror_url", "")
    if saved_mirror and not app_config.HF_MIRROR_URL:
        app_config.HF_MIRROR_URL = saved_mirror

    tab_model, tab_network, tab_system = st.tabs([
        "🤖 AIモデル管理",
        "🌐 ネットワーク・ダウンロード",
        "ℹ️ システム情報",
    ])

    with tab_model:
        _render_model_management(config)

    with tab_network:
        _render_network_settings()

    with tab_system:
        _render_system_info()
