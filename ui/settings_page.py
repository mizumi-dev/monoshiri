"""
モノシリ 設定画面
3タブ構成（旧4タブから統合）:
  1. 🤖 AIモデル   - 回答AI(LLM) + 検索AI(Embedding) を一元管理
  2. 🌐 ネットワーク - HF接続確認・ミラー設定
  3. ℹ️ システム情報 - PCスペック・データ管理

「回答AI」と「検索AI」はどちらもAIモデルであることを明示し、
一般ユーザーが違いを直感的に理解できるようUIを設計している。
"""
from __future__ import annotations
import logging
from pathlib import Path

import streamlit as st

from core.config import (
    load_config, save_config, LLM_MODELS, MODELS_DIR,
    GGUF_DIRECT_URLS,
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


# ─── タブ１: AIモデル（回答AI + 検索AI 統合） ────────────────────

def _render_ai_models_tab(config: dict) -> None:
    """AIモデル管理タブ（回答AI＋検索AIを一元表示）"""
    from core.llm import (
        is_model_downloaded, get_total_ram_gb, get_available_ram_gb,
        recommend_model, get_model_path, load_custom_models,
    )
    from core.downloader import DownloadManager, JobStatus
    from core.config import EMBEDDING_MODEL_DIR, EMBEDDING_MODEL_ID

    dm = DownloadManager.get()
    load_custom_models()

    selected_model: str = config.get("selected_model", "")

    # ─── モノシリの仕組み説明 ──────────────────────────────────
    st.info(
        "🤖 **モノシリは2種類のAIを組み合わせて動作しています**\n\n"
        "どちらも「AI（人工知能）モデル」ですが、役割が異なります。\n\n"
        "**💬 回答AI（LLM）** — あなたの質問文を読んで、社内資料に基づいた"
        "回答文を生成するAIです。複数のモデルから選択できます。\n\n"
        "**🔍 検索AI（Embedding）** — 社内資料を「AIが理解できる数値」に変換し、"
        "質問と関連する文書を見つけやすくするAIです。自動的に管理されます。"
    )

    # ─── 現在のモデル状況サマリー ─────────────────────────────────
    embedding_ready = EMBEDDING_MODEL_DIR.exists() and any(EMBEDDING_MODEL_DIR.iterdir())

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**💬 回答AI（LLM）の状態**")
            if selected_model and is_model_downloaded(selected_model):
                short = selected_model.split("（")[0]
                st.success(f"🟢 使用中: {short}")
            elif selected_model:
                st.warning(f"⚠️ 未DL: {selected_model.split('（')[0]}")
            else:
                # DL完了しているモデルを自動選択
                installed = [n for n in LLM_MODELS if is_model_downloaded(n)]
                if installed:
                    st.warning(f"未選択（{len(installed)}件インストール済み）")
                else:
                    st.warning("モデルがありません")
    with col2:
        with st.container(border=True):
            st.markdown("**🔍 検索AI（Embedding）の状態**")
            if embedding_ready:
                st.success("✅ インストール済み")
                try:
                    size_mb = sum(
                        f.stat().st_size for f in EMBEDDING_MODEL_DIR.rglob("*") if f.is_file()
                    ) / (1024 ** 2)
                    st.caption(f"multilingual-e5-large（{size_mb:.0f}MB）")
                except Exception:
                    st.caption("multilingual-e5-large")
            else:
                st.warning("⚠️ 未インストール")
                st.caption("インデックス作成時に自動ダウンロードします")

    # ─── ダウンロードキューパネル ──────────────────────────────────
    _render_download_queue()

    # ─── RAM情報 ─────────────────────────────────────────────────
    total_ram = get_total_ram_gb()
    available_ram = get_available_ram_gb()
    recommended = recommend_model()

    st.subheader("💻 あなたのPCのメモリ")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("総RAM", f"{total_ram:.1f} GB")
    with col2:
        st.metric("空きRAM", f"{available_ram:.1f} GB")
    with col3:
        current_model = config.get("selected_model", "")
        current_label = current_model.split("（")[0] if current_model else "未設定"
        st.metric("使用中モデル", current_label)

    st.divider()

    # ─── 💬 回答AIモデル一覧（インストール済みのみ） ──────────────
    st.subheader("💬 回答AIモデル一覧")
    st.caption(
        "インストール済みのモデルを選択して使用できます。\n"
        "🟢 使用中　　✅ インストール済み（未選択）"
    )

    installed_any = False
    for model_name, model_info in LLM_MODELS.items():
        downloaded = is_model_downloaded(model_name)
        if not downloaded:
            continue  # インストール済みのみ表示
        installed_any = True
        is_selected = selected_model == model_name

        status_icon = "🟢" if is_selected else "✅"
        status_desc = "使用中" if is_selected else "インストール済み"

        with st.container(border=True):
            col_info, col_action = st.columns([3, 1])

            with col_info:
                st.markdown(f"**{status_icon} {model_name}**")
                desc = model_info.get("description", "")
                size = model_info.get("size_gb", "?")
                min_ram = model_info.get("min_ram_gb", "?")
                speed = model_info.get("speed", "?")
                if desc:
                    st.caption(desc)
                st.caption(f"サイズ: {size}GB  |  必要RAM: {min_ram}GB以上  |  速度: {speed}  |  状態: {status_desc}")

            with col_action:
                if not is_selected:
                    if st.button("選択", key=f"select_{hash(model_name)}",
                                 use_container_width=True, type="primary"):
                        config["selected_model"] = model_name
                        save_config(config)
                        st.session_state.config = config
                        st.success(f"✅ {model_name} を選択しました")
                        st.rerun()
                else:
                    st.button("使用中", key=f"using_{hash(model_name)}",
                              disabled=True, use_container_width=True)
                if not is_selected:
                    if st.button("🗑️", key=f"del_{hash(model_name)}",
                                 use_container_width=True, help="アンインストール"):
                        path = get_model_path(model_name)
                        if path and path.exists():
                            path.unlink()
                            st.success(f"削除しました: {model_name}")
                            if selected_model == model_name:
                                config["selected_model"] = ""
                                save_config(config)
                                st.session_state.config = config
                            st.rerun()

    # DLキュー待機中のモデルも一覧に表示
    for model_name, model_info in LLM_MODELS.items():
        if is_model_downloaded(model_name):
            continue
        job = dm.get_job(model_name)
        if job is not None and job.status in (JobStatus.PENDING, JobStatus.DOWNLOADING):
            installed_any = True
            with st.container(border=True):
                col_info, col_action = st.columns([3, 1])
                with col_info:
                    st.markdown(f"**⬇️ {model_name}**")
                    desc = model_info.get("description", "")
                    if desc:
                        st.caption(desc)
                    st.caption("DL中 / 待機中")
                with col_action:
                    st.button("待機中", key=f"queued_{hash(model_name)}",
                              disabled=True, use_container_width=True)

    if not installed_any:
        st.info("📥 インストール済みの回答AIモデルはありません。下の「回答AIモデルを追加」からダウンロードしてください。")

    # DL完了モデルを自動選択
    _auto_select_downloaded(config, dm)

    st.divider()

    # ─── ➕ 回答AIモデルを追加（3タブ） ────────────────────────────
    st.subheader("➕ 回答AIモデルを追加")
    tab_rec, tab_hf, tab_local = st.tabs(["⭐ おすすめモデル", "HuggingFaceからDL", "ローカルファイルから追加"])
    with tab_rec:
        _render_recommended_llm_models(dm, total_ram, recommended)
    with tab_hf:
        _render_hf_download(dm)
    with tab_local:
        _render_local_add()

    st.divider()

    # ─── 🔍 検索AIモデル一覧（インストール済みのみ） ──────────────
    st.subheader("🔍 検索AIモデル一覧")
    st.caption("インストール済みの検索AIモデルです。")

    if embedding_ready:
        with st.container(border=True):
            col_emb_info, col_emb_status = st.columns([3, 1])
            with col_emb_info:
                st.markdown("**✅ multilingual-e5-large**")
                st.caption("日本語・英語・ドイツ語など多言語で高精度な検索が可能なモデルです。")
                try:
                    size_mb = sum(
                        f.stat().st_size for f in EMBEDDING_MODEL_DIR.rglob("*") if f.is_file()
                    ) / (1024 ** 2)
                    st.caption(f"モデルID: {EMBEDDING_MODEL_ID}  |  ディスク: {size_mb:.0f}MB")
                except Exception:
                    st.caption(f"モデルID: {EMBEDDING_MODEL_ID}")
            with col_emb_status:
                st.success("使用中")
    else:
        st.info("📥 インストール済みの検索AIモデルはありません。下の「検索AIモデルを追加」からインストールしてください。")

    st.divider()

    # ─── ⚡ GPU設定 ────────────────────────────────────────────────
    _render_gpu_settings(config)

    st.divider()

    # ─── 🔍 検索AIモデルを追加（3タブ） ────────────────────────────
    st.subheader("🔍 検索AIモデルを追加")
    tab_emb_rec, tab_emb_hf, tab_emb_manual = st.tabs(
        ["⭐ おすすめモデル", "HuggingFaceからDL", "手動インストール"]
    )
    with tab_emb_rec:
        _render_embedding_recommended(embedding_ready)
    with tab_emb_hf:
        _render_embedding_hf_download()
    with tab_emb_manual:
        _render_embedding_manual_install()


def _auto_select_downloaded(config: dict, dm) -> None:
    """DL完了したモデルを自動選択する。既に選択済みなら何もしない。"""
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


def _render_recommended_llm_models(dm, total_ram: float, recommended: str) -> None:
    """おすすめ回答AIモデル一覧（未ダウンロードのプリセットモデルを表示）"""
    from core.llm import is_model_downloaded
    from core.downloader import JobStatus

    st.caption(
        "あなたのPCのRAMに合わせたおすすめモデルを1クリックでダウンロードできます。\n"
        "⭐ マークは現在のPCに最適な推奨モデルです。"
    )

    not_downloaded = [
        (n, i) for n, i in LLM_MODELS.items()
        if not is_model_downloaded(n)
    ]

    if not not_downloaded:
        st.success("✅ 全ての推奨モデルがインストール済みです")
        return

    for model_name, model_info in not_downloaded:
        job = dm.get_job(model_name)
        is_queued = job is not None and job.status in (JobStatus.PENDING, JobStatus.DOWNLOADING)
        is_recommended = (model_name == recommended)

        with st.container(border=True):
            col_info, col_action = st.columns([3, 1])
            with col_info:
                badge = "⭐ " if is_recommended else ""
                st.markdown(f"**{badge}{model_name}**")
                desc = model_info.get("description", "")
                size = model_info.get("size_gb", "?")
                min_ram = model_info.get("min_ram_gb", "?")
                speed = model_info.get("speed", "?")
                if desc:
                    st.caption(desc)
                st.caption(f"サイズ: {size}GB  |  必要RAM: {min_ram}GB以上  |  速度: {speed}")
                if isinstance(min_ram, (int, float)) and total_ram < min_ram:
                    st.caption(f"⚠️ RAM不足の可能性（現在{total_ram:.0f}GB / 推奨{min_ram}GB以上）")
            with col_action:
                if is_queued:
                    st.button("待機中", key=f"rec_queued_{hash(model_name)}",
                              disabled=True, use_container_width=True)
                else:
                    repo_id = model_info.get("repo_id")
                    if repo_id:
                        if st.button(
                            "＋ DLキュー",
                            key=f"rec_dl_{hash(model_name)}",
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


def _render_gpu_settings(config: dict) -> None:
    """GPU設定セクション（Ollamaバックエンド版）"""
    from core.llm import (
        check_ollama_running,
        is_registered_with_ollama, register_model_with_ollama,
        is_model_downloaded,
    )

    st.subheader("⚡ GPU設定")

    # ─── GPU状態表示 ───────────────────────────────────────────────
    cuda_available = False
    cuda_info = ""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_available = True
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            cuda_info = f"✅ {gpu_name}（VRAM: {vram_gb:.1f}GB）"
        else:
            cuda_info = "⚠️ CUDAが使用できません（ドライバー未インストール、またはCPU専用環境）"
    except ImportError:
        cuda_info = "⚠️ PyTorchが未インストール（CUDAチェック不可）"

    st.caption(f"GPU状態: {cuda_info}")

    col_emb, col_llm = st.columns(2)

    # ─── 検索AI（Embedding）GPU設定 ──────────────────────────────
    with col_emb:
        with st.container(border=True):
            st.markdown("**🔍 検索AI（Embedding）**")
            current_emb_device = config.get("embedding_device", "auto")
            emb_device_options = ["auto", "cuda", "cpu"]
            emb_device_labels = {
                "auto": "🤖 自動（推奨）",
                "cuda": "⚡ GPU強制",
                "cpu": "💻 CPU強制",
            }
            emb_device_idx = (
                emb_device_options.index(current_emb_device)
                if current_emb_device in emb_device_options else 0
            )
            new_emb_device = st.selectbox(
                "デバイス",
                options=emb_device_options,
                index=emb_device_idx,
                format_func=lambda x: emb_device_labels.get(x, x),
                key="emb_device_select",
                help="「自動」はCUDAが使えればGPU、使えなければCPUを選択します",
            )
            if new_emb_device == "auto":
                st.caption("CUDAが使える場合は自動でGPUを使います")
            elif new_emb_device == "cuda":
                if not cuda_available:
                    st.warning("CUDAが使えません。CPUにフォールバックします")
                else:
                    st.success("GPUを使います")
            else:
                st.caption("CPUを使います")

            if st.button("検索AI設定を保存", key="save_emb_device", type="primary"):
                config["embedding_device"] = new_emb_device
                save_config(config)
                st.session_state.config = config
                app_config.EMBEDDING_DEVICE = new_emb_device
                try:
                    from core.embedder import reset_model
                    reset_model()
                except Exception:
                    pass
                st.success("✅ 保存しました。次回のインデックス作成時から有効になります。")

    # ─── 回答AI（LLM） Ollama GPU設定 ────────────────────────────
    with col_llm:
        with st.container(border=True):
            st.markdown("**💬 回答AI（LLM） — Ollama**")

            # Ollama稼働確認
            ollama_running = check_ollama_running()
            if ollama_running:
                st.success("🟢 Ollama 起動中", icon=None)
            else:
                st.error("🔴 Ollama 未起動", icon=None)
                st.caption(
                    "タスクトレイのOllamaアイコンをクリックして起動してください。\n"
                    "未インストールの場合は https://ollama.com からダウンロードしてください。"
                )

            st.divider()

            # インストール済みモデルのOllama登録状態を表示
            installed_models = [n for n in LLM_MODELS if is_model_downloaded(n)]
            if not installed_models:
                st.caption("インストール済みのモデルがありません")
            else:
                st.caption("各モデルのOllama登録状態（GPU推論に必要）：")
                for model_name in installed_models:
                    registered = is_registered_with_ollama(model_name) if ollama_running else False
                    short = model_name.split("（")[0]

                    col_m, col_btn, col_rereg = st.columns([3, 2, 2])
                    with col_m:
                        if registered:
                            st.markdown(f"✅ {short}")
                        else:
                            st.markdown(f"⬜ {short}")
                    with col_btn:
                        if not ollama_running:
                            st.button(
                                "Ollama未起動",
                                key=f"ollama_reg_disabled_{hash(model_name)}",
                                disabled=True,
                                use_container_width=True,
                            )
                        elif registered:
                            st.button(
                                "登録済み",
                                key=f"ollama_reg_done_{hash(model_name)}",
                                disabled=True,
                                use_container_width=True,
                            )
                        else:
                            if st.button(
                                "Ollamaに登録",
                                key=f"ollama_reg_action_{hash(model_name)}",
                                use_container_width=True,
                                type="primary",
                            ):
                                with st.spinner(f"登録中: {short}"):
                                    try:
                                        register_model_with_ollama(model_name)
                                        st.success(f"✅ 登録完了: {short}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"登録失敗: {e}")
                    # BUG-004修正: 設定変更（n_ctx拡張）を反映するための再登録ボタン
                    with col_rereg:
                        if ollama_running and registered:
                            if st.button(
                                "🔄 再登録",
                                key=f"ollama_rereg_{hash(model_name)}",
                                use_container_width=True,
                                help="n_ctxなどの設定変更を反映するために再登録します",
                            ):
                                with st.spinner(f"再登録中: {short}"):
                                    try:
                                        register_model_with_ollama(model_name)
                                        st.success(f"✅ 再登録完了: {short}（設定を更新しました）")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"再登録失敗: {e}")


def _render_embedding_recommended(embedding_ready: bool) -> None:
    """おすすめ検索AIモデル（multilingual-e5-large）を表示"""
    from core.config import EMBEDDING_MODEL_DIR, EMBEDDING_MODEL_ID

    if embedding_ready:
        st.success("✅ 推奨モデルはインストール済みです")
        return

    st.caption(
        "モノシリ推奨の検索AIモデルです。\n"
        "インデックス作成時に自動ダウンロードされますが、ここから状態を確認できます。"
    )
    with st.container(border=True):
        col_info, col_status = st.columns([3, 1])
        with col_info:
            st.markdown("**⭐ multilingual-e5-large**")
            st.caption("日本語・英語・ドイツ語など多言語で高精度な検索が可能なモデルです。")
            st.caption(f"サイズ: 約570MB  |  モデルID: {EMBEDDING_MODEL_ID}")
        with col_status:
            st.warning("未取得")
    st.info(
        "💡 インデックス作成時に自動でダウンロードされます。\n"
        "「ドキュメント管理」タブでフォルダを登録してインデックスを作成してください。"
    )


def _render_embedding_hf_download() -> None:
    """HuggingFace経由で検索AIモデルを追加（将来拡張用）"""
    from core.config import EMBEDDING_MODEL_ID

    st.caption(
        "HuggingFaceのモデルIDを入力して検索AIモデルを追加します。\n"
        "SentenceTransformers形式のモデルのみ対応しています。"
    )
    st.info(
        f"⚠️ 現在のバージョンでは、検索AIモデルは **{EMBEDDING_MODEL_ID}** 固定です。\n"
        "カスタム検索モデルへの対応は今後のアップデートで追加予定です。"
    )


def _render_embedding_manual_install() -> None:
    """検索AIモデルの手動インストール方法（自動DLに失敗した場合）"""
    from core.config import EMBEDDING_MODEL_DIR, EMBEDDING_MODEL_ID

    st.markdown("#### 手動インストール方法")
    st.caption("自動ダウンロードに失敗した場合は、以下の手順でインストールしてください。")
    st.markdown("**ステップ 1** — 以下のURLをブラウザで開き、「Files and versions」タブから全ファイルをダウンロード")
    st.code(f"https://huggingface.co/{EMBEDDING_MODEL_ID}", language=None)
    st.markdown("**ステップ 2** — ダウンロードしたファイルを以下のフォルダに配置")
    st.code(str(EMBEDDING_MODEL_DIR), language=None)
    st.markdown("**ステップ 3** — アプリを再起動")


def _render_hf_download(dm) -> None:
    """HuggingFaceダウンロードフォーム"""
    st.caption(
        "HuggingFaceのリポジトリIDとGGUFファイル名を入力してキューに追加します。\n"
        "例: `mmnga/Llama-3-ELYZA-JP-8B-GGUF` / `Llama-3-ELYZA-JP-8B-q4_k_m.gguf`"
    )
    with st.form("hf_download_form"):
        repo_id = st.text_input("リポジトリID", placeholder="例: mmnga/Llama-3-ELYZA-JP-8B-GGUF")
        filename = st.text_input("ファイル名（.gguf）", placeholder="例: Llama-3-ELYZA-JP-8B-q4_k_m.gguf")
        display_name = st.text_input("表示名（任意）", placeholder="例: ELYZA-JP-8B")
        submitted = st.form_submit_button("＋ キューに追加", type="primary")

    if submitted:
        if not repo_id or not filename:
            st.warning("リポジトリIDとファイル名を入力してください")
        elif not filename.endswith(".gguf"):
            st.warning("ファイル名は .gguf で終わる必要があります")
        else:
            name = display_name or filename.replace(".gguf", "")
            key = f"{name}（カスタム）"
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
    """ローカルGGUFファイル追加フォーム"""
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


# ─── タブ２: ネットワーク ─────────────────────────────────────

def _render_network_settings() -> None:
    """ネットワーク設定タブ"""
    from core.downloader import check_network_connectivity

    st.subheader("🌐 ネットワーク診断")
    if st.button("接続テスト実行", type="primary"):
        with st.spinner("HuggingFaceへの接続を確認中..."):
            ok, msg = check_network_connectivity()
        if ok:
            st.success(msg)
        else:
            st.error("接続に失敗しました")
            st.code(msg, language=None)

    st.divider()

    st.subheader("🔗 HuggingFaceミラー設定")
    st.caption(
        "HuggingFace本体に接続できない場合、ミラーサイトを利用できます。\n"
        "空欄にすると通常のHuggingFace（https://huggingface.co）を使用します。"
    )
    current_mirror = app_config.HF_MIRROR_URL
    new_mirror = st.text_input(
        "ミラーURL", value=current_mirror,
        placeholder="例: https://hf-mirror.com", key="hf_mirror_input",
    )
    if st.button("ミラーURLを保存"):
        app_config.HF_MIRROR_URL = new_mirror.strip()
        config = st.session_state.get("config", load_config())
        config["hf_mirror_url"] = new_mirror.strip()
        save_config(config)
        st.success(f"保存しました: {new_mirror.strip() or '（デフォルト）'}")

    st.divider()

    st.subheader("📥 手動ダウンロードガイド（回答AIモデル）")
    st.caption("自動ダウンロードに失敗する場合、ブラウザで直接ダウンロードできます。")
    st.markdown("**配置先フォルダ**")
    st.code(str(MODELS_DIR / "llm"), language=None)
    st.markdown("**各モデルの直接URL**")
    for model_name, model_info in LLM_MODELS.items():
        repo_id = model_info.get("repo_id")
        filename = model_info["filename"]
        if repo_id:
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            st.markdown(f"**{model_name}**")
            st.code(url, language=None)


# ─── タブ３: システム情報 ──────────────────────────────────────

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

        # GPU情報
        st.markdown("**GPU情報**")
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                vram_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                vram_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                st.text(f"GPU: {gpu_name}")
                st.text(f"VRAM: {vram_total:.1f}GB（使用中: {vram_allocated:.1f}GB）")
            else:
                st.text("GPU: CUDAが使用できません（CPUモード）")
        except ImportError:
            st.text("GPU: PyTorchが未インストール（CPUモード）")

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

        # Free層使用量表示
        try:
            from core.usage_tracker import get_usage_summary
            usage = get_usage_summary()
            st.markdown("**Free層 使用量**")
            st.text(f"インデックスファイル: {usage['files_used']:,} / {usage['files_max']:,} 件")
            st.text(f"{usage['month']} 質問数: {usage['questions_used']} / {usage['questions_max']} 回")
        except Exception:
            pass

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
            from core.config import CHROMA_DIR, HASH_MANIFEST_FILE
            shutil.rmtree(str(CHROMA_DIR), ignore_errors=True)
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            # ハッシュマニフェストを削除（単一ファイル方式・v3.4）
            if HASH_MANIFEST_FILE.exists():
                HASH_MANIFEST_FILE.unlink()
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

    tab_ai, tab_network, tab_system = st.tabs([
        "🤖 AIモデル",
        "🌐 ネットワーク",
        "ℹ️ システム情報",
    ])

    with tab_ai:
        _render_ai_models_tab(config)

    with tab_network:
        _render_network_settings()

    with tab_system:
        _render_system_info()
