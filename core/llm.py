"""
モノシリ LLM管理モジュール (Ollama バックエンド)
Ollama経由でGGUFモデルをローカルGPU推論する。
外部送信なし・完全ローカル動作。

TODO(仕様書更新): v3.4仕様書はLLMバックエンドとしてllama.cppを記載しているが、
  ADR-001に基づきOllama HTTP APIに移行済み。仕様書の「技術構成」セクションを更新すること。
TODO(仕様書更新): 実装では14B/Qwen3.5-9B/Nemotron-Nano-Japaneseの3モデルを追加対応していて、
  v3.4仕様書には3B/7Bの2モデルのみ記載。仕様書に追加モデルを反映すること。
"""
from __future__ import annotations
import json
import logging
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

import psutil

from core.config import (
    MODELS_DIR, LLM_MODELS, LLM_MAX_TOKENS, LLM_MODEL_MAX_TOKENS,
    OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)

# BUG-037修正: ollama rm→create のアトミック性を保証するロック
# RLock（再入可能）を使用し、force_reregister_model が register_model_with_ollama を
# 呼んでもデッドロックが起きないようにする。
_OLLAMA_REGISTER_LOCK = threading.RLock()

# カスタムモデルの永続化ファイル
CUSTOM_MODELS_FILE = MODELS_DIR / "custom_models.json"


# ─── RAM / システム情報 ───────────────────────────────────────

def get_total_ram_gb() -> float:
    """総RAM容量（GB）を返す"""
    return psutil.virtual_memory().total / (1024 ** 3)


def get_available_ram_gb() -> float:
    """現在の空きRAM（GB）を返す"""
    return psutil.virtual_memory().available / (1024 ** 3)


def recommend_model() -> str:
    """
    総RAMに基づいて推奨モデルを返す。
    v3.4仕様: 全環境で3B（軽量）がデフォルト推奨。
    16GB以上では7Bも選択可能だが、CPU環境での応答速度を優先し3Bを推奨。
    """
    model_keys = list(LLM_MODELS.keys())
    # 常に軽量（3B）をデフォルト推奨
    for key in model_keys:
        if "軽量" in key:
            return key
    return model_keys[0] if model_keys else ""


# ─── モデルパス管理 ───────────────────────────────────────────

def get_model_path(model_name: str) -> Path | None:
    """
    モデルのローカルパスを返す。
    ダウンロードされていない場合はNone。
    """
    if model_name not in LLM_MODELS:
        return None
    filename = LLM_MODELS[model_name]["filename"]
    path = MODELS_DIR / "llm" / filename
    return path if path.exists() else None


def is_model_downloaded(model_name: str) -> bool:
    """モデルがダウンロード済みかどうかを返す"""
    return get_model_path(model_name) is not None


def get_installed_models() -> list[str]:
    """ダウンロード済みモデルの表示名一覧を返す"""
    return [name for name in LLM_MODELS if is_model_downloaded(name)]


# ─── カスタムモデル管理 ───────────────────────────────────────

def load_custom_models() -> None:
    """カスタムモデル定義を読み込んでLLM_MODELSに追加する"""
    if not CUSTOM_MODELS_FILE.exists():
        return
    try:
        with open(CUSTOM_MODELS_FILE, "r", encoding="utf-8") as f:
            custom = json.load(f)
        for name, info in custom.items():
            if name not in LLM_MODELS:
                LLM_MODELS[name] = info
    except Exception as e:
        logger.warning(f"カスタムモデル読み込みエラー: {e}")


def save_custom_models() -> None:
    """カスタムモデル定義を保存する"""
    custom = {
        name: info
        for name, info in LLM_MODELS.items()
        if "（カスタム）" in name
    }
    CUSTOM_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CUSTOM_MODELS_FILE, "w", encoding="utf-8") as f:
        json.dump(custom, f, ensure_ascii=False, indent=2)


def add_local_gguf(gguf_path: Path, display_name: str | None = None) -> str:
    """
    ローカルGGUFファイルをモデルとして追加する（コピーして登録）。

    Returns:
        登録した表示名
    """
    model_dir = MODELS_DIR / "llm"
    model_dir.mkdir(parents=True, exist_ok=True)

    dest = model_dir / gguf_path.name
    if not dest.exists():
        import shutil
        shutil.copy2(str(gguf_path), str(dest))

    name = display_name or gguf_path.stem
    key = f"{name}（カスタム）"

    LLM_MODELS[key] = {
        "repo_id": None,
        "filename": gguf_path.name,
        "size_gb": round(dest.stat().st_size / (1024 ** 3), 1),
        "min_ram_gb": 8,
        "speed": "不明",
        "description": f"カスタムモデル: {gguf_path.name}",
        "chat_template": "chatml",
        "ollama_name": f"monoshiri-custom-{gguf_path.stem.lower()[:20]}",
    }
    save_custom_models()
    return key


# ─── Ollama API ───────────────────────────────────────────────

def is_model_loaded_in_ollama(model_name: str) -> bool:
    """
    Ollamaの /api/ps エンドポイントで、モデルが現在VRAMにロードされているか確認する。
    keep_alive=30m 設定でも30分以上アイドルだとアンロードされるため、
    warmup前にこれで確認することで不要なwarmupを省略できる。
    """
    model_info = LLM_MODELS.get(model_name, {})
    ollama_name = model_info.get("ollama_name", "")
    if not ollama_name:
        return False
    try:
        result = _ollama_get("/api/ps", timeout=3)
        running = result.get("models", [])
        return any(m.get("name", "").startswith(ollama_name) for m in running)
    except Exception:
        return False


def _ollama_get(endpoint: str, timeout: int = 5) -> dict:
    """OllamaへのGETリクエスト（urllib使用・追加依存なし）"""
    url = f"{OLLAMA_BASE_URL}{endpoint}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ollama_post(endpoint: str, payload: dict, timeout: int = 300) -> dict:
    """OllamaへのPOSTリクエスト（urllib使用・追加依存なし）"""
    url = f"{OLLAMA_BASE_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_ollama_running() -> bool:
    """Ollamaが起動しているかどうかを確認する"""
    try:
        _ollama_get("/api/version")
        return True
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    """
    Ollamaに登録済みのモデル名一覧を返す。
    Ollamaが未起動の場合は空リストを返す。
    """
    try:
        result = _ollama_get("/api/tags")
        return [m["name"] for m in result.get("models", [])]
    except Exception:
        return []


def is_registered_with_ollama(model_name: str) -> bool:
    """
    指定モデルがOllamaに登録済みかどうかを返す。
    登録名は LLM_MODELS[model_name]["ollama_name"] を使用する。
    """
    model_info = LLM_MODELS.get(model_name, {})
    ollama_name = model_info.get("ollama_name", "")
    if not ollama_name:
        return False
    registered = list_ollama_models()
    # "monoshiri-qwen25-7b" や "monoshiri-qwen25-7b:latest" を両方チェック
    return any(
        r == ollama_name or r.startswith(f"{ollama_name}:")
        for r in registered
    )


def register_model_with_ollama(model_name: str) -> None:
    """
    GGUFモデルをOllamaに登録する。
    Modelfileを一時ファイルに書き出し、`ollama create` を実行する。

    num_gpu 99 を設定することで、VRAMに収まる分だけ自動でGPUオフロードする。
    7B以下は全層GPU、14B（8.9GB）はRTX 4060 Ti（8GB VRAM）で部分オフロード＋RAM補完。

    Raises:
        FileNotFoundError: GGUFファイルが未ダウンロード
        RuntimeError: ollama createコマンドが失敗
    """
    model_path = get_model_path(model_name)
    if model_path is None:
        raise FileNotFoundError(
            f"GGUFファイルが見つかりません: {model_name}\n"
            "先にモデルをダウンロードしてください。"
        )

    model_info = LLM_MODELS.get(model_name, {})
    ollama_name = model_info.get("ollama_name", "")
    if not ollama_name:
        raise ValueError(f"ollama_nameが設定されていません: {model_name}")

    total_ram = get_total_ram_gb()
    # BUG-004修正: RAG_CONTEXT_MAX_LENGTH=6000文字 ≒ 3000トークン
    # + システムプロンプト(約300トークン) + 余裕 = 最低4096必要
    # 16GB以上では8192に拡張してコンテキスト欠落を防ぐ
    # SPEED修正: n_ctx=2048 に固定
    #   VRAM 8GB環境でQwen2.5-7B Q4_K_Mを使う場合:
    #     モデルウェイト: ~4.4GB
    #     KVキャッシュ(n_ctx=4096): ~1.6GB → 合計6.0GB (余裕あり)
    #     KVキャッシュ(n_ctx=2048): ~0.8GB → 合計5.2GB (さらに余裕)
    #   2048でRAGコンテキスト3000トークン+システム500トークンは収まらない場合は
    #   rag.pyのRAG_CONTEXT_MAX_LENGTHも合わせて調整する。
    #   社内文書RAGの典型的な質問+回答は2048以内で十分。
    n_ctx = 2048  # SPEED修正: 4096→2048 KVキャッシュ半減でVRAM節約・GPUフル活用

    # BUG-003修正: Modelfileにデフォルトシステムプロンプトを埋め込み
    # → Ollamaがモデルをメモリにロードする際から日本語モードを強制
    default_system = (
        "あなたは日本語専用の社内文書AIアシスタントです。"
        "必ず日本語のみで回答してください。英語で回答することは禁止です。"
    )

    # Modelfile: GGUFパスを絶対パスで指定。num_gpu 99 でVRAM上限まで自動オフロード
    # 14B（8.9GB）はRTX 4060 Ti（8GB VRAM）で部分オフロード、残りはRAMで処理
    # BUG-037b修正: "PARAMETER think false" を Modelfile から除去。
    #   この Ollama バージョンでは未サポートのパラメータで ollama create が rc=1 で失敗する。
    #   think 抑制は stream_answer / generate_answer の API ペイロード ("think": False) で実施済み。
    modelfile_content = (
        f"FROM {model_path.as_posix()}\n"
        f"PARAMETER num_gpu 99\n"
        f"PARAMETER num_ctx {n_ctx}\n"
        f"PARAMETER num_thread 8\n"
        f'SYSTEM """{default_system}"""\n'
    )

    logger.info(f"Ollamaに登録中: {ollama_name}")
    logger.debug(f"Modelfile:\n{modelfile_content}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".Modelfile", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(modelfile_content)
        modelfile_path = tf.name

    # BUG-037修正: _OLLAMA_REGISTER_LOCK で同時 ollama create を防ぐ
    with _OLLAMA_REGISTER_LOCK:
        try:
            result = subprocess.run(
                ["ollama", "create", ollama_name, "-f", modelfile_path],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                # BUG-037修正: result.stderr が None になるケース（race condition等）への対処
                # Ollamaはエラーをstdoutに書くこともあるため両方チェック
                stderr_msg = (result.stderr or "").strip()
                stdout_msg = (result.stdout or "").strip()
                detail = stderr_msg or stdout_msg or "(no output)"
                raise RuntimeError(
                    f"ollama createが失敗しました (rc={result.returncode}):\n{detail}"
                )
            logger.info(f"Ollama登録完了: {ollama_name}")
        finally:
            try:
                Path(modelfile_path).unlink()
            except Exception:
                pass


def force_reregister_model(model_name: str) -> None:
    """Ollama から既存登録を削除して再登録する（Modelfile 変更時に使用）。
    BUG-037修正: _OLLAMA_REGISTER_LOCK を保持したまま rm→create するため、
    他スレッドが rm と create の間に割り込んで create を実行することを防ぐ。
    RLock なので同スレッドからの register_model_with_ollama 呼び出しは再入可能。
    """
    with _OLLAMA_REGISTER_LOCK:
        model_info = LLM_MODELS.get(model_name, {})
        ollama_name = model_info.get("ollama_name", "")
        if ollama_name:
            try:
                subprocess.run(
                    ["ollama", "rm", ollama_name],
                    capture_output=True, timeout=30,
                )
                logger.info(f"Removed from Ollama: {ollama_name}")
            except Exception as e:
                logger.warning(f"ollama rm skipped: {e}")
        register_model_with_ollama(model_name)


def ensure_think_disabled_for_qwen35(model_name: str) -> None:
    """BUG-036: Qwen3.x モデルが PARAMETER think false 付き Modelfile で
    登録されていることを保証する。フラグファイルで初回のみ実行。"""
    if "Qwen3" not in model_name:
        return
    from core.config import CACHE_DIR
    flag_file = CACHE_DIR / ".think_fix_v1"
    if flag_file.exists():
        return
    if not is_model_downloaded(model_name):
        return
    logger.info(f"BUG-036: re-registering {model_name} with think=false ...")
    try:
        force_reregister_model(model_name)
        flag_file.write_text("think_fix_v1_applied\n", encoding="utf-8")
        logger.info("BUG-036: think fix applied.")
    except Exception as e:
        logger.warning(f"BUG-036: think fix failed: {e}")


# ─── 回答生成 ──────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """日本語専用RAGアシスタント用のシステムプロンプトを返す（BUG-003修正）

    BUG-006修正: thinking mode 無効化は Ollama API の think=False で実施。
        システムプロンプトへの /no_think は不要かつ逆効果（回答混入リスク）のため削除。
    BUG-015修正: /no_think をシステムプロンプトから削除。
        think=False（APIレベル）が正式な無効化手段。
        テキストとして残すと Qwen3 が回答に混入させる場合があった。
    """
    # BUG-016修正: グラウンディング強化。「自分の知識・推測を使わない」を明示する。
    #   Qwen3系は事前学習知識が豊富なため、文書内容と自前知識を混在させる傾向がある。
    #   「RAGコンテキスト外の知識は使用禁止」と明示的に制約することでハルシネーションを抑制。
    # BUG-033修正: 単位メタコメンタリー禁止を明示追加
    # BUG-035修正: 情報なし時に代替手段・外部情報源を案内しない旨を明示追加
    return (
        "あなたは日本語専用の社内文書AIアシスタントです。\n"
        "【最重要ルール】\n"
        "・必ず日本語のみで回答すること。英語での回答は禁止。\n"
        "・回答は必ず提供された社内文書の記述のみに基づくこと。\n"
        "・自分の事前学習知識・推測・補完は絶対に使わないこと。文書に書いてある情報だけを使う。\n\n"
        "厳守ルール：\n"
        "1. 【出典のみ】文書に書かれた情報だけで回答する。自分の知識で補足・推測しない\n"
        "2. 【不明は1文のみ】文書に記載がない場合は「文書には記載がありません。」の1文だけで答える。"
        "それ以上は何も書かない。以下の文は絶対に書いてはならない:\n"
        "  禁止例: 「〜を確認してください」「〜報告書をご覧ください」「〜の情報を得るためには」\n"
        "  禁止例: 「〜する必要があります」「〜をお勧めします」「より正確な情報」\n"
        "  禁止例: 追加の説明段落、まとめ段落、「要するに〜」「つまり〜」\n"
        "3. 【1〜2文で完結】回答は最大2文。余分な説明・謝辞・案内・注釈・まとめは一切不要\n"
        "4. 【無関係は即終了】社内文書が質問と無関係なら「社内資料に該当情報はありませんでした」とだけ答えて終了。"
        "それ以上何も言わない\n"
        "5. 英語の文書が参考資料に含まれていても、回答は必ず日本語にする\n"
        "6. 【単位厳守】数値は文書に記載の単位・数字をそのまま使うこと。"
        "百万円を億円に換算したり、億円を兆円に換算したりする変換は絶対に行わない。"
        "単位・数値型についてのコメント・注釈（例：「[億円]単位ではなく」）は禁止"
    )


def _build_messages_with_history(
    system_prompt: str,
    question: str,
    context: str,
    chat_history: list[dict] | None = None,
    max_history_turns: int = 2,
) -> list[dict]:
    """
    システムプロンプト・会話履歴・現在の質問からメッセージリストを構築する。
    BUG-001修正: 会話履歴をLLMに渡すことでマルチターン対話を実現。

    Args:
        system_prompt: システムプロンプト
        question: 現在の質問
        context: RAG検索で取得したコンテキスト
        chat_history: [{"role": "user"|"assistant", "content": str}] 形式の履歴
        max_history_turns: 過去何ターン分を含めるか（トークン溢れ防止）
            BUG-029修正: 3→2に削減（前の質問の回答内容が次の回答に汚染するのを軽減）
            デフォルト2: 2往復=4メッセージ≒1200トークン
            コンテキスト3000 + システム500 + 回答1024 = 合計5724 / 4096トークン考慮
    """
    messages = [{"role": "system", "content": system_prompt}]

    # 会話履歴を追加（直近N往復分）
    if chat_history:
        # role が "user" か "assistant" のもののみ
        valid_history = [
            m for m in chat_history
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        # 直近 max_history_turns 往復（= 2N メッセージ）に絞る
        recent = valid_history[-(max_history_turns * 2):]
        for msg in recent:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # 現在の質問（コンテキスト付き）
    # BUG-008修正: 【】括弧形式がQ&A生成テンプレートと誤解釈される問題を修正。
    #   「回答:」で終わるシンプルな形式に変更し、1件の回答だけ生成させる。
    # BUG-009修正: ユーザーメッセージ先頭の /no_think を削除。
    #   Qwen2.5は /no_think を特殊トークンとして認識せず、通常テキストとして学習済み。
    #   先頭に置くとモデルが回答後に /no_think を「プロンプトの続き」として出力してしまう。
    #   Qwen3用の /no_think はシステムプロンプトのみに残す（Qwen2.5には無害）。
    # BUG-009修正: ユーザーメッセージ先頭の /no_think を削除。
    # BUG-009b修正: 無関係な場合は「一言だけ答えてそれ以上何も言わない」を強調。
    # BUG-009c修正: [社内文書]...[/社内文書] の XML タグを == 区切りに変更。
    #   モデルが [/回答] のような対応する閉じタグを勝手に生成してしまう問題を防ぐ。
    # BUG-011修正: ==文書終わり== セパレータを除去。
    # BUG-012修正: ユーザーメッセージから「社内資料に該当情報はありませんでした」フレーズを削除。
    #   このフレーズをユーザーメッセージに含めると、モデルが正解を返した後に
    #   meta-commentaryとしてこのフレーズを再生成する問題が発生していた。
    #   「not relevant」ケースの指示はシステムプロンプトのみで与える。
    #   ユーザーメッセージは「1文で簡潔に答えて終了」のみ。
    # BUG-018修正: ユーザーメッセージに「文書外知識禁止」を明示。
    # BUG-019a修正: "==社内文書==" デリミタを廃止。
    #   Qwen3がプロンプトの "==社内文書==" を回答の先頭にエコーバックし、
    #   それがストップトークンに一致して空回答になる問題を修正。
    #   デリミタなしでコンテキストを直接埋め込む形式に変更。
    # BUG-019b修正: 数値単位保持ルールを追加。
    #   モデルが「197,345百万円」を「1兆9,700億円」と誤変換するハルシネーションを防ぐ。
    #   文書の数値・単位をそのまま使うよう明示的に指示する。
    # BUG-021修正: /no_think を回答:の直前に配置。
    #   Qwen3/3.5はChatML形式でユーザーターン末尾に /no_think があるとthinkingモードを無効化する。
    #   think: False パラメータだけでは不十分な場合があるため、トークンレベルで明示する。
    #   （Qwen2.5には無害: 通常テキストとして無視される）
    # BUG-022修正: 質問に指定された年度・期間・対象のデータのみを使うよう明示。
    #   複数年度のデータがコンテキストに入った場合に誤った年度を先に出力する問題を防ぐ。
    user_message = (
        f"以下の社内文書の抜粋を読み、末尾の質問に日本語で簡潔に答えてください。"
        f"文書にない情報は絶対に使わないこと。"
        f"数値は文書に記載の単位・値をそのまま使うこと（変換・換算禁止）。"
        f"質問に記載の年度・期間・対象に一致するデータのみを使うこと。答えたら終了。\n\n"
        f"{context}\n\n"
        f"質問: {question}\n"
        f"/no_think\n"
        f"回答:"
    )
    messages.append({"role": "user", "content": user_message})

    return messages


def generate_answer(
    model_name: str,
    question: str,
    context: str,
    max_tokens: int | None = None,
    chat_history: list[dict] | None = None,
) -> str:
    """
    Ollama経由で質問に対する回答を生成する。

    モデルが未登録の場合は自動でOllamaに登録してから推論する。

    Args:
        model_name: LLM_MODELSの表示名
        question: ユーザーの質問
        context: RAG検索で取得した文書コンテキスト
        max_tokens: 最大生成トークン数
        chat_history: 会話履歴（BUG-001修正）

    Returns:
        生成された回答文字列

    Raises:
        RuntimeError: Ollamaが起動していない場合
        FileNotFoundError: GGUFファイルが未ダウンロードの場合
    """
    if not check_ollama_running():
        raise RuntimeError(
            "Ollamaが起動していません。\n"
            "Ollamaアプリを起動してから再度お試しください。\n"
            "（タスクトレイのOllamaアイコンを確認してください）"
        )

    # 未登録の場合は自動登録
    if not is_registered_with_ollama(model_name):
        logger.info(f"Ollamaに未登録のため自動登録します: {model_name}")
        try:
            register_model_with_ollama(model_name)
        except Exception as _reg_err:
            # BUG-037修正: レース条件対応 - 再チェックで登録済みなら続行
            if is_registered_with_ollama(model_name):
                logger.info(f"再チェックで登録確認: {model_name}")
            else:
                raise

    model_info = LLM_MODELS.get(model_name, {})
    ollama_name = model_info.get("ollama_name", "")
    if not ollama_name:
        raise ValueError(f"ollama_nameが設定されていません: {model_name}")

    # ADR-002修正: モデル別MAX_TOKENSを優先使用（呼び出し側から明示指定された場合はそちらを使う）
    effective_max_tokens = max_tokens if max_tokens is not None else LLM_MODEL_MAX_TOKENS.get(model_name, LLM_MAX_TOKENS)

    system_prompt = _build_system_prompt()
    messages = _build_messages_with_history(
        system_prompt=system_prompt,
        question=question,
        context=context,
        chat_history=chat_history,
    )

    payload = {
        "model": ollama_name,
        "messages": messages,
        "stream": False,
        "think": False,          # BUG-006修正: Qwen3/3.5 thinking mode をOllamaレベルで無効化
        "keep_alive": "30m",     # BUG-031修正: モデルをVRAMに30分保持（デフォルト5分でアンロード→タイムアウト防止）
        "options": {
            "num_predict": effective_max_tokens,
            # BUG-014修正: temperature=0.1 に設定（デフォルト0.8はRAGに不適切）
            #   低温度にすることでモデルが文書内容に忠実な回答を生成する。
            #   ハルシネーション（文書にない情報の生成）を大幅に抑制する最重要設定。
            "temperature": 0,
            "top_p": 0.9,
            # BUG-017修正: repeat_penalty=1.15 で同一トークンの繰り返し生成を抑制
            #   デフォルト1.1より若干強めにして、冗長な繰り返し表現を防ぐ
            "repeat_penalty": 1.2,   # BUG-030修正: 1.15→1.2 段落繰り返し抑制強化
            # BUG-007修正: 特殊トークンでアシスタント応答の暴走を防ぐ
            # BUG-008修正: "\n質問:" を追加 → Q&A形式の連続生成を強制停止
            # BUG-009修正: "/no_think" を追加 → Qwen2.5がプロンプト断片を出力した時点で停止
            # BUG-009b修正: "\nそれでは" "\n（注" を追加 → 無関係回答後のメタコメンタリー生成を停止
            # BUG-009c修正: " [/" "[/" を追加 → "[/回答]"等のXML閉じタグ生成を停止
            #              "\nこの回答は" を追加 → 回答説明コメンタリー生成を停止
            # BUG-011修正: "==文書終わり" を追加 → プロンプトの区切り文字がエコーされる問題を停止
            # BUG-012修正: "```" "\n\n社内資料" "\n\n質問に直接" を追加 → meta-commentary停止
            # BUG-013修正: "\n以上は" "\nこれでご質問" "\nご確認いただければ" を追加
            #   → 回答後の自己否定ループ「以上は間違っており...」による繰り返しを停止
            # BUG-017修正: Qwen3系のメタコメンタリーパターンを追加
            #   "\n補足:" "\n注記:" "\n参考:" → 回答後の補足説明生成を停止
            #   "\n上記の" → 自己言及的な繰り返しを停止
            # BUG-019a修正: "==社内文書==" をストップトークンから削除
            #   user_messageから"==社内文書=="デリミタを廃止したため、
            #   モデルが生成中にこのトークンを出すリスクがなくなった。
            #   廃止前はモデルが回答先頭に"==社内文書=="をエコーバックして空回答になる問題があった。
            # BUG-023修正: "\n【" を追加 → コンテキストエコーを停止
            #   無関係クエリで「not found」回答後にモデルがRAGコンテキスト
            #   （【ファイル名 p.X】形式）をそのまま出力し続ける問題を防ぐ。
            # BUG-025修正: "【" 単独をストップトークンから削除
            #   "【" alone が引用形式の回答（「【出典】...」等）を空にしてしまう問題を修正。
            #   改行後のエコーのみ停止すれば十分なため "\n【" のみ残す。
            # BUG-027修正: "/no_think" をストップトークンから削除（stream_answerと同様）
            # BUG-030修正: "\nご提示いただいた" を追加
            #   Qwen3.5が正解を出した後にメタコメンタリー段落（「ご提示いただいた文書には...」）を
            #   繰り返すループを防ぐ。第2段落以降のみ \n が先行するため、先頭の「not found」
            #   応答（"ご提示..." が最初のテキスト）には影響しない。
            # BUG-032修正: "/answer" を追加
            #   Qwen2.5が回答末尾に "/answer" を生成するケースへの対処。
            # BUG-035修正: 情報なし時の代替案内パターンを追加
            "stop": [
                "<|im_end|>", "<|endoftext|>", "<|im_start|>",
                "\n質問:",
                "\nそれでは", "\n（注", "\nもし具体的",
                " [/", "[/", "\nこの回答は",
                "==文書終わり",
                "```", "\n\n質問に直接",
                "\n以上は", "\nこれでご質問", "\nご確認いただければ",
                "\n補足:", "\n注記:", "\n参考:", "\n上記の",
                "\n【",
                "\n\n回答:", "\n回答:",     # BUG-029修正: 回答エコーループ防止
                "\nご提示いただいた",       # BUG-030修正: メタコメンタリー段落ループ防止
                "\n/answer", "/answer",     # BUG-032修正: /answerストップトークン漏れ防止
                "\n以下の方法", "\n以下の",  # BUG-035修正: 代替手段リスト案内を停止
                "\nをお勧め", "\nお勧め",    # BUG-035修正: お勧めパターン
                "\nより詳細", "\n詳しくは",  # BUG-035修正: 詳細案内パターン
                # BUG-035修正（段落版）: "要するに" 等の冗長まとめ段落を止める
                "\n要するに", "\nつまり、", "\nすなわち",
                "\nまとめると", "\n要約すると", "\n言い換えれば",
            ],
        },
    }
    # BUG-020修正: "\n\n社内資料" をストップトークンから削除
    #   system_promptのルール4が "社内資料に該当情報はありませんでした" と返すよう指示しているため、
    #   "\n\n社内資料" を止めると正当な応答が空になってしまう問題があった。

    logger.info(f"Ollama推論開始: {ollama_name}")
    try:
        response = _ollama_post("/api/chat", payload, timeout=300)
        answer = response["message"]["content"].strip()
        # BUG-006修正: <think>...</think> ブロックを除去（安全ネット）
        import re as _re
        answer = _re.sub(r"<think>.*?</think>", "", answer, flags=_re.DOTALL).strip()
        # BUG-029修正: 先頭の「回答:」「回答：」プレフィックスを除去
        answer = _re.sub(r"^回答[:：]\s*", "", answer).strip()
        # BUG-008修正: "\n質問:" 以降のQ&A連続生成を後処理でも除去（安全ネット）
        if "\n質問:" in answer:
            answer = answer.split("\n質問:")[0].strip()
        # BUG-008/009/011/012修正: プロンプト断片・区切り文字・メタコメンタリーの後処理除去（安全ネット）
        # BUG-019a修正: "==社内文書==" をpost-processingリストから削除
        # BUG-020修正: "\n\n社内資料" をpost-processingリストから削除（stop tokenからも削除済み）
        # BUG-023修正: "\n【" を追加 → コンテキストエコーをpost-processingでも除去
        # BUG-032修正: "/answer" を追加（安全ネット）
        # BUG-035修正: 代替手段案内パターンを追加
        for _sep in ["\n---", " ---", "[/社内文書]", "```", "/no_think",
                     "\nそれでは", "\n（注", "\nもし具体的",
                     " [/", "[/", "\nこの回答は", "==文書終わり",
                     "\n\n質問に直接",
                     "\n以上は", "\nこれでご質問", "\nご確認いただければ",
                     "\n補足:", "\n注記:", "\n参考:", "\n上記の",
                     "\n【",
                     "\n\n回答:", "\n回答:",       # BUG-029修正: 回答エコーループ除去
                     "\n/answer", "/answer",       # BUG-032修正: /answerストップトークン漏れ
                     "\n以下の方法", "\n以下の",    # BUG-035修正: 代替手段リスト
                     "\nをお勧め", "\nお勧め",      # BUG-035修正: お勧めパターン（改行あり）
                     "\nより詳細", "\n詳しくは"]:   # BUG-035修正: 詳細案内パターン
            if _sep in answer:
                answer = answer.split(_sep)[0].strip()
        # BUG-035修正（インライン版）: 改行なしでも推薦文・外部参照文を除去
        #   例: "...できません。より正確な回答を得るには..." → "...できません。" まで
        #   例: "...不可能です。三菱重工の公式プレスリリース...をお勧めします。" → 不可能です。まで
        for _inline_sep in ["。より正確", "。詳しくは", "をお勧めします", "ことをお勧め",
                             "をご覧ください", "をご参照ください",
                "確認する必要があります", "確認する必要が", "をご確認ください",
                             "してみてください", "をお確かめください"]:
            if _inline_sep in answer:
                # 「。」を含む場合はその直後（句点含む）でカット
                if _inline_sep.startswith("。"):
                    answer = answer.split(_inline_sep)[0] + "。"
                else:
                    answer = answer.split(_inline_sep)[0].strip()
                    # 末尾に句点がなければ補完（切り捨て感を減らす）
                    if answer and not answer.endswith("。") and not answer.endswith("。"):
                        answer = answer.rstrip("、") + "。"
                break
        # BUG-033修正: 単位メタコメンタリー除去（安全ネット）
        #   例: "112,300名です。[億円]単位ではなく..." → "112,300名です。" のみ残す
        #   文末に「。」がある場合、それ以降に "[" が続くパターンをカット
        _answer_clean = _re.sub(r"。\[.*", "。", answer)
        if _answer_clean != answer:
            logger.info("BUG-033: 単位メタコメンタリーを除去")
            answer = _answer_clean.strip()
        logger.info(f"Ollama推論完了: {len(answer)}文字")
        return answer
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollamaとの通信に失敗しました: {e}\n"
            "Ollamaが起動しているか確認してください。"
        )
    except (KeyError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Ollamaの応答を解析できませんでした: {e}")


def stream_answer(
    model_name: str,
    question: str,
    context: str,
    max_tokens: int | None = None,
    chat_history: list[dict] | None = None,
):
    """
    Ollama経由でストリーミング回答を生成するジェネレータ。
    トークンが届くたびに文字列を yield する。

    使い方:
        for chunk in stream_answer(...):
            print(chunk, end="", flush=True)

    Args:
        model_name: LLM_MODELSの表示名
        question: ユーザーの質問
        context: RAG検索で取得した文書コンテキスト
        max_tokens: 最大生成トークン数
        chat_history: 会話履歴（BUG-001修正）

    Raises:
        RuntimeError: Ollamaが起動していない場合
        FileNotFoundError: GGUFファイルが未ダウンロードの場合
    """
    if not check_ollama_running():
        raise RuntimeError(
            "Ollamaが起動していません。\n"
            "タスクトレイのOllamaアイコンを確認してください。"
        )

    if not is_registered_with_ollama(model_name):
        logger.info(f"Ollamaに未登録のため自動登録します: {model_name}")
        try:
            register_model_with_ollama(model_name)
        except Exception as _reg_err:
            # BUG-037修正: バックグラウンドスレッドとのレース条件で登録失敗しても
            # 再チェックして登録済みなら続行（同時実行で他スレッドが完了した場合）
            if is_registered_with_ollama(model_name):
                logger.info(f"再チェックで登録確認: {model_name}（レース条件で別スレッドが完了）")
            else:
                raise

    model_info = LLM_MODELS.get(model_name, {})
    ollama_name = model_info.get("ollama_name", "")
    if not ollama_name:
        raise ValueError(f"ollama_nameが設定されていません: {model_name}")

    # ADR-002修正: モデル別MAX_TOKENSを優先使用
    effective_max_tokens = max_tokens if max_tokens is not None else LLM_MODEL_MAX_TOKENS.get(model_name, LLM_MAX_TOKENS)

    system_prompt = _build_system_prompt()
    messages = _build_messages_with_history(
        system_prompt=system_prompt,
        question=question,
        context=context,
        chat_history=chat_history,
    )

    payload = json.dumps({
        "model": ollama_name,
        "messages": messages,
        "stream": True,
        "think": False,          # BUG-006修正: Qwen3/3.5 thinking mode をOllamaレベルで無効化
        "keep_alive": "30m",     # BUG-031修正: モデルをVRAMに30分保持（デフォルト5分でアンロード→タイムアウト防止）
        "options": {
            "num_predict": effective_max_tokens,
            # BUG-014修正: temperature=0.1 に設定（デフォルト0.8はRAGに不適切）
            #   低温度にすることでモデルが文書内容に忠実な回答を生成する。
            #   ハルシネーション（文書にない情報の生成）を大幅に抑制する最重要設定。
            "temperature": 0,
            "top_p": 0.9,
            # BUG-017修正: repeat_penalty=1.15 で同一トークンの繰り返し生成を抑制
            # BUG-030修正: repeat_penalty=1.2 に強化（段落レベル繰り返し抑制）
            "repeat_penalty": 1.2,
            # BUG-007修正: ストップトークンを明示してアシスタント応答の後に
            # <endoftext> や次のユーザーターンを生成し続けるのを防ぐ
            # BUG-008修正: "\n質問:" を追加 → Q&A形式の連続生成を強制停止
            # BUG-009修正: "/no_think" を追加 → Qwen2.5がプロンプト断片を出力した時点で停止
            # BUG-009b修正: "\nそれでは" "\n（注" を追加 → メタコメンタリー生成を停止
            # BUG-009c修正: " [/" "[/" を追加 → XML閉じタグ生成を停止
            #              "\nこの回答は" を追加 → 回答説明コメンタリー生成を停止
            # BUG-011修正: "==文書終わり" を追加 → プロンプト区切り文字エコーを停止
            # BUG-012修正: "```" "\n\n社内資料" "\n\n質問に直接" を追加 → meta-commentary停止
            # BUG-013修正: "\n以上は" "\nこれでご質問" "\nご確認いただければ" を追加
            #   → 回答後の自己否定ループ「以上は間違っており...」による繰り返しを停止
            # BUG-017修正: Qwen3系のメタコメンタリーパターンを追加
            # BUG-019a修正: "==社内文書==" をストップトークンから削除（デリミタ廃止に伴う）
            # BUG-028修正: Ollamaのstop tokensを特殊トークンのみに絞る
            #   従来のstop tokens（"\n質問:", "```", "[/", "\nそれでは"等）は
            #   Qwen3.5 の <think>...</think> ブロック内部でも発火し、Ollamaが
            #   </think> 生成前にストリームを終了させて _in_think=True のまま
            #   done:True → 空回答 / 誤情報なしメッセージ になる問題が判明。
            #   解決策: Ollamaには特殊EOTトークンのみを渡し、それ以外の
            #   メタコメンタリー停止パターンはストリーミングフィルタで処理する
            #   （thinking終了後の回答部分にのみ適用される）。
            "stop": [
                "<|im_end|>", "<|endoftext|>", "<|im_start|>",
                # BUG-035修正: 代替手段・推薦文を Ollama レベルで止める（streaming版）
                "\nをお勧め", "\nお勧め",
                "\nより詳細", "\n詳しくは",
                "\n以下の方法", "\n以下の",
                "\n要するに", "\nつまり、", "\nすなわち",
                "\nまとめると", "\n言い換えれば",
                "より正確な", "確認する必要が",
                "/answer",
            ],
        },
        # BUG-020修正: "\n\n社内資料" をstream_answerのストップトークンからも削除
    }).encode("utf-8")

    url = f"{OLLAMA_BASE_URL}/api/chat"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    logger.info(f"Ollamaストリーミング推論開始: {ollama_name}")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            # BUG-006修正: Qwen3 thinking token フィルタリング
            # /no_think で無効化するが、万一 <think>...</think> が来た場合も除去する
            _in_think = False
            _buf = ""
            # BUG-024修正: thinkingが長引く場合の時間ベースタイムアウト
            #   think:False + /no_think 設定でも一部クエリでthinkingが抑制されない問題への対処。
            #   CPU推論では字数カウントが不安定なため、時間で計測。
            # BUG-026修正: タイムアウト値を20→90秒に変更
            #   天気などの無関係クエリ: thinking が80+秒継続（infinite loop的）→ 90秒で打ち切り
            #   JR東日本等の有効クエリ: thinking が ~30-50秒で完了 → 90秒以内で正常回答
            #   20秒では有効クエリのthinkingも打ち切ってしまい "情報なし" 誤回答が発生していた
            # BUG-036修正: タイムアウト値を90→300秒に変更
            #   Qwen3.5-9B使用時にthink:False + /no_think 設定でも全クエリでthinkingが発動。
            #   英語PDF（米国企業財務報告書）に対する日本語質問では thinking が130〜170秒継続。
            #   90秒タイムアウトでは全件「社内文書に含まれていません」誤回答になる問題を修正。
            #   300秒ならほぼすべての正規クエリでthinkingが完了して正答が得られる。
            import time as _time
            _think_start_time: float | None = None
            _THINK_TIMEOUT_SECS = 300  # BUG-036: 90→300秒（Qwen3.5-9B thinking完了待ち）
            # BUG-028修正: Ollamaのstop tokensから除外した自然言語パターンを
            #   ストリーミングフィルタ側で処理するためのリスト。
            #   _in_think=False（回答部分）のみに適用し、thinkingブロック内では無視。
            #   ループ外で一度だけ計算してパフォーマンス最適化。
            _STREAM_STOP_PATTERNS = [
                "\n質問:", "/no_think", "\n---", " ---", "[/社内文書]", "```",
                "\nそれでは", "\n（注", "\nもし具体的",
                " [/", "[/", "\nこの回答は", "==文書終わり",
                "\n\n質問に直接",
                "\n以上は", "\nこれでご質問", "\nご確認いただければ",
                "\n補足:", "\n注記:", "\n参考:", "\n上記の",
                "\n【",
                "\nご提示いただいた",  # BUG-030修正: メタコメンタリー段落ループ防止
                "\n/answer", "/answer",     # BUG-032修正: /answerストップトークン漏れ防止
                "\n以下の方法", "\n以下の",  # BUG-035修正: 代替手段リスト案内を停止
                "\nをお勧め", "\nお勧め",    # BUG-035修正: お勧めパターン
                "\nより詳細", "\n詳しくは",  # BUG-035修正: 詳細案内パターン
                # BUG-035修正（段落版）: "要するに" 等の冗長まとめ段落を止める
                "\n要するに", "\n\n要するに",
                "\nつまり、", "\nすなわち",
                "\nまとめると", "\n要約すると", "\n言い換えれば",
                # BUG-035修正（インライン版）: 改行なし推薦文をストリーミング中に検知
                "をお勧めします", "ことをお勧め",
                "より正確な",
                "をご覧ください", "をご参照ください",
            ]
            _MAX_BUF_TAIL = max(len(p) for p in _STREAM_STOP_PATTERNS) + 1

            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        # <think>ブロックをフィルタリング
                        _buf += token
                        while True:
                            if not _in_think:
                                start = _buf.find("<think>")
                                # BUG-028修正: Ollama stop tokensから除外したパターンを
                                #   ストリーミングフィルタで処理（回答部分のみ適用）。
                                #   thinking内部では発火しないよう _in_think=False の場合のみ。
                                _earliest_stop = -1
                                for _pat in _STREAM_STOP_PATTERNS:
                                    _pos = _buf.find(_pat)
                                    if _pos != -1 and (_earliest_stop == -1 or _pos < _earliest_stop):
                                        _earliest_stop = _pos

                                if start == -1 or (_earliest_stop != -1 and _earliest_stop < start):
                                    # stop patternが先に出現（または<think>なし）
                                    if _earliest_stop != -1:
                                        # stop patternを検出: それまでを出力してストリーミング終了
                                        if _earliest_stop > 0:
                                            yield _buf[:_earliest_stop]
                                        _buf = ""
                                        return  # ジェネレータ終了
                                    # stop patternなし・<think>なし: 末尾は部分パターンの可能性があるため保留
                                    safe_end = max(0, len(_buf) - _MAX_BUF_TAIL)
                                    if safe_end > 0:
                                        yield _buf[:safe_end]
                                        _buf = _buf[safe_end:]
                                    break
                                else:
                                    # <think> 前のテキストを出力
                                    if start > 0:
                                        yield _buf[:start]
                                    _buf = _buf[start + 7:]
                                    _in_think = True
                                    _think_start_time = _time.time()  # BUG-024: thinking開始時刻記録
                            else:
                                end = _buf.find("</think>")
                                if end == -1:
                                    # まだ</think>が来ていない: バッファをクリア
                                    # BUG-021b修正: 空文字をyieldしてStreamlitのWebSocket接続を維持。
                                    #   thinking中はトークンを捨てるが、yieldしないとStreamlitが
                                    #   タイムアウトする問題を防ぐ。
                                    _buf = _buf[-8:] if len(_buf) > 8 else _buf
                                    yield ""  # keep-alive: Streamlit接続維持
                                    # BUG-024修正: thinkingが長くなりすぎたら時間で打ち切り
                                    if (_think_start_time is not None and
                                            _time.time() - _think_start_time > _THINK_TIMEOUT_SECS):
                                        elapsed = _time.time() - _think_start_time
                                        logger.warning(
                                            f"BUG-024: thinking {elapsed:.1f}秒超過 → 早期打ち切り"
                                        )
                                        yield "申し訳ありませんが、ご質問に関連する情報は社内文書には含まれていません。"
                                        return
                                    break
                                else:
                                    _buf = _buf[end + 8:]
                                    _in_think = False
                                    _think_start_time = None  # thinking終了

                    if chunk.get("done"):
                        # BUG-027修正: thinking中にストリームが終了した場合（stop token が
                        #   thinking内部で発火 or max_tokens到達）のフォールバック
                        if _in_think:
                            logger.warning(
                                "BUG-027: thinking未完了で done:True → stop token が"
                                " thinking内部で発火した可能性あり。情報なしメッセージを返す。"
                            )
                            yield "申し訳ありませんが、ご質問に関連する情報は社内文書には含まれていません。"
                            break
                        # ストリーム終了: バッファに残った未出力テキストをflush
                        if not _in_think and _buf:
                            import re as _re
                            # BUG-007/009修正: 末尾の特殊トークン・プロンプト断片を除去
                            clean = _buf.split("<|im_end|>")[0].split("<|endoftext|>")[0].split("<|im_start|>")[0]
                            # BUG-008/009修正: プロンプト断片・think ブロックの後処理（安全ネット）
                            clean = _re.sub(r"<think>.*?</think>", "", clean, flags=_re.DOTALL)
                            # BUG-019a修正: "==社内文書==" をdone-blockのpost-processingから削除
                            # BUG-020修正: "\n\n社内資料" をdone-blockのpost-processingからも削除
                            # BUG-023修正: "\n【" を追加 → コンテキストエコー後処理除去
                            for _sep in ["\n質問:", "/no_think", "\n---", " ---", "[/社内文書]", "```",
                                         "\nそれでは", "\n（注", "\nもし具体的",
                                         " [/", "[/", "\nこの回答は", "==文書終わり",
                                         "\n\n質問に直接",
                                         "\n以上は", "\nこれでご質問", "\nご確認いただければ",
                                         "\n補足:", "\n注記:", "\n参考:", "\n上記の",
                                         "\n【",
                                         "\n/answer", "/answer",     # BUG-032修正: /answerストップトークン漏れ防止
                                         "\n以下の方法", "\n以下の",  # BUG-035修正: 代替手段リスト案内を停止
                                         "\nをお勧め", "\nお勧め",    # BUG-035修正: お勧めパターン
                                         "\nより詳細", "\n詳しくは"]: # BUG-035修正: 詳細案内パターン
                                if _sep in clean:
                                    clean = clean.split(_sep)[0]
                            # BUG-035修正（インライン版）: 改行なし推薦文・外部参照文を除去
                            for _isep in ["。より正確", "。詳しくは", "をお勧めします",
                                          "ことをお勧め", "をご覧ください", "をご参照ください",
                                          "をご確認ください", "してみてください"]:
                                if _isep in clean:
                                    if _isep.startswith("。"):
                                        clean = clean.split(_isep)[0] + "。"
                                    else:
                                        clean = clean.split(_isep)[0].strip()
                                    break
                            # BUG-033修正: 単位メタコメンタリー除去（安全ネット）
                            import re as _re_clean
                            _cm = _re_clean.sub(r"。\[.*", "。", clean)
                            if _cm != clean:
                                clean = _cm
                            if clean.strip():
                                yield clean.strip()
                            _buf = ""
                        break
                except json.JSONDecodeError:
                    continue
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollamaとの通信に失敗しました: {e}\n"
            "Ollamaが起動しているか確認してください。"
        )


def warmup_model(model_name: str) -> bool:
    """
    モデルをOllamaにロードしてウォームアップする。
    アプリ起動時に呼ぶことで、初回質問の待ち時間を短縮する。

    Returns:
        True: ウォームアップ成功 / False: 失敗（Ollama未起動など）
    """
    if not check_ollama_running():
        return False
    if not is_registered_with_ollama(model_name):
        try:
            register_model_with_ollama(model_name)
        except Exception as e:
            logger.warning(f"ウォームアップ中の登録失敗: {e}")
            return False

    model_info = LLM_MODELS.get(model_name, {})
    ollama_name = model_info.get("ollama_name", "")
    if not ollama_name:
        return False

    try:
        # 最小リクエストでモデルをVRAMにロード
        _ollama_post("/api/chat", {
            "model": ollama_name,
            "messages": [{"role": "user", "content": "."}],
            "stream": False,
            "options": {"num_predict": 1},
        })
        logger.info(f"warmup complete: {model_name}")
        return True
    except Exception as e:
        logger.warning(f"warmup failed: {e}")
        return False