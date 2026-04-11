"""
モノシリ 使用量トラッキングモジュール（v3.4 新規）

Free層制限:
  - インデックス済みファイル: 最大 500件
  - 月間質問数: 最大 300回（暦月リセット）

Phase 1 β方針:
  - 上限到達時は通知のみ。課金要求は行わない。
  - カウント残数はチャット画面上部に常時表示。
  - 送信ボタンを押した時点で1カウント（LLM再試行はカウント外）。
"""
from __future__ import annotations
import json
import logging
import threading
from datetime import date
from pathlib import Path

from core.config import (
    USAGE_FILE,
    FREE_MAX_FILES,
    FREE_MAX_QUESTIONS,
)

logger = logging.getLogger(__name__)

_lock = threading.Lock()


# ─── 使用量ファイルの読み書き ────────────────────────────────────

def _load_usage() -> dict:
    """使用量データを読み込む"""
    if USAGE_FILE.exists():
        try:
            with open(USAGE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"使用量データ読み込みエラー: {e}")
    return {}


def _save_usage(data: dict) -> None:
    """使用量データを保存する"""
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── 月間質問カウント ─────────────────────────────────────────────

def _current_month_key() -> str:
    """今月のキー文字列を返す（例: "2026-04"）"""
    today = date.today()
    return f"{today.year:04d}-{today.month:02d}"


def get_monthly_question_count() -> int:
    """今月の質問数を返す"""
    data = _load_usage()
    month_key = _current_month_key()
    return data.get("questions", {}).get(month_key, 0)


def increment_question_count() -> int:
    """
    質問数を1増やして新しいカウントを返す。
    送信ボタン押下時に呼び出す（LLM再試行時は呼ばない）。
    """
    with _lock:
        data = _load_usage()
        month_key = _current_month_key()
        questions = data.setdefault("questions", {})
        count = questions.get(month_key, 0) + 1
        questions[month_key] = count
        _save_usage(data)
        logger.info(f"質問数カウント: {count}/{FREE_MAX_QUESTIONS}（{month_key}）")
        return count


def get_remaining_questions() -> int:
    """今月の残り質問数を返す（マイナスになっても0止まり）"""
    used = get_monthly_question_count()
    return max(0, FREE_MAX_QUESTIONS - used)


def is_question_limit_reached() -> bool:
    """今月の質問上限に達しているか返す"""
    return get_monthly_question_count() >= FREE_MAX_QUESTIONS


# ─── ファイル数チェック ───────────────────────────────────────────

def get_indexed_file_count() -> int:
    """
    インデックス済みのユニークファイル数を返す。
    hash_manifest のエントリ数（フォルダIDプレフィックスを除いたユニークパス数）から算出。
    """
    try:
        from core.hash_manager import _load_manifest
        manifest = _load_manifest()
        # キー形式 "folder_id:file_path" → file_path部分をカウント
        file_paths = set()
        for key in manifest:
            if ":" in key:
                file_paths.add(key.split(":", 1)[1])
        return len(file_paths)
    except Exception as e:
        logger.warning(f"ファイル数取得エラー: {e}")
        return 0


def get_remaining_files() -> int:
    """残りインデックス可能ファイル数を返す（0止まり）"""
    used = get_indexed_file_count()
    return max(0, FREE_MAX_FILES - used)


def is_file_limit_reached() -> bool:
    """インデックスファイル数が上限に達しているか返す"""
    return get_indexed_file_count() >= FREE_MAX_FILES


# ─── 使用量サマリー ───────────────────────────────────────────────

def get_usage_summary() -> dict:
    """
    チャット画面表示用の使用量サマリーを返す。

    Returns:
        {
            "questions_used": int,
            "questions_max": int,
            "questions_remaining": int,
            "questions_limit_reached": bool,
            "files_used": int,
            "files_max": int,
            "files_remaining": int,
            "files_limit_reached": bool,
            "month": str,  # 例: "2026年4月"
        }
    """
    today = date.today()
    month_str = f"{today.year}年{today.month}月"

    q_used = get_monthly_question_count()
    f_used = get_indexed_file_count()

    return {
        "questions_used": q_used,
        "questions_max": FREE_MAX_QUESTIONS,
        "questions_remaining": max(0, FREE_MAX_QUESTIONS - q_used),
        "questions_limit_reached": q_used >= FREE_MAX_QUESTIONS,
        "files_used": f_used,
        "files_max": FREE_MAX_FILES,
        "files_remaining": max(0, FREE_MAX_FILES - f_used),
        "files_limit_reached": f_used >= FREE_MAX_FILES,
        "month": month_str,
    }
