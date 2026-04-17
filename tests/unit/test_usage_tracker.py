"""
core/usage_tracker.py の単体テスト
"""
from __future__ import annotations

import pytest

from core import usage_tracker as ut


class TestQuestionCount:
    def test_initial_count_zero(self, isolate_data_paths):
        assert ut.get_monthly_question_count() == 0

    def test_increment(self, isolate_data_paths):
        assert ut.increment_question_count() == 1
        assert ut.increment_question_count() == 2
        assert ut.get_monthly_question_count() == 2

    def test_remaining(self, isolate_data_paths):
        import core.config as cfg
        assert ut.get_remaining_questions() == cfg.FREE_MAX_QUESTIONS
        ut.increment_question_count()
        assert ut.get_remaining_questions() == cfg.FREE_MAX_QUESTIONS - 1

    def test_limit_not_reached_initially(self, isolate_data_paths):
        assert ut.is_question_limit_reached() is False


class TestFileCount:
    def test_initial_file_count(self, isolate_data_paths):
        assert ut.get_indexed_file_count() == 0

    def test_count_from_hash_manifest(self, isolate_data_paths):
        from core import hash_manager as hm
        hm.save_hashes("f1", {"/a.pdf": "h1", "/b.pdf": "h2"})
        hm.save_hashes("f2", {"/c.pdf": "h3"})
        assert ut.get_indexed_file_count() == 3

    def test_remaining_files(self, isolate_data_paths):
        import core.config as cfg
        assert ut.get_remaining_files() == cfg.FREE_MAX_FILES


class TestSummary:
    def test_summary_structure(self, isolate_data_paths):
        s = ut.get_usage_summary()
        assert "questions_used" in s
        assert "questions_max" in s
        assert "questions_remaining" in s
        assert "files_used" in s
        assert "files_max" in s
        assert "month" in s
        assert "年" in s["month"]
        assert "月" in s["month"]
