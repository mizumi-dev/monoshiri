"""
core/checkpoint_manager.py の単体テスト
"""
from __future__ import annotations

import pytest

from core import checkpoint_manager as cpm


class TestChunkProcessed:
    def test_is_chunk_processed_false_initially(self, isolate_data_paths):
        assert cpm.is_chunk_processed("abc") is False

    def test_mark_and_check(self, isolate_data_paths):
        cpm.mark_chunks_processed(["chunk-1", "chunk-2"])
        assert cpm.is_chunk_processed("chunk-1") is True
        assert cpm.is_chunk_processed("chunk-2") is True
        assert cpm.is_chunk_processed("chunk-999") is False

    def test_empty_list_noop(self, isolate_data_paths):
        cpm.mark_chunks_processed([])
        # エラーなく完了


class TestIncompleteFiles:
    def test_no_incomplete_initially(self, isolate_data_paths):
        assert cpm.get_incomplete_files("folder-1") == {}

    def test_mark_incomplete_then_complete(self, isolate_data_paths):
        cpm.mark_file_incomplete(
            folder_id="folder-1",
            file_path="/docs/a.pdf",
            file_hash="h1",
            expected_chunks=10,
            processed_chunks=3,
        )
        incomplete = cpm.get_incomplete_files("folder-1")
        assert "/docs/a.pdf" in incomplete
        assert incomplete["/docs/a.pdf"]["expected_chunks"] == 10

        cpm.mark_file_complete("folder-1", "/docs/a.pdf")
        incomplete = cpm.get_incomplete_files("folder-1")
        assert "/docs/a.pdf" not in incomplete


class TestClearAll:
    def test_clear_removes_file(self, isolate_data_paths):
        cpm.mark_chunks_processed(["c1"])
        assert cpm.CHECKPOINT_FILE.exists()
        cpm.clear_all_checkpoints()
        assert not cpm.CHECKPOINT_FILE.exists()


class TestRemoveFileChunks:
    def test_removes_prefix_matching(self, isolate_data_paths):
        cpm.mark_chunks_processed([
            "/docs/a.pdf_chunk_0",
            "/docs/a.pdf_chunk_1",
            "/docs/b.pdf_chunk_0",
        ])
        cpm.remove_file_chunks_from_checkpoint("/docs/a.pdf")
        assert cpm.is_chunk_processed("/docs/a.pdf_chunk_0") is False
        assert cpm.is_chunk_processed("/docs/b.pdf_chunk_0") is True
