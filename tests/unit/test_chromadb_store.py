"""
core/chromadb_store.py の単体テスト
"""
from __future__ import annotations

import pytest

from core import chromadb_store as cs


class TestSingleton:
    def test_get_chroma_client_returns_same_instance(self, isolate_data_paths):
        c1 = cs.get_chroma_client()
        c2 = cs.get_chroma_client()
        assert c1 is c2

    def test_reset_creates_new_instance(self, isolate_data_paths):
        c1 = cs.get_chroma_client()
        cs.reset_chroma_client()
        c2 = cs.get_chroma_client()
        assert c1 is not c2


class TestCollection:
    def test_get_collection_creates_with_cosine_space(self, isolate_data_paths):
        col = cs.get_collection()
        assert col is not None
        # コレクション名は config.CHROMA_COLLECTION
        import core.config as cfg
        assert col.name == cfg.CHROMA_COLLECTION

    def test_get_index_stats_empty(self, isolate_data_paths):
        stats = cs.get_index_stats()
        assert "total_chunks" in stats
        assert stats["total_chunks"] == 0

    def test_add_and_count(self, isolate_data_paths):
        col = cs.get_collection()
        col.add(
            ids=["x1", "x2"],
            embeddings=[[0.1] * 1024, [0.2] * 1024],
            documents=["doc1", "doc2"],
            metadatas=[{"src": "a"}, {"src": "b"}],
        )
        stats = cs.get_index_stats()
        assert stats["total_chunks"] == 2


class TestDeleteAll:
    def test_delete_all_chunks_clears_collection(self, isolate_data_paths):
        col = cs.get_collection()
        col.add(
            ids=["x1"],
            embeddings=[[0.1] * 1024],
            documents=["d"],
            metadatas=[{"src": "test"}],
        )
        assert cs.get_index_stats()["total_chunks"] == 1
        cs.delete_all_chunks()
        assert cs.get_index_stats()["total_chunks"] == 0
