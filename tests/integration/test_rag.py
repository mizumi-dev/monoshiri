"""
core/rag.py の結合テスト（要 Ollama + インデックス済データ）
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core import rag


pytestmark = [pytest.mark.ollama, pytest.mark.slow, pytest.mark.gpu]


@pytest.fixture
def indexed_collection(isolate_data_paths, tmp_path):
    """テスト用のテキストを手動インデックスしたコレクション"""
    from core.chromadb_store import get_collection
    from core.embedder import embed_texts

    collection = get_collection()

    texts = [
        "モノシリは完全ローカル動作の社内資料探索AIです。資料は一切外に出ません。",
        "ChromaDBをベクトルDBとして採用し、HNSWlibベースの高速検索を実現しています。",
        "Ollamaを通じてローカルLLMで推論を行い、インターネット接続なしで動作します。",
    ]
    embeddings = embed_texts(texts)
    metadatas = [
        {"file_name": "intro.txt", "file_path": "/test/intro.txt", "folder_path": "/test", "page": 1},
        {"file_name": "tech.txt", "file_path": "/test/tech.txt", "folder_path": "/test", "page": 1},
        {"file_name": "tech.txt", "file_path": "/test/tech.txt", "folder_path": "/test", "page": 2},
    ]
    collection.add(
        ids=[f"test_chunk_{i}" for i in range(len(texts))],
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return collection


class TestSearchSimilar:
    def test_empty_collection_returns_empty(self, isolate_data_paths):
        """未インデックスでは空結果"""
        result = rag.search_similar("モノシリとは", top_k=5, similarity_threshold=0.0)
        assert result == []

    def test_finds_relevant_chunks(self, indexed_collection):
        result = rag.search_similar(
            "モノシリはどんなAIですか？",
            top_k=5,
            similarity_threshold=0.3,
        )
        assert len(result) > 0
        assert all("similarity" in r for r in result)
        assert all(0.0 <= r["similarity"] <= 1.0 for r in result)

    def test_high_threshold_filters_results(self, indexed_collection):
        """閾値を高くすれば結果が減る"""
        low = rag.search_similar("AI", top_k=10, similarity_threshold=0.0)
        high = rag.search_similar("AI", top_k=10, similarity_threshold=0.95)
        assert len(high) <= len(low)


class TestBuildContext:
    def test_includes_file_info(self):
        items = [{
            "text": "サンプルテキスト",
            "metadata": {"file_name": "a.pdf", "page": 3},
            "similarity": 0.8,
        }]
        ctx = rag.build_context(items)
        assert "a.pdf" in ctx
        assert "p.3" in ctx
        assert "サンプルテキスト" in ctx

    def test_respects_max_length(self):
        items = [{
            "text": "あ" * 1000,
            "metadata": {"file_name": f"f{i}.pdf"},
            "similarity": 0.9,
        } for i in range(10)]
        ctx = rag.build_context(items, max_length=500)
        assert len(ctx) < 2000  # 1件分は入るがそこで打ち切り


class TestExtractEvidence:
    def test_deduplicates_same_file_page(self):
        items = [
            {"text": "t1", "metadata": {"file_name": "a", "file_path": "/a", "page": 1}, "similarity": 0.9},
            {"text": "t2", "metadata": {"file_name": "a", "file_path": "/a", "page": 1}, "similarity": 0.8},
            {"text": "t3", "metadata": {"file_name": "a", "file_path": "/a", "page": 2}, "similarity": 0.7},
        ]
        ev = rag.extract_evidence(items)
        assert len(ev) == 2  # page=1 は重複排除

    def test_respects_max_items(self):
        items = [
            {"text": f"t{i}", "metadata": {"file_name": f"f{i}", "file_path": f"/f{i}", "page": i}, "similarity": 0.5}
            for i in range(20)
        ]
        ev = rag.extract_evidence(items, max_items=5)
        assert len(ev) == 5


class TestAnswerQuestion:
    def test_not_found_returns_honest_message(self, isolate_data_paths, ollama_model):
        """ADR-002: フォールバック廃止、見つからなければ正直に返す"""
        result = rag.answer_question(
            question="関連のない質問テスト",
            model_name=ollama_model,
            similarity_threshold=0.9,  # 高閾値で何もヒットしない
        )
        assert result["found"] is False
        assert "見つかりませんでした" in result["answer"]
        assert result["evidence"] == []

    @pytest.mark.timeout(120)
    def test_found_returns_answer_with_evidence(self, indexed_collection, ollama_model):
        """正常系: 関連文書ありで回答とエビデンスが返る"""
        result = rag.answer_question(
            question="モノシリはどんなAIですか？",
            model_name=ollama_model,
            similarity_threshold=0.3,
        )
        assert result["found"] is True
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
        assert len(result["evidence"]) >= 1
