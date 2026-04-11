"""
モノシリ ChromaDB接続モジュール

ベクトルDBへの接続・コレクション管理を担当する共有インフラ。
indexer（インデックス作成）と rag（検索）の両方から利用される。

疎結合設計:
  - indexer.py はこのモジュールを通じてのみ ChromaDB を操作する
  - rag.py はこのモジュールを通じてのみ ChromaDB を操作する
  - ChromaDB の設定変更はここ1箇所で完結する
"""
from __future__ import annotations
import logging

from core.config import CHROMA_DIR, CHROMA_COLLECTION

logger = logging.getLogger(__name__)


def get_chroma_client():
    """ChromaDBクライアントを取得する"""
    import chromadb
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    """ChromaDBコレクションを取得または作成する"""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def get_index_stats() -> dict:
    """インデックスの統計情報を返す"""
    try:
        collection = get_collection()
        return {"total_chunks": collection.count()}
    except Exception:
        return {"total_chunks": 0}
