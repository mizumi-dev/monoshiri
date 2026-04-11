"""
モノシリ ChromaDB接続モジュール

TODO(仕様書更新): v3.4仕様書はベクトルDBとしてFAISを記載しているが、
  実装はChromaDBを使用している。仕様書の「技術構成」セクションをChromaDBに更新すること。
TODO(セキュリティ): v3.4仕様書のAES-256-GCM暗号化（ベクトルDB）は未実装。
  Phase 2以降で対応予定。

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

# PERF修正: PersistentClient をモジュールレベルでシングルトン管理。
# 毎回新規生成するとDBロック・メモリリークの原因になる。
_chroma_client_instance = None


def get_chroma_client():
    """ChromaDBクライアントを取得する（シングルトン）"""
    global _chroma_client_instance
    if _chroma_client_instance is None:
        import chromadb
        _chroma_client_instance = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client_instance


def reset_chroma_client() -> None:
    """インデックス再作成後などクライアントをリセットする際に呼ぶ"""
    global _chroma_client_instance
    _chroma_client_instance = None


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


def delete_chunks_by_file(file_path: Path) -> int:
    """
    特定ファイルに紐づく全チャンクをChromaDBから削除する。
    部分的に処理されたファイルの再処理時に使用。

    Returns:
        削除されたチャンク数
    """
    try:
        collection = get_collection()
        file_path_str = str(file_path)

        # where条件でmetadata.source_fileを検索
        # 注意: ChromaDBのwhereは厳密マッチのみなので、IDプレフィックスで検索する方が確実
        # チャンクIDは "{file_path}_chunk_{index}" の形式
        prefix = f"{file_path_str}_chunk_"

        # 全データを取得してフィルタ（小規模データ向け）
        # 大規模データの場合はページネーションが必要
        all_data = collection.get()
        if not all_data or not all_data.get("ids"):
            return 0

        ids_to_delete = [
            cid for cid in all_data["ids"]
            if cid.startswith(prefix)
        ]

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            logger.info(f"[ChromaDB] {len(ids_to_delete)}チャンク削除: {file_path.name}")
            return len(ids_to_delete)

        return 0

    except Exception as e:
        logger.error(f"[ChromaDB] チャンク削除失敗 {file_path}: {e}")
        return 0


def delete_all_chunks() -> int:
    """全チャンクを削除（コレクション再作成）"""
    try:
        client = get_chroma_client()
        try:
            client.delete_collection(name=CHROMA_COLLECTION)
            logger.info("[ChromaDB] コレクション削除完了")
        except Exception:
            pass  # 存在しない場合は無視
        # シングルトンをリセットして再作成
        reset_chroma_client()
        get_collection()  # 新規作成
        return 1
    except Exception as e:
        logger.error(f"[ChromaDB] 全削除失敗: {e}")
        return 0
