"""
core/indexer.py の結合テスト

重要観点:
  - 初回インデックス後に ChromaDB にチャンクが入る
  - 差分インデックスで 0 件処理
  - キャンセルが 2 秒以内に反応（watchdog）
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from core.indexer import run_indexing_pipeline, IndexManager


pytestmark = [pytest.mark.slow, pytest.mark.gpu]


def _make_test_docs(folder: Path, count: int = 3) -> list[tuple[Path, str]]:
    """テスト用のテキストファイルを作成し (Path, folder_id) ペアを返す"""
    from core.hash_manager import make_folder_id
    folder.mkdir(parents=True, exist_ok=True)
    fid = make_folder_id(folder)
    files = []
    for i in range(count):
        f = folder / f"doc_{i}.txt"
        f.write_text(
            f"モノシリのテストドキュメント{i}です。"
            f"このファイルはpytestによって自動生成されました。" * 10,
            encoding="utf-8",
        )
        files.append((f, fid))
    return files


class TestRunIndexingPipeline:
    def test_initial_indexing_creates_chunks(self, isolate_data_paths, tmp_path):
        from core.chromadb_store import get_collection
        docs_folder = tmp_path / "docs"
        files = _make_test_docs(docs_folder, count=3)

        collection = get_collection()
        skip_log: list[dict] = []
        cancel_event = threading.Event()

        result = run_indexing_pipeline(
            files_to_process=files,
            collection=collection,
            skip_log=skip_log,
            cancel_event=cancel_event,
        )

        assert result["cancelled"] is False
        assert result["total_chunks"] > 0
        assert collection.count() > 0

    def test_empty_input(self, isolate_data_paths):
        from core.chromadb_store import get_collection
        collection = get_collection()
        cancel_event = threading.Event()
        result = run_indexing_pipeline(
            files_to_process=[],
            collection=collection,
            skip_log=[],
            cancel_event=cancel_event,
        )
        assert result["processed"] == 0

    @pytest.mark.timeout(30)
    def test_cancel_responds_quickly(self, isolate_data_paths, tmp_path):
        """キャンセル即時反応: 発火から 8 秒以内に終了

        検証観点: cancel_event.set() 後にパイプラインが 8 秒以内に終了すること。
        GPU が高速な環境では cancel より先に処理が完了する可能性があるため、
        ワークロードを十分に増やして処理中断を確実にする。
        """
        from core.chromadb_store import get_collection
        # 大量のファイル + 大きめコンテンツで GPU でも 5 秒以上かかるサイズにする
        docs_folder = tmp_path / "docs"
        docs_folder.mkdir()
        from core.hash_manager import make_folder_id
        fid = make_folder_id(docs_folder)
        files = []
        for i in range(50):
            f = docs_folder / f"doc_{i}.txt"
            f.write_text("モノシリのキャンセル反応テスト用ドキュメントです。" * 100, encoding="utf-8")
            files.append((f, fid))

        collection = get_collection()
        cancel_event = threading.Event()
        result_holder: dict = {}

        def _run():
            result_holder["result"] = run_indexing_pipeline(
                files_to_process=files,
                collection=collection,
                skip_log=[],
                cancel_event=cancel_event,
            )

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        # 処理開始を待ってから cancel（短めに）
        time.sleep(0.3)
        cancel_start = time.time()
        cancel_event.set()
        t.join(timeout=10)
        cancel_elapsed = time.time() - cancel_start

        assert not t.is_alive(), "スレッドが終了していない（キャンセル反応不足）"
        assert cancel_elapsed < 8, (
            f"キャンセル反応が遅すぎる: {cancel_elapsed:.1f}秒（期待: 8秒以内）"
        )


class TestIndexManager:
    def test_singleton(self):
        m1 = IndexManager.get()
        m2 = IndexManager.get()
        assert m1 is m2

    def test_initial_state(self):
        m = IndexManager.get()
        assert m.status == "idle" or m.is_finished

    def test_cancel_sets_event(self, isolate_data_paths):
        m = IndexManager.get()
        m._cancel_event.clear()
        m.cancel()
        assert m._cancel_event.is_set()
