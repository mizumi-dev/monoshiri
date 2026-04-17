"""
モノシリ 最終 E2E テスト（T1〜T11）

実アプリと同じ起動シーケンスを pytest-qt で再現し、主要シナリオを自動検証する。

T1: QApplication 起動 + スプラッシュ表示
T2: 重いロードの進捗バー遷移
T3: スプラッシュ → メインウィンドウ遷移
T4: CUDA バックエンド検出
T5: 初回インデックス（テスト用 3 ファイル）
T6: 差分インデックス（0 件処理）
T7: キャンセル即時反応
T8: RAG 正常系（evidence ≥ 1）
T9: RAG ハルシネーション抑制（ADR-002）
T10: 履歴保存
T11: クリーン終了
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


# ─── T1〜T3: 起動フロー ──────────────────────────────────────────────
class TestStartupFlow:
    def test_T1_splash_creates_successfully(self, qtbot):
        """T1: QApplication 起動直後にスプラッシュが生成できる

        タイミング検証は test_splash.py::test_construction_fast で実施済。
        ここではオブジェクトの構造健全性のみ検証。
        """
        from ui_qt.splash import SplashScreen
        splash = SplashScreen()
        qtbot.addWidget(splash)
        # 重要な子ウィジェットが存在
        assert splash._status_label is not None
        assert splash._progress is not None
        assert splash._elapsed_label is not None
        assert splash._progress.minimum() == 0
        assert splash._progress.maximum() == 100

    def test_T2_progress_bar_updates(self, qtbot):
        """T2: 進捗バーが 5→100 と更新できる"""
        from ui_qt.splash import SplashScreen
        splash = SplashScreen()
        qtbot.addWidget(splash)
        for msg, pct in [("起動中", 5), ("設定読み込み", 20), ("DB初期化", 35),
                          ("Embedding", 60), ("UI構築", 98), ("準備完了", 100)]:
            splash.set_status(msg, pct)
            assert splash._progress.value() == pct

    def test_T3_main_window_appears_after_load(self, qtbot):
        """T3: ロード完了 → MainWindow 生成"""
        from ui_qt.main_window import MainWindow
        config = {"selected_model": "", "folders": [], "plan": "free"}
        window = MainWindow(config=config)
        qtbot.addWidget(window)
        window.show()
        assert window.isVisible() or window.isHidden() is False


# ─── T4: CUDA 確認 ──────────────────────────────────────────────────
class TestGPUDetection:
    @pytest.mark.gpu
    def test_T4_cuda_backend_detected(self):
        """T4: detect_backend() が CUDA を返す"""
        from core.embedder import detect_backend
        device, backend = detect_backend()
        assert backend == "CUDA"
        assert device == "cuda"


# ─── T5〜T7: インデックスフロー ─────────────────────────────────────
class TestIndexFlow:
    @pytest.mark.gpu
    def test_T5_initial_indexing(self, isolate_data_paths, tmp_path):
        """T5: 3 ファイル初回インデックス"""
        from core.chromadb_store import get_collection
        from core.indexer import run_indexing_pipeline
        from core.hash_manager import make_folder_id

        folder = tmp_path / "docs_t5"
        folder.mkdir()
        fid = make_folder_id(folder)
        files = []
        for i in range(3):
            f = folder / f"doc_{i}.txt"
            f.write_text(f"モノシリテストドキュメント{i}です。" * 20, encoding="utf-8")
            files.append((f, fid))

        cancel = threading.Event()
        result = run_indexing_pipeline(
            files_to_process=files,
            collection=get_collection(),
            skip_log=[],
            cancel_event=cancel,
        )
        assert result["cancelled"] is False
        assert result["total_chunks"] > 0
        assert get_collection().count() >= 3

    def test_T6_differential_indexing(self, isolate_data_paths, tmp_path):
        """T6: 再実行で差分検知が働き、追加処理がゼロ"""
        from core.indexer import scan_and_diff
        from core.hash_manager import make_folder_id, save_hashes, compute_lightweight_hash

        folder = tmp_path / "docs_t6"
        folder.mkdir()
        for i in range(3):
            (folder / f"doc_{i}.txt").write_text(f"テスト{i}" * 50, encoding="utf-8")

        # 1 回目: 差分 = 全件
        files_to_process, _ = scan_and_diff([folder])
        assert len(files_to_process) == 3

        # scan_and_diff は軽量ハッシュ（size+mtime）で比較するので、同じ方式で保存
        fid = make_folder_id(folder)
        hashes = {str(f): compute_lightweight_hash(f) for f in folder.rglob("*") if f.is_file()}
        save_hashes(fid, hashes)

        # 2 回目: 差分 0
        files_to_process_2, _ = scan_and_diff([folder])
        assert len(files_to_process_2) == 0

    @pytest.mark.gpu
    @pytest.mark.timeout(30)
    def test_T7_cancel_responds_quickly(self, isolate_data_paths, tmp_path):
        """T7: キャンセル即時反応 (8 秒以内)"""
        from core.chromadb_store import get_collection
        from core.indexer import run_indexing_pipeline
        from core.hash_manager import make_folder_id

        folder = tmp_path / "docs_t7"
        folder.mkdir()
        fid = make_folder_id(folder)
        files = [(folder / f"d_{i}.txt", fid) for i in range(30)]
        for f, _ in files:
            f.write_text("A" * 3000, encoding="utf-8")

        cancel = threading.Event()
        result_holder = {}

        def _run():
            result_holder["r"] = run_indexing_pipeline(
                files_to_process=files,
                collection=get_collection(),
                skip_log=[],
                cancel_event=cancel,
            )

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(1.0)
        t_cancel = time.time()
        cancel.set()
        t.join(timeout=10)
        elapsed = time.time() - t_cancel

        assert not t.is_alive(), "キャンセル後もスレッド実行中"
        assert elapsed < 8, f"キャンセル反応 {elapsed:.1f}秒（期待: 8秒以内）"


# ─── T8〜T9: RAG ────────────────────────────────────────────────────
class TestRAG:
    @pytest.fixture
    def indexed(self, isolate_data_paths):
        """テスト用コンテンツを ChromaDB に登録"""
        from core.chromadb_store import get_collection
        from core.embedder import embed_texts

        texts = [
            "モノシリは完全ローカル動作の社内資料探索AIです。資料は一切外に出ません。",
            "ChromaDB をベクトルDBとして使用し、HNSWlib 基盤で高速検索を実現。",
            "Ollama を介してローカル LLM で推論し、外部通信ゼロで動作します。",
        ]
        embeddings = embed_texts(texts)
        collection = get_collection()
        collection.add(
            ids=[f"e2e_chunk_{i}" for i in range(len(texts))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"file_name": "intro.txt", "file_path": "/e2e/intro.txt", "folder_path": "/e2e", "page": 1}
                for _ in texts
            ],
        )
        return collection

    @pytest.mark.ollama
    @pytest.mark.gpu
    @pytest.mark.timeout(120)
    def test_T8_rag_normal_case_has_evidence(self, indexed, ollama_model):
        """T8: 関連質問 → 回答 + evidence ≥ 1"""
        from core.rag import answer_question
        result = answer_question(
            question="モノシリはどんなAIですか？",
            model_name=ollama_model,
            similarity_threshold=0.3,
        )
        assert result["found"] is True
        assert len(result["evidence"]) >= 1
        assert len(result["answer"]) > 0

    @pytest.mark.ollama
    @pytest.mark.gpu
    def test_T9_rag_hallucination_suppression(self, indexed, ollama_model):
        """T9: ADR-002 フォールバック廃止、無関係質問は「見つかりません」"""
        from core.rag import answer_question
        result = answer_question(
            question="量子コンピュータの暗号解読について",
            model_name=ollama_model,
            similarity_threshold=0.85,  # 高閾値で意図的に棄却
        )
        assert result["found"] is False
        assert "見つかりませんでした" in result["answer"]
        assert result["evidence"] == []


# ─── T10: 履歴 ───────────────────────────────────────────────────────
class TestHistory:
    def test_T10_history_file_write_read(self, isolate_data_paths):
        """T10: 履歴保存と読み込みのラウンドトリップ"""
        from core.config import HISTORY_FILE

        history = [
            {
                "timestamp": "2026-04-17T10:00:00",
                "question": "テスト質問",
                "answer": "テスト回答",
                "evidence": [],
            }
        ]
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")

        loaded = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        assert len(loaded) == 1
        assert loaded[0]["question"] == "テスト質問"


# ─── T11: クリーン終了 ──────────────────────────────────────────────
class TestCleanExit:
    def test_T11_main_window_close_stops_timers(self, qtbot):
        """T11: ウィンドウ closeEvent で全タイマー停止"""
        from ui_qt.main_window import MainWindow
        config = {"selected_model": "", "folders": [], "plan": "free"}
        w = MainWindow(config=config)
        qtbot.addWidget(w)
        w.show()
        assert w._stats_timer.isActive()
        assert w._ollama_timer.isActive()
        w.close()
        assert not w._stats_timer.isActive()
        assert not w._ollama_timer.isActive()
