# モノシリ 全モジュール + E2E テストレポート (d80605)

**実施日**: 2026-04-17
**総テスト数**: 131 件
**結果**: **全緑 (131 passed)**
**総実行時間**: 約 102 秒
**環境**: Windows 11, Python 3.14.4, CUDA 12.8 (RTX 4060 Ti), Ollama 0.20.3

---

## 1. サマリ

| Phase | 対象 | 件数 | 結果 | 時間 |
|-------|------|-----|------|------|
| A | core/ 単体 (7 モジュール) | 79 | ✅ 全緑 | 13 秒 |
| B | 結合 (indexer / llm / rag) | 21 | ✅ 全緑 | 90 秒 |
| C | ui_qt/ 結合 (splash / main_window / workers) | 20 | ✅ 全緑 | 15 秒 |
| D | E2E (T1〜T11) | 11 | ✅ 全緑 | 54 秒 |
| **合計** | | **131** | **✅ 全緑** | **102 秒** |

---

## 2. Phase 別詳細

### Phase A: core/ 単体テスト (79 passed)

| モジュール | テスト数 | 主要検証 |
|----------|--------|---------|
| `config.py` | 17 | 固定値 (`CHUNK_SIZE=1200`, `DEFAULT_SIMILARITY=0.40`, `TOP_K=8` 等)、LLM 5 モデル定義、ChatML テンプレート、load/save |
| `hash_manager.py` | 17 | SHA-256 正確性、軽量ハッシュ、manifest I/O、差分検知（新規/変更/削除） |
| `extractor.py` | 13 | TXT 抽出、未対応形式、破損PDF、scan_folder (フィルタ・隠しファイル除外)、split_text |
| `chromadb_store.py` | 6 | シングルトン、コレクション、count、全削除 |
| `embedder.py` | 10 | `detect_backend() = CUDA`、VRAM自動調整、空リスト、**CancelledError スコープバグ回帰**、実 Embedding、正規化ベクトル |
| `checkpoint_manager.py` | 7 | チャンク処理管理、不完全ファイル管理、全削除、特定ファイル削除 |
| `usage_tracker.py` | 8 | 月間質問カウント、インデックスファイル数、サマリ構造 |

### Phase B: 結合テスト (21 passed)

| モジュール | テスト数 | 主要検証 |
|----------|--------|---------|
| `indexer.py` | 6 | 初回インデックス、空入力、**キャンセル 8 秒以内反応 (watchdog 検証)**、IndexManager シングルトン |
| `llm.py` | 6 | Ollama API 疎通、モデル一覧、推奨モデル、**実 LLM 回答生成**、warmup |
| `rag.py` | 9 | ベクトル検索、閾値フィルタ、コンテキスト構築、エビデンス重複排除、**ADR-002 フォールバック廃止（見つかりません応答）**、正常系 (evidence ≥ 1) |

### Phase C: ui_qt/ 結合テスト (20 passed)

| 対象 | テスト数 | 主要検証 |
|-----|--------|---------|
| `splash.py` | 4 | **生成 200ms 以内**、set_status 更新、スレッドセーフ版、経過時間タイマ |
| `main_window.py` | 7 | ウィンドウ生成、3 ナビボタン、デフォルトタブ、切替、マウスクリック、Ollama 表示 |
| `workers.py` | 9 | 9 種ワーカー全てのコールバック発火（Stats / Ollama / IndexStats / InstalledModels / RecommendedModels / MemInfo / **CudaInfo (CUDA検出)** / OllamaCheck / NetworkCheck） |

### Phase D: E2E シナリオ (11 passed)

| ID | シナリオ | 結果 |
|----|--------|------|
| T1 | スプラッシュ生成 | ✅ |
| T2 | 進捗バー 5→100 更新 | ✅ |
| T3 | MainWindow 生成・表示 | ✅ |
| T4 | CUDA バックエンド検出 | ✅ |
| T5 | 3 ファイル初回インデックス | ✅ (total_chunks > 0) |
| T6 | 差分インデックス (再実行で 0 件) | ✅ |
| T7 | キャンセル即時反応 (8 秒以内) | ✅ |
| T8 | RAG 正常系 (evidence ≥ 1) | ✅ |
| T9 | ADR-002 ハルシネーション抑制 | ✅ (「見つかりませんでした」) |
| T10 | 履歴保存/読込 ラウンドトリップ | ✅ |
| T11 | ウィンドウクローズでタイマー停止 | ✅ |

---

## 3. 修正履歴（テスト作成中に発見・修正した問題）

### 3.1 実装側修正（アップストリーム）

| # | 症状 | 原因 | 修正 |
|---|------|------|------|
| 1 | テスト環境が `~/.monoshiri` を汚染 | データディレクトリがハードコード | **`MONOSHIRI_DATA_DIR` 環境変数サポート追加** (`core/config.py`, `app_qt.py`, `ui_qt/chat_widget.py`) |

### 3.2 テスト側修正（仕様通りに実装へ合わせた）

| # | 症状 | 対応 |
|---|------|------|
| 1 | ChromaDB が空メタデータを拒否 | テストの metadata を `{"src": "test"}` に修正 |
| 2 | Windows で Path オブジェクトが文字列化時に不一致 | テストで `tmp_path` ベースの一貫したキーに統一 |
| 3 | ChromaDB SharedSystemClient のテスト間汚染 | conftest.py で `SharedSystemClient.clear_system_cache()` を各テスト前後に実行 |
| 4 | T6 が軽量ハッシュで比較 → compute_hash で保存すると不整合 | テストで `compute_lightweight_hash` を使用 |
| 5 | キャンセルテスト Flaky（GPU 高速時に処理完了） | ワークロード 50 ファイル × 100 回反復に増強 |

### 3.3 発見事項（要ユーザー判断）

#### 🟡 スプラッシュ起動時間：仕様 vs 実測の不整合

**計測結果（実 `app_qt.py` 相当）**:
- スプラッシュ生成 (`SplashScreen()` コンストラクタ): **37ms**
- `splash.show()` + `app.processEvents()`: **約 1000ms** (Windows 描画コスト)
- QApplication 起動含む合計: **約 1500ms**

**仕様書 v4.0 の要件**: 「起動 1 秒以内にスプラッシュ表示」

**原因**: Windows 上の `QSplashScreen.show()` 自体に 1 秒程度かかる（`WindowStaysOnTopHint` フラグの有無に関係なく同程度）。これは Qt/Windows の既定描画パスのコスト。

**提案 (3 択)**:
1. **(A) 仕様書を「起動 2 秒以内」に緩和** ← 推奨（実装面から見て現実的、体感良好）
2. **(B) スプラッシュ省略 + 簡易ローディングダイアログ**（挙動変更あり）
3. **(C) 現状維持**（仕様との 0.5 秒乖離を許容）

ご判断をお願いします。

---

## 4. 発見されなかった既知問題

- Windows での `chmod` 失敗（config.py ではすでに OSError を捕捉済）
- ChromaDB の `embeddings.position_ids UNEXPECTED` 警告（モデル間互換性による、動作影響なし）

---

## 5. ファイル構成

```
tests/
├── conftest.py                    # 共通フィクスチャ（MONOSHIRI_DATA_DIR 隔離）
├── __init__.py
├── unit/
│   ├── test_config.py             (17)
│   ├── test_hash_manager.py       (17)
│   ├── test_extractor.py          (13)
│   ├── test_chromadb_store.py     (6)
│   ├── test_embedder.py           (10)
│   ├── test_checkpoint_manager.py (7)
│   └── test_usage_tracker.py      (8)
├── integration/
│   ├── conftest.py                # ollama_model フィクスチャ（5 モデル優先選択）
│   ├── test_indexer.py            (6)
│   ├── test_llm.py                (6)
│   └── test_rag.py                (9)
├── ui/
│   ├── test_splash.py             (4)
│   ├── test_main_window.py        (7)
│   └── test_workers.py            (9)
└── e2e/
    ├── conftest.py
    └── test_full_flow.py          (11)

pytest.ini                          # マーカー定義（ollama/gpu/slow/ui/e2e）
```

## 6. 実行コマンド

```powershell
# 全テスト
.venv\Scripts\python.exe -m pytest tests -v

# Phase 別
.venv\Scripts\python.exe -m pytest tests\unit -v
.venv\Scripts\python.exe -m pytest tests\integration -v
.venv\Scripts\python.exe -m pytest tests\ui -v
.venv\Scripts\python.exe -m pytest tests\e2e -v

# 高速のみ（Ollama/GPU 不要）
.venv\Scripts\python.exe -m pytest tests -v -m "not ollama and not gpu and not slow"
```

Ollama 未起動時は該当テストが自動 skip されます。

---

## 7. 追加インストールされた依存

```
pytest==9.0.3
pytest-qt==4.5.0
pytest-timeout==2.4.0
```

`requirements.txt` への追加は別途ご判断ください（テスト専用のため）。

---

## 8. ADR / 仕様書 整合性確認

| ADR | 確認項目 | 結果 |
|-----|---------|------|
| ADR-001 | Ollama HTTP API 経由のみ | ✅ `llm.py` テストで確認 |
| ADR-002 | フォールバック廃止、見つからなければ正直に返す | ✅ T9 / rag テストで確認 |
| ADR-002 | `DEFAULT_SIMILARITY=0.40` | ✅ `test_config.py` で確認 |
| 先日修正 | `embedder.CancelledError` スコープ | ✅ 回帰テスト追加 |
| 先日修正 | インデックスキャンセル watchdog | ✅ T7 / integration で確認 |
| 先日修正 | CUDA バックエンド自動検出 | ✅ T4 / embedder テストで確認 |
