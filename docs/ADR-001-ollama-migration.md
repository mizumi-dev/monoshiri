# ADR-001: LLM推論バックエンドをllama-cpp-pythonからOllamaに移行

**Status:** Accepted
**Date:** 2026-04-08
**Deciders:** Motoki（開発者）

## Context

モノシリ Phase 1 MVP では、ローカルLLM推論にllama-cpp-pythonを直接利用していた。
しかし、以下の課題が発生していた：

1. **GPU管理の複雑さ**: CUDA対応版llama-cpp-pythonのインストールにはCUDA Toolkitのバージョン合わせが必要で、非エンジニアのユーザーには困難
2. **初回ロードの遅さ**: GGUFモデルのメモリロードに数十秒かかり、初回質問時のUXが悪い
3. **チャットテンプレートの自前管理**: モデルごとのプロンプト形式（ChatML等）を自前で実装・メンテする必要がある
4. **プロセス管理**: Pythonプロセス内でLLMを直接ロードするため、メモリリークやクラッシュ時の影響範囲が大きい

## Decision

LLM推論バックエンドを **llama-cpp-python** から **Ollama HTTP API** に移行する。

- Ollama はローカルで動作するLLMサーバー（MIT License）
- HTTP API (`http://localhost:11434`) 経由でモデル管理・推論を行う
- start.bat でOllamaの自動検出・自動インストール・自動起動を行う

## Options Considered

### Option A: llama-cpp-python を維持（現状維持）

| Dimension | Assessment |
|-----------|------------|
| 複雑さ | High — CUDA版のインストール、チャットテンプレート管理 |
| コスト | Free |
| GPU対応 | 手動（CUDAホイール指定、n_gpu_layers調整） |
| ユーザー体験 | Poor — GPU設定が難解 |

**Pros:** 依存関係が少ない、Pythonプロセス内で完結
**Cons:** GPU設定が難解、チャットテンプレート自前管理、初回ロード遅い

### Option B: Ollama HTTP API に移行（採用）

| Dimension | Assessment |
|-----------|------------|
| 複雑さ | Low — HTTP APIのみ、GPU自動検出 |
| コスト | Free（MIT License、ローカル利用は永続無料） |
| GPU対応 | 自動（Ollama側でCUDA/Metal自動検出、VRAMオフロード） |
| ユーザー体験 | Good — ユーザーはGPU設定を意識しない |

**Pros:** GPU自動管理、チャットテンプレート内蔵、モデルプリウォーム対応、ストリーミング推論標準対応
**Cons:** 外部プロセス（Ollama）の管理が必要、初回インストールに管理者権限が必要な場合がある

### Option C: vLLM / llama.cpp Server

| Dimension | Assessment |
|-----------|------------|
| 複雑さ | Medium — サーバー設定が必要 |
| コスト | Free |
| GPU対応 | vLLMはCUDA必須、llama.cpp Serverは手動設定 |
| ユーザー体験 | Poor — サーバー起動が別途必要 |

**Pros:** 高スループット（vLLM）、柔軟な設定
**Cons:** セットアップが複雑、非エンジニアには不向き

## Trade-off Analysis

Ollama を選択した主な理由：

1. **ターゲットユーザーとの適合**: 製造業の非エンジニアユーザーがGPU設定を意識せず利用できる
2. **運用の簡素化**: start.bat一発で全環境が整う設計が可能
3. **セキュリティ**: 完全ローカル動作、外部通信なし（Ollamaはポート11434でlocalhost通信のみ）
4. **ライセンス**: MIT License、ローカル利用は完全無料（Ollama Cloud は別サービス）
5. **ストリーミング対応**: NDJSON形式のストリーミングAPIで、回答のリアルタイム表示が容易

## Consequences

### 容易になること
- GPU設定（Ollama側で自動検出・オフロード）
- モデルの追加・管理（Modelfile + `ollama create`）
- ストリーミング推論（HTTP APIの標準機能）
- モデルプリウォーム（起動時に最小リクエストで事前ロード）

### 困難になること
- Ollamaプロセスの生存管理（start.batで対応）
- 企業セキュリティポリシーによるインストール制限への対応
- Ollamaバージョンアップへの追従

### 今後見直しが必要なこと
- requirements.txt から llama-cpp-python を削除するタイミング
- Ollama がサポートしないモデル形式への対応方針
- Ollama Cloud との関係性の整理（利用しない旨の明示）

## Implementation Details

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `core/llm.py` | llama-cpp-python → Ollama HTTP API (urllib) に全面書き換え |
| `core/rag.py` | `stream_question()` 追加（ストリーミングRAG） |
| `core/config.py` | `OLLAMA_BASE_URL`追加、RAGパラメータ調整 |
| `ui/chat_page.py` | ストリーミングUI実装、重複キー修正 |
| `ui/settings_page.py` | GPU設定セクションをOllama対応に書き換え |
| `app.py` | モデルプリウォーム機能追加 |
| `start.bat` | Ollama自動検出・インストール・起動処理追加 |

### Ollama API エンドポイント

| Endpoint | Method | 用途 |
|----------|--------|------|
| `/api/version` | GET | ヘルスチェック |
| `/api/tags` | GET | インストール済みモデル一覧 |
| `/api/chat` | POST | 推論（stream: true/false） |

### モデル登録フロー

1. GGUFファイルの絶対パスを取得
2. Modelfile を生成（`FROM /path/to/model.gguf` + GPU/コンテキスト設定）
3. `ollama create [ollama_name] -f [modelfile_path]` で登録
4. `PARAMETER num_gpu 99` で全層GPUオフロード

## Action Items

1. [x] core/llm.py をOllama API対応に書き換え
2. [x] ストリーミング推論の実装（core/rag.py + ui/chat_page.py）
3. [x] start.bat にOllama自動管理を追加
4. [x] ui/settings_page.py のGPU設定セクション更新
5. [x] モデルプリウォーム機能の実装（app.py）
6. [ ] requirements.txt の更新（llama-cpp-python の扱い決定）
7. [ ] E2Eテストの実施
