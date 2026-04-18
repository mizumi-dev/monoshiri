"""
モノシリ チャット精度テストスクリプト

financial_reports フォルダのPDFをインデックスし、
テスト質問に対するRAG回答を生成・評価する。
"""
from __future__ import annotations
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# プロジェクトルートをPythonパスに追加
APP_DIR = Path(__file__).parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from core.config import (
    load_config,
    DATA_DIR,
    CHROMA_DIR,
    LLM_MODELS,
    DEFAULT_SIMILARITY,
)
from core.chromadb_store import get_collection, delete_all_chunks
from core.indexer import scan_and_diff, run_indexing_pipeline
from core.hash_manager import make_folder_id
from core.rag import answer_question


# ─── テスト質問定義 ────────────────────────────────────────────────

TEST_QUESTIONS = {
    "fact_checking": [
        {
            "id": "F1",
            "question": "JR東日本の2021年度の売上高は？",
            "expected_keywords": ["売上高", "2021", "年度", "円"],
            "expected_file": "JR東日本_決算短信_FY2021.pdf",
        },
        {
            "id": "F2",
            "question": "Alphabetの2023年の営業利益は？",
            "expected_keywords": ["営業利益", "2023", "Alphabet", "million", "billion"],
            "expected_file": "Alphabet_2023-02-03.pdf",
        },
        {
            "id": "F3",
            "question": "2022年度のJR東日本の決算短信に記載されている主要な収益源は？",
            "expected_keywords": ["収益", "運輸", "事業", "2022"],
            "expected_file": "JR東日本_決算短信_FY2022.pdf",
        },
        {
            "id": "F4",
            "question": "Alphabetの2025年2月期の決算で最も成長した事業セグメントは？",
            "expected_keywords": ["成長", "セグメント", "事業", "2025"],
            "expected_file": "Alphabet_2025-02-05.pdf",
        },
        {
            "id": "F5",
            "question": "JR東日本の2023年度の純利益は前年度比でどう変化したか？",
            "expected_keywords": ["純利益", "前年度", "変化", "2023"],
            "expected_file": "JR東日本_決算短信_FY2023.pdf",
        },
    ],
    "general_exploration": [
        {
            "id": "G1",
            "question": "この資料で一番よく出てくるキーワードは？",
            "expected_keywords": ["キーワード", "頻出", "多い"],
            "expected_file": None,  # 複数ファイル
        },
        {
            "id": "G2",
            "question": "売上高が最も高い企業は？",
            "expected_keywords": ["売上高", "高い", "企業"],
            "expected_file": None,  # 比較質問
        },
        {
            "id": "G3",
            "question": "最近の決算で業績が改善した企業は？",
            "expected_keywords": ["業績", "改善", "決算"],
            "expected_file": None,  # 比較質問
        },
        {
            "id": "G4",
            "question": "赤字を出している企業はあるか？",
            "expected_keywords": ["赤字", "企業"],
            "expected_file": None,  # 比較質問
        },
        {
            "id": "G5",
            "question": "各企業の主要な事業内容を教えて",
            "expected_keywords": ["事業", "内容", "企業"],
            "expected_file": None,  # 総合質問
        },
    ],
}


# ─── ログ設定 ───────────────────────────────────────────────────────

def setup_logging():
    LOG_DIR = DATA_DIR / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "chat_accuracy_test.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ─── インデックス処理 ───────────────────────────────────────────────

def index_financial_reports():
    """financial_reports フォルダをインデックスする"""
    logger.info("=== financial_reports インデックス開始 ===")
    
    financial_dir = Path(__file__).parent / "financial_reports"
    if not financial_dir.exists():
        logger.error(f"financial_reports ディレクトリが存在しません: {financial_dir}")
        return False
    
    # 既存のコレクションをクリア
    logger.info("既存コレクションをクリア...")
    delete_all_chunks()
    
    # フォルダをスキャン
    logger.info(f"フォルダスキャン: {financial_dir}")
    files_to_process, skip_log = scan_and_diff([financial_dir])
    
    logger.info(f"処理対象ファイル数: {len(files_to_process)}")
    logger.info(f"スキップファイル数: {len(skip_log)}")
    
    if len(files_to_process) == 0:
        logger.warning("処理対象ファイルがありません")
        return False
    
    # インデックス実行
    logger.info("インデックス実行開始...")
    collection = get_collection()
    
    from threading import Event
    cancel_event = Event()
    
    start_time = time.time()
    result = run_indexing_pipeline(
        files_to_process=files_to_process,
        collection=collection,
        skip_log=skip_log,
        cancel_event=cancel_event,
    )
    elapsed = time.time() - start_time
    
    logger.info(f"インデックス完了: {elapsed:.1f}秒")
    logger.info(f"  - 処理ファイル数: {result['processed']}")
    logger.info(f"  - チャンク数: {result['total_chunks']}")
    logger.info(f"  - キャンセル: {result['cancelled']}")
    
    return result['total_chunks'] > 0


# ─── 質問実行 ───────────────────────────────────────────────────────

def run_question_test(question_data: dict, model_name: str, similarity_threshold: float):
    """単一質問をテストする"""
    qid = question_data["id"]
    question = question_data["question"]
    expected_keywords = question_data.get("expected_keywords", [])
    expected_file = question_data.get("expected_file")
    
    logger.info(f"\n--- 質問 {qid}: {question} ---")
    
    start_time = time.time()
    result = answer_question(
        question=question,
        model_name=model_name,
        similarity_threshold=similarity_threshold,
    )
    elapsed = time.time() - start_time
    
    logger.info(f"回答生成時間: {elapsed:.1f}秒")
    logger.info(f"found: {result['found']}")
    logger.info(f"evidence数: {len(result['evidence'])}")
    
    # エビデンス情報をログ
    for i, ev in enumerate(result['evidence'][:3]):  # 最初の3件のみ表示
        logger.info(f"  Evidence {i+1}: {ev['file_name']} (similarity: {ev['similarity']:.3f})")
    
    # 回答内容をログ（最初の200文字）
    answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
    logger.info(f"回答プレビュー: {answer_preview}")
    
    return {
        "question": question,
        "answer": result['answer'],
        "found": result['found'],
        "evidence": result['evidence'],
        "elapsed_seconds": elapsed,
        "expected_keywords": expected_keywords,
        "expected_file": expected_file,
    }


# ─── 結果評価 ───────────────────────────────────────────────────────

def evaluate_result(result: dict):
    """テスト結果を評価する"""
    answer = result['answer']
    evidence = result['evidence']
    expected_keywords = result['expected_keywords']
    expected_file = result['expected_file']
    
    evaluation = {
        "has_evidence": len(evidence) > 0,
        "evidence_count": len(evidence),
        "found": result['found'],
        "keyword_matches": [],
        "file_match": False,
        "overall_score": 0,
    }
    
    # キーワードマッチング
    for keyword in expected_keywords:
        if keyword.lower() in answer.lower():
            evaluation['keyword_matches'].append(keyword)
    
    # ファイルマッチング
    if expected_file:
        for ev in evidence:
            if expected_file in ev.get('file_name', ''):
                evaluation['file_match'] = True
                break
    else:
        # 期待ファイルが指定されていない場合は、エビデンスがあればOK
        evaluation['file_match'] = len(evidence) > 0
    
    # 総合スコア計算（0-100）
    score = 0
    if evaluation['found']:
        score += 30
    if evaluation['has_evidence']:
        score += 30
    if len(evaluation['keyword_matches']) > 0:
        score += 20
    if evaluation['file_match']:
        score += 20
    
    evaluation['overall_score'] = score
    
    return evaluation


# ─── レポート生成 ───────────────────────────────────────────────────

def generate_report(all_results: dict, output_file: str):
    """テスト結果レポートを生成する"""
    lines = []
    lines.append("# モノシリ チャット精度テストレポート")
    lines.append(f"\n**実施日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n---\n")
    
    # サマリ
    total_tests = sum(len(results) for results in all_results.values())
    passed = sum(1 for results in all_results.values() for r in results if r['evaluation']['overall_score'] >= 60)
    
    lines.append("## サマリ")
    lines.append(f"- **総テスト数**: {total_tests}")
    lines.append(f"- **合格数** (スコア≥60): {passed}")
    lines.append(f"- **合格率**: {passed/total_tests*100:.1f}%")
    lines.append("\n---\n")
    
    # カテゴリ別結果
    for category, results in all_results.items():
        lines.append(f"## {category}")
        lines.append("")
        
        for r in results:
            qid = r['result']['question'][:30] + "..."
            score = r['evaluation']['overall_score']
            status = "✅" if score >= 60 else "❌"
            
            lines.append(f"### {status} {qid}")
            lines.append(f"**スコア**: {score}/100")
            lines.append(f"**質問**: {r['result']['question']}")
            lines.append(f"**回答時間**: {r['result']['elapsed_seconds']:.1f}秒")
            lines.append(f"**エビデンス数**: {r['evaluation']['evidence_count']}")
            lines.append(f"**キーワードマッチ**: {', '.join(r['evaluation']['keyword_matches']) or 'なし'}")
            lines.append(f"**ファイルマッチ**: {'✅' if r['evaluation']['file_match'] else '❌'}")
            lines.append("")
            lines.append("**回答**:")
            lines.append(f"```\n{r['result']['answer']}\n```")
            lines.append("")
            
            if r['result']['evidence']:
                lines.append("**エビデンス**:")
                for ev in r['result']['evidence'][:5]:
                    lines.append(f"- {ev['file_name']} (類似度: {ev['similarity']:.3f})")
                lines.append("")
            
            lines.append("---")
            lines.append("")
    
    # レポート保存
    report_text = "\n".join(lines)
    Path(output_file).write_text(report_text, encoding="utf-8")
    logger.info(f"レポート保存: {output_file}")
    
    return report_text


# ─── メイン処理 ─────────────────────────────────────────────────────

def main():
    logger.info("=== モノシリ チャット精度テスト開始 ===")
    
    # 設定読み込み
    config = load_config()
    model_name = config.get("selected_model", "")
    
    if not model_name:
        # モデルが選択されていない場合は推奨モデルを使用
        model_name = "Qwen2.5-7B-Instruct Q4_K_M（標準・推奨）"
        logger.warning(f"モデル未設定のため推奨モデルを使用: {model_name}")
    
    # Ollamaモデル名を取得
    ollama_model = LLM_MODELS.get(model_name, {}).get("ollama_name", model_name)
    logger.info(f"使用モデル: {model_name} (Ollama名: {ollama_model})")
    
    # インデックス実行
    if not index_financial_reports():
        logger.error("インデックスに失敗しました")
        return
    
    # 質問テスト実行
    all_results = {}
    similarity_threshold = DEFAULT_SIMILARITY
    
    for category, questions in TEST_QUESTIONS.items():
        logger.info(f"\n=== {category} テスト開始 ===")
        category_results = []
        
        for q_data in questions:
            try:
                result = run_question_test(q_data, ollama_model, similarity_threshold)
                evaluation = evaluate_result(result)
                
                category_results.append({
                    "result": result,
                    "evaluation": evaluation,
                })
                
                # 結果を一時保存
                time.sleep(1)  # Ollama負荷軽減
            except Exception as e:
                logger.error(f"質問 {q_data['id']} でエラー: {e}")
                category_results.append({
                    "result": {
                        "question": q_data["question"],
                        "answer": f"エラー: {str(e)}",
                        "found": False,
                        "evidence": [],
                        "elapsed_seconds": 0,
                        "expected_keywords": q_data.get("expected_keywords", []),
                        "expected_file": q_data.get("expected_file"),
                    },
                    "evaluation": {
                        "has_evidence": False,
                        "evidence_count": 0,
                        "found": False,
                        "keyword_matches": [],
                        "file_match": False,
                        "overall_score": 0,
                    },
                })
        
        all_results[category] = category_results
    
    # レポート生成
    report_file = Path(__file__).parent / "CHAT_ACCURACY_TEST_REPORT.md"
    generate_report(all_results, str(report_file))
    
    # JSON結果も保存
    json_file = Path(__file__).parent / "CHAT_ACCURACY_TEST_RESULTS.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON結果保存: {json_file}")
    
    logger.info("=== テスト完了 ===")


if __name__ == "__main__":
    main()
