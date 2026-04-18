"""
モノシリ チャット精度テスト（UI操作版）

Qtアプリを実際に起動し、UI上の質問欄に直接入力してテストを行う。
"""
from __future__ import annotations
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ui, pytest.mark.slow, pytest.mark.ollama]


# ─── テスト質問定義 ────────────────────────────────────────────────

TEST_QUESTIONS = {
    "資料関連": [
        {
            "id": "A1",
            "question": "JR東日本の2021年度の売上高は？",
            "expected_keywords": ["売上高", "2021", "年度", "円"],
            "expected_behavior": "回答あり",
        },
        {
            "id": "A2",
            "question": "トヨタ自動車の2023年度の決算で最も利益が出た事業は？",
            "expected_keywords": ["トヨタ", "2023", "利益", "事業"],
            "expected_behavior": "回答あり",
        },
        {
            "id": "A3",
            "question": "ソフトバンクグループの2025年度の売上高は？",
            "expected_keywords": ["ソフトバンク", "2025", "売上高"],
            "expected_behavior": "回答あり",
        },
    ],
    "資料外": [
        {
            "id": "B1",
            "question": "量子コンピュータの現在の技術動向は？",
            "expected_keywords": [],
            "expected_behavior": "見つかりません",
        },
        {
            "id": "B2",
            "question": "人工知能の倫理規定について教えて",
            "expected_keywords": [],
            "expected_behavior": "見つかりません",
        },
        {
            "id": "B3",
            "question": "気候変動による農業への影響は？",
            "expected_keywords": [],
            "expected_behavior": "見つかりません",
        },
    ],
    "全く無関係": [
        {
            "id": "C1",
            "question": "今日の天気は？",
            "expected_keywords": [],
            "expected_behavior": "見つかりません",
        },
        {
            "id": "C2",
            "question": "今週の野球の試合結果は？",
            "expected_keywords": [],
            "expected_behavior": "見つかりません",
        },
        {
            "id": "C3",
            "question": "おすすめのレストランは？",
            "expected_keywords": [],
            "expected_behavior": "見つかりません",
        },
    ],
}


# ─── テストクラス ───────────────────────────────────────────────────

@pytest.fixture
def config():
    return {
        "selected_model": "Qwen2.5-7B-Instruct Q4_K_M（標準・推奨）",
        "folders": [],
        "similarity_threshold": 0.40,
        "plan": "free",
        "llm_n_gpu_layers": 0,
        "embedding_device": "auto",
    }


class TestChatAccuracy:
    @pytest.mark.timeout(900)
    def test_chat_accuracy_ui(self, qtbot, config, ollama_model, use_production_data):
        """
        Qtアプリを起動し、UI上で質問を入力して回答を検証する。
        
        手順:
        1. MainWindowを起動
        2. インデックス管理タブでfinancial_reportsをインデックス
        3. チャットタブに切り替え
        4. 各質問を入力・送信
        5. 回答を記録・評価
        """
        from ui_qt.main_window import MainWindow
        from ui_qt.chat_widget import ChatWidget
        
        # MainWindow起動
        window = MainWindow(config=config)
        qtbot.addWidget(window)
        window.show()
        qtbot.wait(1000)  # UI初期化待機
        
        # ChatWidgetを取得
        chat_widget = window._chat_widget
        
        # 本番データの既存インデックスを使用（事前にインデックス済みであることを前提）
        from core.chromadb_store import get_collection
        count = get_collection().count()
        logging.getLogger(__name__).info(f"ChromaDB チャンク数: {count}")
        if count == 0:
            pytest.skip("本番 ChromaDB にインデックスがありません。事前にアプリでインデックスを実行してください。")
        
        # チャットタブに切り替え
        window._switch_page(0)  # チャットタブ
        qtbot.wait(500)
        
        # テスト結果記録
        all_results = {}
        
        # 各質問を実行
        for category, questions in TEST_QUESTIONS.items():
            category_results = []
            
            for q_data in questions:
                try:
                    result = self._run_question(qtbot, chat_widget, q_data, config)
                    evaluation = self._evaluate_result(result, q_data)
                    category_results.append({
                        "result": result,
                        "evaluation": evaluation,
                    })
                    time.sleep(2)  # 負荷軽減
                except Exception as e:
                    logging.getLogger(__name__).error(f"質問 {q_data['id']} でエラー: {e}")
                    category_results.append({
                        "result": {
                            "question": q_data["question"],
                            "answer": f"エラー: {str(e)}",
                            "found": False,
                            "evidence": [],
                            "elapsed_seconds": 0,
                        },
                        "evaluation": {
                            "has_evidence": False,
                            "evidence_count": 0,
                            "found": False,
                            "keyword_matches": [],
                            "expected_behavior": q_data["expected_behavior"],
                            "actual_behavior": "エラー",
                            "overall_score": 0,
                        },
                    })
            
            all_results[category] = category_results
        
        # レポート生成
        self._generate_report(all_results)
        
        # ウィンドウを閉じる
        window.close()
    
    def _run_question(self, qtbot, chat_widget, q_data, config):
        """単一質問をUI上で実行する"""
        question = q_data["question"]
        
        # 質問を入力
        chat_widget._input_box.setText(question)
        
        # 送信
        start_time = time.time()
        chat_widget._on_send()
        
        # ストリーミング完了待機（最大60秒）
        max_wait = 60
        waited = 0
        while chat_widget._streaming and waited < max_wait:
            qtbot.wait(100)
            waited += 0.1
        
        elapsed = time.time() - start_time
        
        # 回答を取得
        if chat_widget._messages:
            last_msg = chat_widget._messages[-1]
            answer = last_msg.get("content", "")
            evidence = last_msg.get("evidence", [])
            found = len(evidence) > 0
        else:
            answer = ""
            evidence = []
            found = False
        
        return {
            "question": question,
            "answer": answer,
            "found": found,
            "evidence": evidence,
            "elapsed_seconds": elapsed,
        }
    
    def _evaluate_result(self, result, q_data):
        """テスト結果を評価する"""
        answer = result['answer']
        evidence = result['evidence']
        expected_keywords = q_data.get("expected_keywords", [])
        expected_behavior = q_data.get("expected_behavior", "回答あり")
        
        evaluation = {
            "has_evidence": len(evidence) > 0,
            "evidence_count": len(evidence),
            "found": result['found'],
            "keyword_matches": [],
            "expected_behavior": expected_behavior,
            "actual_behavior": "",
            "overall_score": 0,
        }
        
        # キーワードマッチング
        for keyword in expected_keywords:
            if keyword.lower() in answer.lower():
                evaluation['keyword_matches'].append(keyword)
        
        # 期待される動作の判定
        if expected_behavior == "回答あり":
            evaluation['actual_behavior'] = "回答あり" if evaluation['found'] else "見つかりません"
        else:  # "見つかりません"
            evaluation['actual_behavior'] = "見つかりません" if not evaluation['found'] else "回答あり"
        
        # スコア計算
        score = 0
        if expected_behavior == "回答あり":
            if evaluation['found']:
                score += 40
            if evaluation['has_evidence']:
                score += 30
            if len(evaluation['keyword_matches']) > 0:
                score += 30
        else:  # "見つかりません"
            if not evaluation['found']:
                score += 50
            if "見つかりません" in answer:
                score += 50
        
        evaluation['overall_score'] = score
        
        return evaluation
    
    def _generate_report(self, all_results):
        """テスト結果レポートを生成する"""
        lines = []
        lines.append("# モノシリ チャット精度テストレポート（UI操作版）")
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
                lines.append(f"**期待動作**: {r['evaluation']['expected_behavior']}")
                lines.append(f"**実際動作**: {r['evaluation']['actual_behavior']}")
                lines.append(f"**回答時間**: {r['result']['elapsed_seconds']:.1f}秒")
                lines.append(f"**エビデンス数**: {r['evaluation']['evidence_count']}")
                lines.append("")
                lines.append("**回答**:")
                lines.append(f"```\n{r['result']['answer']}\n```")
                lines.append("")
                
                if r['result']['evidence']:
                    lines.append("**エビデンス**:")
                    for ev in r['result']['evidence'][:5]:
                        lines.append(f"- {ev.get('file_name', '不明')} (類似度: {ev.get('similarity', 0):.3f})")
                    lines.append("")
                
                lines.append("---")
                lines.append("")
        
        # レポート保存
        report_text = "\n".join(lines)
        report_file = Path(__file__).parent.parent.parent / "CHAT_ACCURACY_UI_TEST_REPORT.md"
        report_file.write_text(report_text, encoding="utf-8")
        logging.getLogger(__name__).info(f"レポート保存: {report_file}")
        
        # JSONも保存
        json_file = Path(__file__).parent.parent.parent / "CHAT_ACCURACY_UI_TEST_RESULTS.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logging.getLogger(__name__).info(f"JSON結果保存: {json_file}")
