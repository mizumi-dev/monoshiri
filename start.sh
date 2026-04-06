#!/bin/bash
set -e

echo ""
echo "████████████████████████████████████████████"
echo "　　モノシリ - 社内資料探索AI"
echo "　　社内資料は、一切外に出ません。"
echo "████████████████████████████████████████████"
echo ""

# Pythonバージョン確認
if ! command -v python3 &> /dev/null; then
    echo "[エラー] Python3 が見つかりません。"
    echo "Python 3.10 以上をインストールしてください。"
    exit 1
fi

# 依存ライブラリの確認
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "[初回セットアップ] 必要なライブラリをインストール中..."
    pip3 install -r requirements.txt
fi

echo "[起動中] ブラウザが自動的に開きます..."
echo "         手動で開く場合: http://localhost:8501"
echo ""

# Streamlitを起動
streamlit run app.py \
    --server.port 8501 \
    --server.headless false \
    --browser.gatherUsageStats false
