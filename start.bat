@echo off
chcp 65001 > nul

echo.
echo  =============================================
echo   Monoshiri - Local Document Search AI
echo   Starting...
echo  =============================================
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    echo Please install Python 3.10 or later.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo [SETUP] Installing required packages...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Installation failed.
        pause
        exit /b 1
    )
)

echo [OK] Launching browser automatically...
echo      Manual access: http://localhost:8501
echo.

python -m streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

pause
