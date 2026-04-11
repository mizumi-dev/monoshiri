@echo off
chcp 65001 > nul

cd /d "%~dp0"

echo.
echo  =============================================
echo   Monoshiri - Local Document Search AI
echo   Starting...
echo  =============================================
echo.

:: --- Python check ---
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    echo Please install Python 3.10 or later.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

:: --- Resolve real Python path (WindowsApps stub workaround) ---
for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%i"
echo [OK] Python: %PYTHON_EXE%

:: --- Package check ---
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

:: --- Already running check ---
echo [CHECK] Checking if Monoshiri is already running...
powershell -NoProfile -Command "$conn = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue; if ($conn) { exit 0 } else { exit 1 }" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Monoshiri is already running.
    echo      Access: http://localhost:8501
    echo.
    start http://localhost:8501
    echo This window will close automatically.
    timeout /t 2 /nobreak >nul
    exit /b 0
)

:: --- Port cleanup ---
echo [CHECK] Releasing port 8501...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
timeout /t 1 /nobreak >nul

:: --- Ollama check ---
echo [CHECK] Checking Ollama (AI inference engine)...
call :check_ollama
goto :start_app

:: --- Launch app (background process - keeps running after this window closes) ---
:start_app
echo.
echo [OK] Launching Monoshiri in background...
echo      (Continues running even after this window closes)
echo.

:: Create logs directory
if not exist "%~dp0logs" mkdir "%~dp0logs"

:: Launch Streamlit as a completely detached background process via PowerShell
:: -NoNewWindow + -RedirectStandardOutput ensures the process survives terminal close
powershell -NoProfile -Command "$p = Start-Process -FilePath \"%PYTHON_EXE%\" -ArgumentList '-m','streamlit','run','app.py','--server.port','8501','--server.headless','true','--browser.gatherUsageStats','false' -WorkingDirectory '%~dp0' -NoNewWindow -RedirectStandardOutput '%~dp0logs\streamlit.log' -RedirectStandardError '%~dp0logs\streamlit_err.log' -PassThru; $p.Id | Out-File -FilePath '%~dp0logs\streamlit.pid' -Encoding ascii"

if %errorlevel% neq 0 (
    echo [ERROR] Failed to start Monoshiri.
    pause
    exit /b 1
)

:: Wait for Streamlit to be ready
echo [WAIT] Waiting for startup...
set /a RETRY=0
:wait_loop
    timeout /t 1 /nobreak >nul
    powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://localhost:8501' -TimeoutSec 1 -UseBasicParsing | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
    if %errorlevel% equ 0 goto :ready
    set /a RETRY+=1
    if %RETRY% lss 15 goto :wait_loop

:ready
echo [OK] Monoshiri is running.
echo      Access: http://localhost:8501
echo      To stop: run stop.bat
echo.
start http://localhost:8501

echo This window will close in 3 seconds.
echo Monoshiri continues running in the background.
timeout /t 3 /nobreak >nul
exit /b 0


:check_ollama
    powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://localhost:11434/api/version' -TimeoutSec 2 -UseBasicParsing | Out-Null; exit 0 } catch { exit 1 }"
    if %errorlevel% equ 0 (
        echo [OK] Ollama already running.
        goto :eof
    )

    where ollama >nul 2>&1
    if %errorlevel% equ 0 (
        echo [INFO] Starting Ollama...
        start /B "" ollama serve
        timeout /t 3 /nobreak >nul
        echo [OK] Ollama started.
        goto :eof
    )

    echo.
    echo  ------------------------------------------------
    echo   Ollama (AI engine) is not installed.
    echo   Ollama runs LOCALLY. No data is sent externally.
    echo  ------------------------------------------------
    echo.
    choice /C YN /M "  Install Ollama now? [Y=Yes / N=Skip]"
    if %errorlevel% equ 2 (
        echo [SKIP] AI answer generation will be unavailable.
        goto :eof
    )

    call :install_ollama
    goto :eof


:install_ollama
    set "OLLAMA_LOG=%~dp0ollama_install.log"
    set "OLLAMA_INSTALLER=%TEMP%\OllamaSetup.exe"

    echo [INFO] Downloading Ollama installer...
    powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://ollama.com/download/OllamaSetup.exe' -OutFile '%OLLAMA_INSTALLER%' -UseBasicParsing" 2>"%OLLAMA_LOG%"
    if %errorlevel% neq 0 (
        echo [ERROR] Download failed.
        if exist "%OLLAMA_LOG%" type "%OLLAMA_LOG%"
        pause
        goto :eof
    )

    echo [INFO] Running installer (UAC approval may be required)...
    "%OLLAMA_INSTALLER%" /S
    if %errorlevel% neq 0 goto :install_failed

    powershell -NoProfile -Command "$mp = [System.Environment]::GetEnvironmentVariable('PATH','Machine'); $up = [System.Environment]::GetEnvironmentVariable('PATH','User'); $env:PATH = $mp + ';' + $up" >nul 2>&1

    echo [OK] Ollama installed.
    timeout /t 2 /nobreak >nul

    start /B "" ollama serve
    timeout /t 3 /nobreak >nul
    echo [OK] Ollama started.
    goto :eof


:install_failed
    echo.
    echo [ERROR] Ollama installation failed.
    echo.
    echo   Possible causes:
    echo     - Administrator permission was not granted (UAC denied)
    echo     - Company security policy blocks software installation
    echo.
    echo   What to do:
    echo     - Please contact your system administrator
    echo     - Share this log file: %OLLAMA_LOG%
    echo.
    if exist "%OLLAMA_LOG%" type "%OLLAMA_LOG%"
    echo.
    echo   Monoshiri will start without AI answer generation.
    pause
    goto :eof
