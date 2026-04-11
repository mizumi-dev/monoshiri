@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 > nul

cd /d "%~dp0"

echo.
echo =============================================
echo   Monoshiri - Local Document Search AI
echo =============================================
echo.

rem --- Check Python ---
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found.
    echo Please install Python 3.10 or later.
    pause
    exit /b 1
)

for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%i"
echo [OK] Python: !PYTHON_EXE!

python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install required packages.
        pause
        exit /b 1
    )
)

if not exist "%~dp0logs" mkdir "%~dp0logs"

rem --- If Monoshiri is already running, just open browser ---
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue; if ($c) { exit 0 } else { exit 1 }" >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] Monoshiri is already running.
    start "" "http://localhost:8501"
    exit /b 0
)

rem --- Find ollama.exe ---
set "OLLAMA_EXE="

for /f "delims=" %%i in ('where ollama.exe 2^>nul') do (
    set "OLLAMA_EXE=%%i"
    goto :ollama_found
)
if exist "%LocalAppData%\Programs\Ollama\ollama.exe" (
    set "OLLAMA_EXE=%LocalAppData%\Programs\Ollama\ollama.exe"
    goto :ollama_found
)
if exist "%ProgramFiles%\Ollama\ollama.exe" (
    set "OLLAMA_EXE=%ProgramFiles%\Ollama\ollama.exe"
    goto :ollama_found
)

echo [WARN] ollama.exe was not found. Chat may not work until Ollama is installed.
goto :start_monoshiri

:ollama_found
echo [OK] Ollama: !OLLAMA_EXE!

rem --- Check if Ollama is already responding ---
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 2 -UseBasicParsing | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] Ollama is already running.
    goto :start_monoshiri
)

rem --- Start Ollama and wait for it ---
echo [INFO] Starting Ollama...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '!OLLAMA_EXE!' -ArgumentList 'serve' -WindowStyle Hidden" >nul 2>&1

set "OLLAMA_WAIT=0"
:ollama_wait_loop
timeout /t 1 /nobreak >nul
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 2 -UseBasicParsing | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] Ollama is running.
    goto :start_monoshiri
)
set /a OLLAMA_WAIT+=1
if !OLLAMA_WAIT! lss 15 goto :ollama_wait_loop

echo [WARN] Ollama did not respond within 15 seconds. Continuing anyway...

rem --- Start Monoshiri ---
:start_monoshiri
echo [INFO] Starting Monoshiri...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -WorkingDirectory '%~dp0' -WindowStyle Hidden -FilePath '%PYTHON_EXE%' -ArgumentList '-m','streamlit','run','app.py','--server.port','8501','--server.headless','true','--browser.gatherUsageStats','false'" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to start Monoshiri.
    pause
    exit /b 1
)

set "MONO_WAIT=0"
:mono_wait_loop
timeout /t 1 /nobreak >nul
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8501' -TimeoutSec 2 -UseBasicParsing | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if !errorlevel! equ 0 goto :mono_ready
set /a MONO_WAIT+=1
if !MONO_WAIT! lss 20 goto :mono_wait_loop

echo [ERROR] Monoshiri did not become ready in time.
pause
exit /b 1

:mono_ready
echo [OK] Monoshiri is running.
echo      URL: http://localhost:8501
echo      Stop: stop.bat
start "" "http://localhost:8501"
timeout /t 2 /nobreak >nul
exit /b 0
