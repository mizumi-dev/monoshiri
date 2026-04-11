@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 > nul

echo.
echo =============================================
echo   Monoshiri - Stopping...
echo =============================================
echo.

set "STOPPED_SOMETHING=0"

rem --- Stop Monoshiri (port 8501) ---
set "MONO_PID="
for /f "tokens=5" %%p in ('netstat -aon 2^>nul ^| findstr ":8501 " ^| findstr "LISTEN"') do (
    if not defined MONO_PID set "MONO_PID=%%p"
)

if defined MONO_PID (
    echo [INFO] Stopping Monoshiri process tree. PID: !MONO_PID!
    taskkill /PID !MONO_PID! /T /F >nul 2>&1
    if !errorlevel! equ 0 (
        echo [OK] Monoshiri stopped.
        set "STOPPED_SOMETHING=1"
    ) else (
        echo [WARN] Failed to stop Monoshiri. It may already be stopped.
    )
) else (
    echo [INFO] Monoshiri is not running.
)

rem --- Clean up legacy PID file if it exists ---
if exist "%~dp0logs\streamlit.pid" del /q "%~dp0logs\streamlit.pid" >nul 2>&1

rem --- Stop Ollama ---
set "OLLAMA_PID="
for /f "tokens=5" %%p in ('netstat -aon 2^>nul ^| findstr ":11434 " ^| findstr "LISTEN"') do (
    if not defined OLLAMA_PID set "OLLAMA_PID=%%p"
)

if defined OLLAMA_PID (
    echo [INFO] Stopping Ollama. PID: !OLLAMA_PID!
    taskkill /PID !OLLAMA_PID! /T /F >nul 2>&1
    if !errorlevel! equ 0 (
        echo [OK] Ollama stopped.
        set "STOPPED_SOMETHING=1"
    ) else (
        echo [WARN] Failed to stop Ollama. It may already be stopped.
    )
) else (
    echo [INFO] Ollama is not running.
)

echo.
if "!STOPPED_SOMETHING!"=="1" (
    echo [OK] All services stopped.
) else (
    echo [INFO] Nothing was running.
)
timeout /t 2 /nobreak >nul
exit /b 0
