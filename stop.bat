@echo off
chcp 65001 > nul

echo.
echo  =============================================
echo   Monoshiri - Stopping...
echo  =============================================
echo.

powershell -NoProfile -Command "
    $conn = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue;
    if ($conn) {
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue;
        Write-Host '[OK] Monoshiri stopped.'
    } else {
        Write-Host '[INFO] Monoshiri is not running.'
    }
    # PIDファイルが残っていれば削除
    $pidFile = Join-Path $PSScriptRoot 'logs\streamlit.pid'
    if (Test-Path $pidFile) { Remove-Item $pidFile -Force }
"

echo.
timeout /t 2 /nobreak >nul
