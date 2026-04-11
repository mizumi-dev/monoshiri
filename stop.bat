@echo off
chcp 65001 > nul

echo.
echo  =============================================
echo   Monoshiri - Stopping...
echo  =============================================
echo.

powershell -NoProfile -Command "
    $pidFile = Join-Path $PSScriptRoot 'logs\streamlit.pid'
    
    # PIDファイルが存在するかチェック
    if (-not (Test-Path $pidFile)) {
        Write-Host '[INFO] Monoshiriは実行中ではありません（PIDファイルなし）'
        exit 0
    }

    # 親PIDを読み込み
    $parentPid = Get-Content $pidFile -ErrorAction SilentlyContinue
    if (-not $parentPid) {
        Write-Host '[INFO] PIDファイルが空です'
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
        exit 0
    }

    # 親プロセスが存在するか確認
    $parentProc = Get-Process -Id $parentPid -ErrorAction SilentlyContinue
    if (-not $parentProc) {
        Write-Host '[INFO] 親プロセスは既に終了しています (PID: ' $parentPid ')'
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
        exit 0
    }

    Write-Host '[INFO] 親プロセス停止: PID ' $parentPid ' (' $parentProc.ProcessName ')'

    # 子孫プロセス（親PIDの子、孫、ひ孫...）を再帰的に収集
    function Get-Descendants($ppid) {
        $children = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | 
            Where-Object { $_.ParentProcessId -eq $ppid } |
            Select-Object -ExpandProperty ProcessId
        
        $allDescendants = @($children)
        foreach ($child in $children) {
            $grandchildren = Get-Descendants $child
            $allDescendants += $grandchildren
        }
        return $allDescendants
    }

    # 子孫プロセスを収集（深さ優先）
    $descendants = Get-Descendants $parentPid
    Write-Host '[INFO] 子孫プロセス: ' $descendants.Count '個'

    # 子孫プロセスを先に停止（深いものから順に）
    foreach ($pid in ($descendants | Sort-Object -Descending)) {
        $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($proc) {
            try {
                # まず穏やかに終了を試みる
                $proc.CloseMainWindow() | Out-Null
                Start-Sleep -Milliseconds 100
                if (-not $proc.HasExited) {
                    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                }
                Write-Host '[OK] 子プロセス停止: PID ' $pid ' (' $proc.ProcessName ')'
            } catch {
                Write-Host '[WARN] 子プロセス停止失敗: PID ' $pid
            }
        }
    }

    # 親プロセスを停止
    try {
        $parentProc.CloseMainWindow() | Out-Null
        Start-Sleep -Milliseconds 200
        if (-not $parentProc.HasExited) {
            Stop-Process -Id $parentPid -Force -ErrorAction SilentlyContinue
        }
        Write-Host '[OK] 親プロセス停止完了: PID ' $parentPid
    } catch {
        Write-Host '[WARN] 親プロセス停止失敗: ' $_.Exception.Message
    }

    # PIDファイルを削除
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    Write-Host '[OK] Monoshiri停止完了（他のPython開発作業には影響しません）'
"

echo.
timeout /t 2 /nobreak >nul
