@echo off
title GesturePlay Launcher
echo ====================================================
echo Starting GesturePlay...
echo ====================================================

REM 1. Terminate any previous orphaned python instances holding ports
powershell -Command "Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Id -ne $PID } | Stop-Process -Force -ErrorAction SilentlyContinue" >nul 2>&1

REM 2. Start WebSocket & HTTP server in background
echo Starting GesturePlay Server (HTTP 8080 + WS 8765)...
start "GesturePlay Server" /min cmd /c ".\venv\Scripts\python.exe server.py"

REM Give server 1.5 seconds to bind ports
timeout /t 2 /nobreak >nul

REM 3. Open Web Interface in default browser via HTTP (prevents YouTube Error 153)
echo Opening GesturePlay web interface in browser...
start http://127.0.0.1:8080/index.html

REM 4. Start Hand Detection camera process in foreground
echo Starting AI Hand Detection (hand_detection.py)...
echo (Press 'q' in the camera window to exit)
.\venv\Scripts\python.exe hand_detection.py

echo.
echo GesturePlay closed.
pause
