@echo off
title WeatherWise Launcher
echo ===================================================
echo           Starting WeatherWise Intelligence
echo ===================================================
echo.
cd /d "%~dp0"

echo [1/2] Checking Python Virtual Environment...
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe!
    pause
    exit /b 1
)

echo [2/2] Launching Flask server...
echo.
echo Application running on http://127.0.0.1:5000
echo Press Ctrl+C in this window to stop the server anytime.
echo.

REM Automatically open the browser after a 2-second delay
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:5000"

REM Run the Flask app
.venv\Scripts\python.exe app.py
pause
