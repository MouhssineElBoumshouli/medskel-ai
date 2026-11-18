@echo off
rem Double-click this file to start the MedSkel AI app.
rem It opens the app in your browser by itself. Close this black window to stop the app.
title MedSkel AI
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The Python environment .venv was not found in this folder.
    echo Follow the "Quick start" in README.md first.
    pause
    exit /b 1
)
echo Starting MedSkel AI... the browser opens in a few seconds.
echo Keep this window open while you use the app. Close it to stop the app.
echo.
set MEDSKEL_OPEN_BROWSER=1
".venv\Scripts\python.exe" app.py
echo.
echo The app has stopped. If you see an error above, send it to Claude.
pause
