@echo off
rem One-click launcher for Windows: sets up .venv on first run, then starts the app.
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10+ from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 goto :failed
)

".venv\Scripts\python.exe" -c "import streamlit, sklearn, graphviz, matplotlib" >nul 2>nul
if errorlevel 1 (
    echo Installing requirements, this takes a minute the first time ...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
)

echo Starting Arbor at http://localhost:8501  (press Ctrl+C to stop)
".venv\Scripts\python.exe" -m streamlit run app.py
exit /b 0

:failed
echo.
echo Setup failed. Read the messages above, then run this file again.
pause
exit /b 1
