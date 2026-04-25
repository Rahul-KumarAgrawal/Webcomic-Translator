@echo off
chcp 65001 >nul
REM ═══════════════════════════════════════════════════════════════════════════
REM  CBZ Translator — Launch Web UI
REM  Automatically resolves Python from: local, venv, or system.
REM  Run setup.bat first if this is your first time!
REM ═══════════════════════════════════════════════════════════════════════════
cd /d "%~dp0"

REM ── Resolve Python ────────────────────────────────────────────────────────
set "PYTHON="

if exist ".python_path.txt" (
    set /p PYTHON=<.python_path.txt
)

if not defined PYTHON (
    if exist ".\python\python.exe"          set "PYTHON=.\python\python.exe"
    if exist ".\venv\Scripts\python.exe"    set "PYTHON=.\venv\Scripts\python.exe"
)

if not defined PYTHON (
    python --version >nul 2>&1
    if not errorlevel 1 set "PYTHON=python"
)

if not defined PYTHON (
    echo [ERROR] Python not found. Please run setup.bat first!
    pause & exit /b 1
)

echo [WEB] Using Python: %PYTHON%
echo [WEB] Starting CBZ Translator Web UI at http://localhost:5000
echo [WEB] Press Ctrl+C to stop.
echo.
"%PYTHON%" web\app.py
pause
