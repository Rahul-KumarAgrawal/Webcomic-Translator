@echo off
chcp 65001 >nul
REM ═══════════════════════════════════════════════════════════════════════════
REM  CBZ Translator — Batch Processing CLI
REM  Usage: run_batch.bat --input ./input --output D:/translated --series "Solo Leveling"
REM  Automatically resolves Python from: local, venv, or system.
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

echo [BATCH] Using Python: %PYTHON%
"%PYTHON%" core\batch_processor.py %*
pause
