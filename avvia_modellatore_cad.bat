@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Ambiente Python non trovato nella cartella .venv
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "modellatore_cad.py"
if errorlevel 1 pause
