@echo off
setlocal
cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0run.py"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python was not found on PATH.
        echo Install Python 3.11 or later and the project dependencies first.
        exit /b 9009
    )
    python "%~dp0run.py"
)

exit /b %ERRORLEVEL%
