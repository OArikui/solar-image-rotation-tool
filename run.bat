@echo off
setlocal
cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m solar_rotation_tool
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python was not found on PATH.
        echo Install Python 3.11 or later and the project dependencies first.
        exit /b 9009
    )
    python -m solar_rotation_tool
)

exit /b %ERRORLEVEL%
