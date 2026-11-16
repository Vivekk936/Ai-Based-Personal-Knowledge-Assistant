
@echo off
title Ada - Document Intelligence Agent

:: Move to the folder containing this BAT file
cd /d "%~dp0"

echo.
echo Starting Ada...
echo.

:: Check whether the project's virtual environment exists
if exist "%~dp0backend\venv\Scripts\python.exe" (
    echo Using project virtual environment...
    "%~dp0backend\venv\Scripts\python.exe" "%~dp0run_app.py" %*
) else (
    echo Using system Python...
    python "%~dp0run_app.py" %*
)

echo.
echo Ada has stopped.
pause
