@echo off
title Ada — Document Intelligence Agent
echo Starting Ada...

if exist "backend\venv\Scripts\python.exe" (
    backend\venv\Scripts\python.exe run_app.py %*
) else (
    python run_app.py %*
)
pause
