@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found in this folder.
  echo Create it first: python -m venv .venv
  pause
  exit /b 1
)

".venv\Scripts\python.exe" main.py
echo.
pause
