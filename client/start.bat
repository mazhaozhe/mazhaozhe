@echo off
setlocal
cd /d "%~dp0"

if not exist "..\.venv\Scripts\python.exe" (
  echo [ERROR] Python environment not found: ..\.venv
  echo Please install the original GeneralTaskAgent dependencies first.
  pause
  exit /b 1
)

if not exist "node_modules\electron" (
  echo Installing desktop client dependencies...
  set ELECTRON_RUN_AS_NODE=
  call npm install
  if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
  )
)

set ELECTRON_RUN_AS_NODE=
call npm start
if errorlevel 1 pause
endlocal
