@echo off
setlocal EnableExtensions
title ImageLab Launcher
cd /d "%~dp0"

rem Prefer the project virtual environment so Python does not need to be on PATH.
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%PYTHON%" goto :python_ready

where py >nul 2>nul
if not errorlevel 1 (
  set "BOOTSTRAP=py -3"
  goto :create_venv
)

where python >nul 2>nul
if not errorlevel 1 (
  set "BOOTSTRAP=python"
  goto :create_venv
)

echo [ERROR] Python was not found.
echo Install Python 3.11 or newer, then run this file again.
pause
exit /b 1

:create_venv
echo [ImageLab] Creating the project virtual environment...
%BOOTSTRAP% -m venv .venv
if errorlevel 1 goto :failed
set "PYTHON=%~dp0.venv\Scripts\python.exe"
echo [ImageLab] Installing dependencies...
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

:python_ready
if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs
echo [ImageLab] Starting at http://127.0.0.1:8000
start "ImageLab Server" "%PYTHON%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8000"
exit /b 0

:failed
echo [ERROR] ImageLab setup failed. Check the messages above.
pause
exit /b 1
