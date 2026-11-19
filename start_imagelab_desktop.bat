@echo off
setlocal EnableExtensions
title ImageLab Desktop Launcher
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%PYTHON%" goto :ready

where py >nul 2>nul
if not errorlevel 1 (
  set "BOOTSTRAP=py -3"
  goto :create
)
where python >nul 2>nul
if not errorlevel 1 (
  set "BOOTSTRAP=python"
  goto :create
)
echo [ERROR] Python was not found. Install Python 3.11 or newer.
pause
exit /b 1

:create
echo [ImageLab] Creating virtual environment...
%BOOTSTRAP% -m venv .venv
if errorlevel 1 goto :failed
set "PYTHON=%~dp0.venv\Scripts\python.exe"
echo [ImageLab] Installing dependencies...
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

:ready
"%PYTHON%" -c "import PIL, numpy, cv2, PySide6, tifffile; import psd_tools; from importlib.metadata import version; assert tuple(int(x) for x in version('psd-tools').split('.')[:2]) >= (1, 27)" >nul 2>nul
if not errorlevel 1 goto :launch
echo [ImageLab] Updating required dependencies...
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

:launch
if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs
echo [ImageLab] Starting desktop application...
start "ImageLab Desktop" "%PYTHON%" -m app.desktop
exit /b 0

:failed
echo [ERROR] ImageLab setup failed.
pause
exit /b 1
