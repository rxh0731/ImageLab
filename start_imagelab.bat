@echo off
setlocal EnableExtensions
title ImageLab Launcher
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] 未找到 Python。请先安装 Python 3.11 或更高版本，并勾选 Add Python to PATH。
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [ImageLab] 首次运行，正在创建 Python 虚拟环境...
  python -m venv .venv
  if errorlevel 1 goto :failed
  echo [ImageLab] 正在安装依赖，这可能需要几分钟...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :failed
)

if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs

echo [ImageLab] 启动服务：http://127.0.0.1:8000
start "ImageLab Server" cmd /k ""%CD%\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8000"
exit /b 0

:failed
echo [ERROR] ImageLab 环境初始化失败，请查看上面的错误信息。
pause
exit /b 1
