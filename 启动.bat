@echo off
chcp 65001 >nul
title ComfyUI Image Studio
cd /d "%~dp0"

REM 按需修改下面两行指向你的 ComfyUI 安装位置
if "%COMFY_DIR%"=="" set "COMFY_DIR=%~dp0..\ComfyUI"
if "%EASYAI_PORT%"=="" set "EASYAI_PORT=7860"

echo.
echo  ComfyUI Image Studio
echo  ComfyUI : %COMFY_DIR%
echo  Port    : %EASYAI_PORT%
echo.

python server.py
pause
