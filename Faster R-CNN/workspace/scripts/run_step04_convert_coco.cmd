@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0..\.."
if errorlevel 1 exit /b 1
if not "%~1"=="" if /i not "%~1"=="--audit-only" goto bad_argument
set "logdir=%CD%\workspace\logs\step04_%RANDOM%_%RANDOM%"
mkdir "%logdir%"
if errorlevel 1 exit /b 1
echo STEP04_YOLO_TO_COCO
echo LOG_DIR=%logdir%
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step04_convert_coco.py" --log-dir "%logdir%" %1
set "stage_exit=%ERRORLEVEL%"
echo LOG_DIR=%logdir%
echo STEP04_EXIT_CODE=%stage_exit%
exit /b %stage_exit%

:bad_argument
echo ERROR: Only --audit-only is supported. Omit it to perform conversion.
exit /b 1
