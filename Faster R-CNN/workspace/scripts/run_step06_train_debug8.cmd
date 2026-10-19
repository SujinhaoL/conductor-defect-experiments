@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0..\.."
if errorlevel 1 exit /b 1
if /i "%~1"=="--check-preparation" goto check_preparation
if not "%~1"=="" exit /b 1
set "logdir=%CD%\workspace\logs\step06_%RANDOM%_%RANDOM%"
mkdir "%logdir%"
if errorlevel 1 exit /b 1
echo STEP06_DEBUG8_USER_TRAINING
echo LOG_DIR=%logdir%
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step06_run_debug8.py" --log-dir "%logdir%"
set "stage_exit=%ERRORLEVEL%"
echo LOG_DIR=%logdir%
echo STEP06_EXIT_CODE=%stage_exit%
exit /b %stage_exit%

:check_preparation
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step06_run_debug8.py" --check-preparation
exit /b %ERRORLEVEL%
