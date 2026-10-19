@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0..\.."
if errorlevel 1 exit /b 1
if /i "%~1"=="--check-preparation" goto check_preparation
if not "%~1"=="" exit /b 1
set "logdir=%CD%\workspace\logs\step09_%RANDOM%_%RANDOM%"
mkdir "%logdir%"
if errorlevel 1 exit /b 1
echo STEP09_MERGE130_USER_TRAINING
echo LOG_DIR=%logdir%
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step09_run_merge130.py" --log-dir "%logdir%"
set "stage_exit=%ERRORLEVEL%"
echo LOG_DIR=%logdir%
echo STEP09_EXIT_CODE=%stage_exit%
exit /b %stage_exit%

:check_preparation
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step09_run_merge130.py" --check-preparation
exit /b %ERRORLEVEL%
