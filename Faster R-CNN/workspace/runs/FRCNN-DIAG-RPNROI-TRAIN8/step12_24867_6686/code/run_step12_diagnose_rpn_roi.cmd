@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0..\.."
if errorlevel 1 exit /b 1
if /i "%~1"=="--check-preparation" goto check_preparation
if not "%~1"=="" exit /b 1
set "logdir=%CD%\workspace\logs\step12_%RANDOM%_%RANDOM%"
mkdir "%logdir%"
if errorlevel 1 exit /b 1
echo STEP12_RPN_ROI_TRAIN8_USER_DIAGNOSTIC
echo Inference only: two existing epoch100 weights, eight original train images. No training or test.
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step12_diagnose_rpn_roi.py" --log-dir "%logdir%"
set "result=%ERRORLEVEL%"
echo STEP12_EXIT_CODE=%result%
echo LOG_DIR=%logdir%
exit /b %result%
:check_preparation
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step12_diagnose_rpn_roi.py" --check-preparation
exit /b %ERRORLEVEL%
