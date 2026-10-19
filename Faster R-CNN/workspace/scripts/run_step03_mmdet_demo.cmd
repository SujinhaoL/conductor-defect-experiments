@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0..\.."
if errorlevel 1 exit /b 1
if /i "%~1"=="--check-preparation" goto check_preparation
if /i "%~1"=="--show-plan" goto show_plan
if not "%~1"=="" goto bad_argument
set "logdir=%CD%\workspace\logs\step03_%RANDOM%_%RANDOM%"
mkdir "%logdir%"
if errorlevel 1 exit /b 1
echo STEP03_MMDET_OFFICIAL_DEMO
echo LOG_DIR=%logdir%
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step03_mmdet_demo.py" --log-dir "%logdir%"
set "stage_exit=%ERRORLEVEL%"
echo LOG_DIR=%logdir%
echo STEP03_EXIT_CODE=%stage_exit%
exit /b %stage_exit%

:check_preparation
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step03_mmdet_demo.py" --check-preparation
exit /b %ERRORLEVEL%

:show_plan
echo Install six verified local runtime wheels and editable official MMDetection 3.3.0.
echo Existing Python, PyTorch, MMCV and MMEngine versions are checked and preserved.
echo Audit the complete official COCO detector checkpoint, then infer on demo/demo.jpg.
echo Save predictions, visualization, resolved config, package versions and logs.
echo This stage uses 80 COCO classes. No training or conductor dataset access.
exit /b 0

:bad_argument
echo ERROR: Supported options are --show-plan and --check-preparation.
exit /b 1
