@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0..\.."
if errorlevel 1 exit /b 1
if /i "%~1"=="--show-plan" goto show_plan
if /i "%~1"=="--check-downloads" if not exist ".venv\Scripts\python.exe" exit /b 1

rem All experiment packages go into this project's .venv, never into base or YOLO.
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONNOUSERSITE=1"
rem Scope the verified direct connection to this batch process and its children.
rem NO_PROXY also prevents Requests from using the Windows registry proxy.
rem setlocal restores all caller settings on exit. TLS verification stays enabled.
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "NO_PROXY=*"
set "UV_PYTHON_INSTALL_DIR=%CD%\workspace\runtimes\python"
set "UV_CACHE_DIR=%CD%\workspace\cache\uv"
set "UV_VENV_CLEAR="
set "UV_VENV_SEED="
set "logdir=%CD%\workspace\logs\step01_%RANDOM%_%RANDOM%"
mkdir "%logdir%"
if errorlevel 1 exit /b 1
echo STEP01_BASE_ENV_SETUP
echo LOG_DIR=%logdir%
echo DOWNLOAD_MODE=DIRECT_FOR_THIS_SCRIPT_ONLY
where uv >nul 2>&1
if errorlevel 1 goto no_uv
uv --version > "%logdir%\uv_version.txt" 2>&1

if exist ".venv" goto existing_env
echo [1/5] Creating dedicated Python 3.10.20 environment...
uv venv --no-project --no-config --managed-python --python 3.10.20 ".venv"
if errorlevel 1 goto failed
> ".venv\.frcnn_step01" echo FRCNN_STEP01_PYTHON_3_10_20
goto identity

:existing_env
if not exist ".venv\.frcnn_step01" goto unowned_env
echo [1/5] Reusing the environment previously created by this step.

:identity
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step01_check_cuda.py" --identity-only
if errorlevel 1 goto failed
if /i "%~1"=="--check-downloads" goto check_downloads
echo [2/5] Installing pinned package tools and NumPy...
uv pip install --no-config --python ".venv\Scripts\python.exe" --default-index https://pypi.org/simple "pip==25.1.1"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -I -X utf8 -m pip --isolated --disable-pip-version-check --log "%logdir%\pip_install.log" --cache-dir "%CD%\workspace\cache\pip" install --no-input --only-binary=:all: --index-url https://pypi.org/simple --report "%logdir%\base_install_report.json" -r "workspace\configs\step01_base_requirements.txt"
if errorlevel 1 goto failed
echo [3/5] Installing official PyTorch CUDA 12.8 wheels. This is a large download.
".venv\Scripts\python.exe" -I -X utf8 -m pip --isolated --disable-pip-version-check --log "%logdir%\pip_install.log" --cache-dir "%CD%\workspace\cache\pip" install --no-input --only-binary=:all: --index-url https://download.pytorch.org/whl/cu128 --report "%logdir%\torch_install_report.json" "torch==2.7.1+cu128" "torchvision==0.22.1+cu128"
if errorlevel 1 goto failed
echo [4/5] Checking package dependencies and saving installed versions...
".venv\Scripts\python.exe" -I -X utf8 -m pip --isolated check > "%logdir%\pip_check.txt" 2>&1
if errorlevel 1 goto pip_check_failed
type "%logdir%\pip_check.txt"
".venv\Scripts\python.exe" -I -X utf8 -m pip --isolated freeze --all > "%logdir%\requirements_freeze.txt"
if errorlevel 1 goto failed
echo [5/5] Testing CUDA operations. No dataset, optimizer, or training is used.
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step01_check_cuda.py" --output-dir "%logdir%"
if errorlevel 1 goto failed
echo STEP01_BASE_ENV_PASS
echo LOG_DIR=%logdir%
echo STEP01_EXIT_CODE=0
exit /b 0

:check_downloads
echo Checking the original failed dependency with pip --dry-run. No packages will be installed.
".venv\Scripts\python.exe" -I -X utf8 -m pip --isolated --disable-pip-version-check --log "%logdir%\pip_network_check.log" --cache-dir "%CD%\workspace\cache\pip" install --dry-run --ignore-installed --no-deps --retries 0 --timeout 15 --only-binary=:all: --index-url https://pypi.org/simple --report "%logdir%\network_check_report.json" "setuptools==80.9.0"
if errorlevel 1 goto failed
echo STEP01_DOWNLOAD_CHECK_PASS
echo LOG_DIR=%logdir%
exit /b 0

:pip_check_failed
type "%logdir%\pip_check.txt"
goto failed
:no_uv
echo ERROR: uv is not on PATH. Send the complete terminal output.
goto failed
:unowned_env
echo ERROR: Existing .venv was not created by this script. Nothing in it was changed.
goto failed
:failed
echo STEP01_BASE_ENV_FAIL
echo LOG_DIR=%logdir%
echo Send the complete terminal output. Do not continue to MMCV installation yet.
echo STEP01_EXIT_CODE=1
exit /b 1

:show_plan
echo STEP01 PLAN ONLY - no environment creation, downloads, or GPU execution.
echo Environment: %CD%\.venv
echo Python: 3.10.20, downloaded separately by uv into workspace\runtimes\python
echo PyTorch: 2.7.1+cu128
echo torchvision: 0.22.1+cu128
echo NumPy: 1.26.4
echo Official PyTorch index: https://download.pytorch.org/whl/cu128
echo Download mode: direct for this script only; Windows proxy settings are not modified.
echo CUDA checks: tensor math, matrix multiply, convolution, torchvision NMS and RoIAlign.
echo MMCV and MMDetection are not installed in this step.
exit /b 0
