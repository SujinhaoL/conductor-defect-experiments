@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0..\.."
if errorlevel 1 exit /b 1
if /i "%~1"=="--show-plan" goto show_plan
if /i "%~1"=="--check-preparation" goto check_preparation
if not "%~1"=="" if /i not "%~1"=="--verify-only" goto bad_argument
set "logdir=%CD%\workspace\logs\step02c_%RANDOM%_%RANDOM%"
mkdir "%logdir%"
if errorlevel 1 exit /b 1
echo STEP02C_MMCV_SETUP
echo LOG_DIR=%logdir%
if /i "%~1"=="--verify-only" goto run
set "VSLANG=1033"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 10.0.19041.0 -vcvars_ver=14.29 > "%logdir%\vcvarsall.txt" 2>&1
if errorlevel 1 goto compiler_failed
type "%logdir%\vcvarsall.txt"
if not defined VCToolsVersion goto compiler_failed
if not "%VCToolsVersion:~0,5%"=="14.29" goto compiler_failed
if not "%WindowsSDKVersion%"=="10.0.19041.0\" goto compiler_failed
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
set "CXX=cl"
set "MMCV_CUDA_ARGS="
set "MMCV_WITH_OPS=1"
set "FORCE_CUDA=1"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "MAX_JOBS=4"

:run
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step02c_build_mmcv.py" --log-dir "%logdir%" %1
set "stage_exit=%ERRORLEVEL%"
echo LOG_DIR=%logdir%
echo STEP02C_EXIT_CODE=%stage_exit%
exit /b %stage_exit%

:compiler_failed
type "%logdir%\vcvarsall.txt"
echo STEP02C_MMCV_FAIL: MSVC 14.29 and SDK 10.0.19041.0 are required.
echo LOG_DIR=%logdir%
echo STEP02C_EXIT_CODE=1
exit /b 1

:check_preparation
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step02c_build_mmcv.py" --check-preparation
exit /b %ERRORLEVEL%

:bad_argument
echo ERROR: Supported options are --show-plan, --check-preparation, and --verify-only.
exit /b 1

:show_plan
echo PLAN ONLY - no installation, compilation, or GPU operator execution.
echo Official MMCV 2.1.0: commit 57c4e25e06e2d4f8a9357c84bcd24089a284dc88
echo Source: %CD%\workspace\src\mmcv
echo Environment: %CD%\.venv
echo MMEngine 0.10.7; offline dependencies with pinned versions and hashes.
echo MSVC 14.29; CUDA 12.8.1 / nvcc 12.8.93; sm_120; MAX_JOBS=4.
echo Temporary ASCII drive alias for this directory; original files stay in place.
echo Build wheel, record hash, install, check CPU/CUDA NMS and RoIAlign.
echo No MMDetection installation, model training, or dataset access in this step.
exit /b 0
