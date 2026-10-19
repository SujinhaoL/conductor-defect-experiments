@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0..\.."
if errorlevel 1 exit /b 1
set "frcnn_root=%CD%"
set "cuda_root=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "vsroot=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "VSLANG=1033"
set "components=nvcc_12.8 cudart_12.8 cuda_profiler_api_12.8 thrust_12.8 nvrtc_12.8 nvrtc_dev_12.8 nvjitlink_12.8 nvtx_12.8 cublas_12.8 cublas_dev_12.8 cufft_12.8 cufft_dev_12.8 curand_12.8 curand_dev_12.8 cusolver_12.8 cusolver_dev_12.8 cusparse_12.8 cusparse_dev_12.8"
if /i "%~1"=="--show-plan" goto show_plan
set "logdir=%frcnn_root%\workspace\logs\step02b_%RANDOM%_%RANDOM%"
mkdir "%logdir%"
if errorlevel 1 exit /b 1
echo STEP02B_CUDA_TOOLKIT_SETUP
echo LOG_DIR=%logdir%
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step01_check_cuda.py" --identity-only
if errorlevel 1 goto failed
nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader > "%logdir%\gpu_driver_before.txt" 2>&1
if errorlevel 1 goto failed
type "%logdir%\gpu_driver_before.txt"
if /i "%~1"=="--verify-only" goto verify
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step02a_prepare_msvc.py" --require-admin
if errorlevel 1 goto failed
if not exist "%vsroot%\VC\Auxiliary\Build\vcvarsall.bat" goto missing_msvc
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step02b_prepare_cuda.py" --verify-installer > "%logdir%\installer_verification.txt" 2>&1
if errorlevel 1 goto signature_failed
type "%logdir%\installer_verification.txt"
rem Review an existing Toolkit by validation, rather than installing over it.
if exist "%cuda_root%" goto existing
> "%logdir%\installer_components.txt" echo %components%
echo Installing CUDA Toolkit 12.8 Update 1 to %cuda_root%
echo Selected components: %components%
echo Display.Driver is excluded. Automatic restart is disabled.
echo Silent installation may take several minutes with no new console output.
start "" /wait "%frcnn_root%\workspace\installers\cuda_12.8.1_572.61_windows.exe" -s -n %components% > "%logdir%\installer_console.txt" 2>&1
set "installer_exit=%ERRORLEVEL%"
> "%logdir%\installer_exit_code.txt" echo %installer_exit%
echo NVIDIA_INSTALLER_EXIT_CODE=%installer_exit%
if "%installer_exit%"=="3010" goto restart_required
if not "%installer_exit%"=="0" goto failed
goto verify

:existing
echo Existing CUDA v12.8 directory detected. Checking it without reinstalling.
goto verify

:verify
if not exist "%vsroot%\VC\Auxiliary\Build\vcvarsall.bat" goto missing_msvc
if not exist "%cuda_root%\bin\nvcc.exe" goto missing_cuda
call "%vsroot%\VC\Auxiliary\Build\vcvarsall.bat" x64 10.0.19041.0 -vcvars_ver=14.29 > "%logdir%\vcvarsall.txt" 2>&1
if errorlevel 1 goto vcvars_failed
type "%logdir%\vcvarsall.txt"
if not defined VCToolsVersion goto failed
if not "%VCToolsVersion:~0,5%"=="14.29" goto wrong_compiler
if not "%WindowsSDKVersion%"=="10.0.19041.0\" goto wrong_compiler
set "CUDA_PATH=%cuda_root%"
set "CUDA_HOME=%cuda_root%"
set "PATH=%cuda_root%\bin;%PATH%"
> "%logdir%\toolchain_versions.txt" echo VCToolsVersion=%VCToolsVersion%
>> "%logdir%\toolchain_versions.txt" echo WindowsSDKVersion=%WindowsSDKVersion%
>> "%logdir%\toolchain_versions.txt" echo CUDA_PATH=%CUDA_PATH%
type "%logdir%\toolchain_versions.txt"
"%cuda_root%\bin\nvcc.exe" --version > "%logdir%\nvcc_version.txt" 2>&1
if errorlevel 1 goto failed
type "%logdir%\nvcc_version.txt"
rem Official CUDA 12.8.1 redistrib manifest and local version.json list NVCC 12.8.93.
findstr /l /c:"V12.8.93" "%logdir%\nvcc_version.txt" >nul
if errorlevel 1 goto wrong_cuda
for %%F in (include\cuda_runtime.h include\cuda_profiler_api.h include\cub\cub.cuh include\thrust\version.h include\cublas_v2.h include\curand_kernel.h include\cusolverDn.h include\cusparse.h lib\x64\cudart_static.lib lib\x64\cublas.lib lib\x64\curand.lib lib\x64\cusolver.lib lib\x64\cusparse.lib) do (
    if not exist "%cuda_root%\%%F" (
        echo ERROR: Required CUDA file is missing: %%F
        goto failed
    )
)
echo CUDA_DEVELOPMENT_FILES_PASS
nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader > "%logdir%\gpu_driver_after.txt" 2>&1
if errorlevel 1 goto failed
fc /b "%logdir%\gpu_driver_before.txt" "%logdir%\gpu_driver_after.txt" > "%logdir%\driver_comparison.txt" 2>&1
if errorlevel 1 goto driver_changed
echo GPU_DRIVER_UNCHANGED_PASS
echo Compiling and running a synthetic sm_120 CUDA kernel. No training is used.
"%cuda_root%\bin\nvcc.exe" -std=c++17 -arch=sm_120 --cudart=static -ccbin "%VCToolsInstallDir%bin\Hostx64\x64" -Xcompiler /utf-8 "%frcnn_root%\workspace\scripts\step02b_cuda_probe.cu" -o "%logdir%\cuda_probe.exe" > "%logdir%\compile.txt" 2>&1
if errorlevel 1 goto compile_failed
type "%logdir%\compile.txt"
"%logdir%\cuda_probe.exe" > "%logdir%\probe_output.txt" 2>&1
if errorlevel 1 goto kernel_failed
type "%logdir%\probe_output.txt"
echo Rechecking synthetic PyTorch and torchvision operations in the dedicated environment.
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step01_check_cuda.py" --output-dir "%logdir%"
if errorlevel 1 goto failed
echo STEP02B_CUDA_TOOLKIT_PASS
echo LOG_DIR=%logdir%
echo STEP02B_EXIT_CODE=0
exit /b 0

:signature_failed
type "%logdir%\installer_verification.txt"
goto failed
:vcvars_failed
type "%logdir%\vcvarsall.txt"
goto failed
:compile_failed
type "%logdir%\compile.txt"
goto failed
:kernel_failed
type "%logdir%\probe_output.txt"
goto failed
:missing_msvc
echo ERROR: Expected Build Tools 2022 installation is missing.
goto failed
:missing_cuda
echo ERROR: Expected CUDA 12.8 nvcc.exe is missing.
goto failed
:wrong_compiler
echo ERROR: Expected MSVC 14.29 and Windows SDK 10.0.19041.0.
goto failed
:wrong_cuda
echo ERROR: Expected nvcc V12.8.93 from CUDA Toolkit 12.8 Update 1.
goto failed
:driver_changed
echo ERROR: GPU driver information changed. Review before continuing.
type "%logdir%\gpu_driver_after.txt"
goto failed
:restart_required
echo STEP02B_REBOOT_REQUIRED
echo Save your work and restart Windows. Then run this script with --verify-only.
echo LOG_DIR=%logdir%
echo STEP02B_EXIT_CODE=3010
exit /b 3010
:failed
echo STEP02B_CUDA_TOOLKIT_FAIL
echo Send the complete terminal output and LOG_DIR. Do not install MMCV yet.
echo LOG_DIR=%logdir%
echo STEP02B_EXIT_CODE=1
exit /b 1

:show_plan
echo PLAN ONLY - no installer, compiler, or GPU operation is launched.
echo Toolkit: CUDA 12.8 Update 1; expected nvcc V12.8.93
echo Install location: %cuda_root%
echo Installer: %frcnn_root%\workspace\installers\cuda_12.8.1_572.61_windows.exe
echo Components: %components%
echo Display.Driver is excluded; silent mode; no automatic restart.
echo Checks: driver unchanged, nvcc sm_120 compile and kernel, dedicated PyTorch CUDA.
echo No MMCV installation and no model training in this step.
exit /b 0
