@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0..\.."
if errorlevel 1 exit /b 1
set "frcnn_root=%CD%"
set "vsroot=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "sdkversion=10.0.19041.0"
set "VSLANG=1033"
if /i "%~1"=="--show-plan" goto show_plan
set "logdir=%frcnn_root%\workspace\logs\step02a_%RANDOM%_%RANDOM%"
mkdir "%logdir%"
if errorlevel 1 exit /b 1
echo STEP02A_MSVC_SETUP
echo LOG_DIR=%logdir%
if /i "%~1"=="--verify-only" goto verify
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step01_check_cuda.py" --identity-only
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step02a_prepare_msvc.py" --require-admin
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -I -X utf8 "workspace\scripts\step02a_prepare_msvc.py" --verify-installer > "%logdir%\installer_verification.txt" 2>&1
if errorlevel 1 goto signature_failed
type "%logdir%\installer_verification.txt"

rem An existing instance must be reviewed before changing its installed components.
if exist "%vsroot%\VC\Auxiliary\Build\vcvarsall.bat" goto existing
echo Installing Microsoft Build Tools 2022, MSVC v142, and Windows SDK 19041.
echo The Microsoft installer displays progress. Automatic restart is disabled.
start "" /wait "%frcnn_root%\workspace\installers\vs_buildtools_2022.exe" --wait --passive --norestart --installPath "%vsroot%" --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.ComponentGroup.VC.Tools.142.x86.x64 --add Microsoft.VisualStudio.Component.Windows10SDK.19041 --addProductLang en-US
set "installer_exit=%ERRORLEVEL%"
> "%logdir%\installer_exit_code.txt" echo %installer_exit%
echo MICROSOFT_INSTALLER_EXIT_CODE=%installer_exit%
if "%installer_exit%"=="3010" goto restart_required
if not "%installer_exit%"=="0" goto failed
goto verify

:existing
echo Existing Build Tools instance detected. Checking it without changing components.
goto verify

:verify
if not exist "%vsroot%\VC\Auxiliary\Build\vcvarsall.bat" goto missing_tools
echo Selecting the 14.29 x64 compiler and SDK %sdkversion% for this cmd process.
call "%vsroot%\VC\Auxiliary\Build\vcvarsall.bat" x64 %sdkversion% -vcvars_ver=14.29 > "%logdir%\vcvarsall.txt" 2>&1
if errorlevel 1 goto vcvars_failed
type "%logdir%\vcvarsall.txt"
where cl > "%logdir%\compiler_path.txt" 2>&1
if errorlevel 1 goto failed
type "%logdir%\compiler_path.txt"
if not defined VCToolsVersion goto failed
if /i not "%VCToolsVersion:~0,5%"=="14.29" goto wrong_compiler
if not "%WindowsSDKVersion%"=="%sdkversion%\" goto wrong_sdk
> "%logdir%\toolchain_versions.txt" echo VCToolsVersion=%VCToolsVersion%
>> "%logdir%\toolchain_versions.txt" echo WindowsSDKVersion=%WindowsSDKVersion%
>> "%logdir%\toolchain_versions.txt" echo VSCMD_ARG_TGT_ARCH=%VSCMD_ARG_TGT_ARCH%
type "%logdir%\toolchain_versions.txt"
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -all -products * -format json -utf8 > "%logdir%\vswhere.json"
echo Compiling and running a small C++ check. No Python model or training is used.
cl /nologo /Bv /EHsc /std:c++17 /utf-8 "%frcnn_root%\workspace\scripts\step02a_msvc_probe.cpp" /Fo"%logdir%\msvc_probe.obj" /Fe"%logdir%\msvc_probe.exe" /link kernel32.lib > "%logdir%\compile.txt" 2>&1
if errorlevel 1 goto compile_failed
type "%logdir%\compile.txt"
"%logdir%\msvc_probe.exe" > "%logdir%\probe_output.txt" 2>&1
if errorlevel 1 goto failed
type "%logdir%\probe_output.txt"
echo STEP02A_MSVC_PASS
echo LOG_DIR=%logdir%
echo STEP02A_EXIT_CODE=0
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
:missing_tools
echo ERROR: Expected Build Tools 2022 installation was not found at %vsroot%.
goto failed
:wrong_compiler
echo ERROR: Expected VCToolsVersion 14.29, received %VCToolsVersion%.
goto failed
:wrong_sdk
echo ERROR: Expected SDK %sdkversion%, received %WindowsSDKVersion%.
goto failed
:restart_required
echo STEP02A_REBOOT_REQUIRED
echo Save your work and restart Windows. Then run this script with --verify-only.
echo LOG_DIR=%logdir%
echo STEP02A_EXIT_CODE=3010
exit /b 3010
:failed
echo STEP02A_MSVC_FAIL
echo Send the complete terminal output and LOG_DIR. Do not install CUDA Toolkit yet.
echo LOG_DIR=%logdir%
echo STEP02A_EXIT_CODE=1
exit /b 1

:show_plan
echo PLAN ONLY - no installer is launched and no code is compiled.
echo Product: Microsoft Visual Studio Build Tools 2022
echo Install location: %vsroot%
echo Components: C++ build workload, MSVC v142 14.29, Windows SDK %sdkversion%
echo Compiler selection: vcvarsall x64 %sdkversion% -vcvars_ver=14.29
echo Installer: %frcnn_root%\workspace\installers\vs_buildtools_2022.exe
echo Mode: progress UI, no automatic restart
echo Validation: compile and run the C++ probe after installation
echo No CUDA Toolkit, Python package installation, or training in this step.
exit /b 0
