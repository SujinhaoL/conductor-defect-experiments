@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0..\.."
if not exist "D:\anaconda3\python.exe" (
    echo ERROR: D:\anaconda3\python.exe was not found.
    exit /b 1
)
rem Management interpreter, isolated mode, Python standard library only.
rem Never imports another model environment, installs packages, or starts training.
"D:\anaconda3\python.exe" -I -X utf8 "workspace\scripts\step00_probe.py" --dataset
set "probe_exit=%ERRORLEVEL%"
echo PROBE_EXIT_CODE=%probe_exit%
exit /b %probe_exit%
