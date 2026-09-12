@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv >nul 2>&1
for /f "delims=" %%i in ('".venv\Scripts\python.exe" -c "import sysconfig; print(sysconfig.get_path('purelib'))"') do set SITE=%%i
if not exist "%SITE%\PySide6" (
  ".venv\Scripts\python.exe" first_install.py requirements.txt
  if errorlevel 1 exit /b 1
) else (
  ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
)
set PYTHONPATH=%~dp0
".venv\Scripts\python.exe" -m lyra %*
