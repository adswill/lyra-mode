@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%~dp0
py -3 -m lyra.check_online %*
