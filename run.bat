@echo off
rem Inkling 2 Mini: start the OpenAI-compatible server on 127.0.0.1 (port from config\inkling.json).
rem   run.bat                 start (the first time, without settings, it runs setup.bat first)
rem   run.bat --port 8081     another port this time
rem   run.bat --api-key KEY   require a key
rem It starts with the settings in config\inkling.json and downloads nothing.
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" if exist "config\inkling.json" goto start
echo No settings yet: running setup.bat first.
call "%~dp0setup.bat" || exit /b 1
if not exist "config\inkling.json" (
  echo ERROR: setup.bat wrote no config\inkling.json
  exit /b 1
)
:start
".venv\Scripts\python.exe" -I scripts\run.py %*
exit /b %errorlevel%
