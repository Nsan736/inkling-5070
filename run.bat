@echo off
rem Inkling 2 Mini: start the OpenAI-compatible server on 127.0.0.1 (port from config\inkling.json).
rem   run.bat                 start
rem   run.bat --port 8081     another port this time
rem   run.bat --api-key KEY   require a key
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: run setup.bat first
  exit /b 1
)
".venv\Scripts\python.exe" -I scripts\run.py %*
exit /b %errorlevel%
