@echo off
rem Inkling 2 Mini: check the PC, download the model (about 15.7 GB), choose the settings.
rem   setup.bat                                   everything
rem   setup.bat --configure-only                  choose the settings again
rem   setup.bat --limit-ram-gb 16 --limit-vram-gb 8   cautious settings for a smaller PC
rem   setup.bat --port 8081                       the server's port (default 8080)
setlocal
cd /d "%~dp0"
set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if defined PY goto have_python
where python >nul 2>nul
if not errorlevel 1 set "PY=python"
:have_python
if not defined PY (
  echo ERROR: Python 3.10 or newer is needed. Install it from https://www.python.org/ or with
  echo        winget install Python.Python.3.12
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating the Python environment in .venv ...
  %PY% -m venv .venv || exit /b 1
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt || exit /b 1
".venv\Scripts\python.exe" -I scripts\setup.py %*
exit /b %errorlevel%
