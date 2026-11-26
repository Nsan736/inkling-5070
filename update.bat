@echo off
rem Inkling 2 Mini: update to the newest release of Nsan736/inkling-5070 on GitHub.
rem Keeps the model (models\), the settings (config\), .venv, the logs and the saved conversations.
rem The program as it was goes to old\<version>, so that it can be put back.
rem   update.bat              check, and update if a newer release is out
rem   update.bat --check      only say whether a newer release is out
rem   update.bat --rollback   put back the newest version kept in old\
rem Stop the server (close run.bat's window) first.
setlocal
cd /d "%~dp0"
if not exist "scripts\update.ps1" (
  echo ERROR: scripts\update.ps1 is missing
  exit /b 1
)
if not exist "update_tmp" mkdir "update_tmp"
copy /y "scripts\update.ps1" "update_tmp\update.ps1" >nul || exit /b 1
rem Windows PowerShell with its own modules (a PSModulePath from PowerShell 7 hides some of them)
set "PSModulePath="
rem One line: this file may be replaced while PowerShell runs, and cmd reads a batch file line by line;
rem call expands errorlevel when it runs, after PowerShell
powershell -NoProfile -ExecutionPolicy Bypass -File "update_tmp\update.ps1" -Root "%~dp0." %* & call exit /b %%errorlevel%%
