@echo off
rem Build strata-vision.exe (the image encoder, CPU only: no VRAM) into build-vision\bin.
rem It is PolyStrata's tools\vision (MIT) on llama.cpp's mtmd library, which the engine's third_party tree does not
rem carry: give a llama.cpp checkout at 690b87d88 (or later with Inkling 2's encoder, inkling2v) as LLAMA_DIR.
rem   set LLAMA_DIR=C:\src\llama.cpp
rem   tools\build_vision.bat            portable: AVX2 baseline, static C++ runtime (what the release zip carries)
rem   tools\build_vision.bat --native   this PC's CPU only
setlocal
cd /d "%~dp0.."
if not defined LLAMA_DIR (
  echo ERROR: set LLAMA_DIR to a llama.cpp checkout ^(690b87d88 or later^)
  exit /b 1
)
if not exist "%LLAMA_DIR%\tools\mtmd\mtmd.h" (
  echo ERROR: %LLAMA_DIR% has no tools\mtmd\mtmd.h
  exit /b 1
)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%i"
if not defined VS (
  echo ERROR: Visual Studio with the C++ tools was not found
  exit /b 1
)
rem the code page as in build.bat, so that Ninja records the header dependencies
chcp 65001 >nul
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not defined JOBS set JOBS=12
set "BDIR=build-vision"
set "PORTABLE=ON"
if /i "%~1"=="--native" set "PORTABLE=OFF"
where python >nul 2>nul && python -I tools\ninja_deps_check.py "%BDIR%"
if not exist "%BDIR%\build.ninja" (
  cmake -S tools\vision -B "%BDIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="%LLAMA_DIR%" -DSTRATA_PORTABLE=%PORTABLE% || exit /b 2
)
cmake --build "%BDIR%" -j %JOBS% --target strata-vision || exit /b 3
echo Built: %BDIR%\bin\strata-vision.exe
