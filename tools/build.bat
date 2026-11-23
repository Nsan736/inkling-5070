@echo off
rem Build strata-inkling2.exe and inkling2_logits_dump.exe into build\bin.
rem Needs: Visual Studio 2022 or newer with "Desktop development with C++", the CUDA Toolkit 13.x, CMake 3.24+ and
rem Ninja (both come with Visual Studio).  Run from any prompt: it finds Visual Studio with vswhere.
rem   tools\build.bat            release build: CUDA 7.5 / 8.6 / 8.9 / 12.0 (+ PTX), CPU code for AVX2
rem   tools\build.bat --native   this PC's GPU and CPU only (much faster to build)
rem   set JOBS=8 to build with fewer parallel jobs (default 12)
setlocal
cd /d "%~dp0.."
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo ERROR: Visual Studio was not found
  exit /b 1
)
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%i"
if not defined VS (
  echo ERROR: Visual Studio with the C++ tools was not found
  exit /b 1
)
rem The header dependencies: Ninja reads cl's /showIncludes lines by the prefix the configure saw, and cl writes them
rem in the console's code page (a Japanese cl: cp932 under 932, UTF-8 under 65001).  A build under another code page
rem than its configure records no headers, and a changed header rebuilds nothing.  So the code page is UTF-8 for
rem both, and tools\ninja_deps_check.py repairs a build directory configured under another one.
chcp 65001 >nul
set VSLANG=1033
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not defined JOBS set JOBS=12
set "BDIR=build"
set "OPTS=-DINK_NATIVE=OFF"
if /i "%~1"=="--native" (
  set "BDIR=build-native"
  set "OPTS=-DINK_NATIVE=ON -DCMAKE_CUDA_ARCHITECTURES=native"
)
where python >nul 2>nul && python -I tools\ninja_deps_check.py "%BDIR%"
if not exist "%BDIR%\build.ninja" (
  cmake -S . -B "%BDIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release %OPTS% || exit /b 2
)
cmake --build "%BDIR%" -j %JOBS% --target strata-inkling2 inkling2_logits_dump || exit /b 3
if /i "%BDIR%"=="build-native" (
  if not exist build\bin mkdir build\bin
  copy /y "%BDIR%\bin\*.exe" build\bin\ >nul
)
echo Built: build\bin\strata-inkling2.exe
