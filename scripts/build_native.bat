@echo off
:: Build the native CUDA library `mat_mul.dll` on Windows.
:: Requires: Visual Studio Build Tools + CUDA Toolkit, both on PATH.
::
:: Usage: scripts\build_native.bat

setlocal
pushd "%~dp0\.."

where nvcc >nul 2>nul
if errorlevel 1 (
    echo nvcc not found on PATH. Install the CUDA toolkit first.
    exit /b 1
)

if not exist native\lib mkdir native\lib

nvcc --shared -O3 -o native\lib\mat_mul.dll lib\native\src\engine.cu
if errorlevel 1 exit /b 1

dir native\lib\mat_mul.dll
popd
endlocal
