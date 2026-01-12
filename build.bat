@echo off
REM Build script for CUDA CMake project

REM Create build directory if it doesn't exist
if not exist "build" mkdir build

REM Navigate to build directory
cd build

REM Run CMake to generate Visual Studio solution
REM Adjust the Visual Studio version as needed:
REM   -G "Visual Studio 17 2022" for VS 2022
REM   -G "Visual Studio 16 2019" for VS 2019
REM   -G "Visual Studio 15 2017" for VS 2017

cmake -G "Visual Studio 17 2022" -A x64 ..

REM Check if CMake succeeded
if %errorlevel% neq 0 (
    echo CMake configuration failed!
    cd ..
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================
echo Build files generated successfully!
echo Solution file: build\RayTracingCUDA.sln
echo ============================================
echo.

cd ..
pause
