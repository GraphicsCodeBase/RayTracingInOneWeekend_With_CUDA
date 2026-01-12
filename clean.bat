@echo off
REM Clean script - removes all build artifacts and generated files

echo Cleaning build artifacts...
echo.

REM Remove build directories
if exist "build" (
    echo Removing build directory...
    rmdir /s /q "build"
)

if exist "bin" (
    echo Removing bin directory...
    rmdir /s /q "bin"
)

if exist "out" (
    echo Removing out directory...
    rmdir /s /q "out"
)

REM Remove Visual Studio cache
if exist ".vs" (
    echo Removing Visual Studio cache...
    rmdir /s /q ".vs"
)

REM Remove any stray solution/project files in root
if exist "*.sln" del /q "*.sln" 2>nul
if exist "*.vcxproj" del /q "*.vcxproj" 2>nul
if exist "*.vcxproj.filters" del /q "*.vcxproj.filters" 2>nul
if exist "*.vcxproj.user" del /q "*.vcxproj.user" 2>nul

echo.
echo ============================================
echo Cleanup complete!
echo Only source files remain.
echo Ready for git commit.
echo ============================================
echo.

pause
