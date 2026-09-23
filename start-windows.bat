@echo off
rem Double-click this file to start the app.
rem The first run installs what it needs (about two minutes); after that it
rem starts in a few seconds.

cd /d "%~dp0"
title Ask your data

where py >nul 2>nul
if %errorlevel%==0 (set "PY=py -3") else (set "PY=python")

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if errorlevel 1 (
    echo.
    echo   Python 3.11 or newer is needed and was not found.
    echo   Install it from https://www.python.org/downloads/
    echo   and tick "Add python.exe to PATH" during setup, then run this again.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   First run: setting things up. This takes about two minutes.
    echo.
    %PY% -m venv .venv || goto :failed
    ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
    ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt || goto :failed
)

".venv\Scripts\python.exe" app.py
pause
exit /b 0

:failed
echo.
echo   Setup did not finish. Check your internet connection and run this again.
echo   If it keeps failing, delete the .venv folder and try once more.
echo.
pause
exit /b 1
