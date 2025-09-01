@echo off
chcp 65001 > nul
cd /d %~dp0

@if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo Virtual environment not found ^& exit /b 1
)

if exist "%CD%\bin" (
    set "PATH=%CD%\bin;%PATH%"
)

if /I "%TTS_ENGINE%"=="beep" (
    set "TTS_ENGINE="
)

echo Starting RevoicePortable...

if exist ui\main.py (
    python -m ui.main %*
) else (
    python main.py %*
)

set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo RevoicePortable failed to start. You may be missing dependencies.
    exit /b %EXITCODE%
)

