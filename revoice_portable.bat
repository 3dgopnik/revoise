@echo off
chcp 65001 > nul
cd /d %~dp0

where uv >nul 2>&1 || (echo uv not found. Please install uv and try again. & exit /b 1)

uv sync || exit /b %ERRORLEVEL%

if exist "%CD%\bin" (
    set "PATH=%CD%\bin;%PATH%"
)

if /I "%TTS_ENGINE%"=="beep" (
    set "TTS_ENGINE="
)

echo Starting RevoicePortable...

uv run python -m ui.main %*

set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo RevoicePortable failed to start. You may be missing dependencies.
    exit /b %EXITCODE%
)

