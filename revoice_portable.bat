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

:: Ensure llama_cpp is installed (requires internet access and sufficient disk space)
uv run python - <<EOF
try:
    import llama_cpp  # type: ignore
except Exception:
    raise SystemExit(1)
EOF
if errorlevel 1 (
    uv add llama_cpp
    if errorlevel 1 (
        set "EXITCODE=%ERRORLEVEL%"
        echo Failed to install llama_cpp; AI editor won't work
        exit /b %EXITCODE%
    )
    uv run python - <<EOF
try:
    import llama_cpp  # type: ignore
except Exception:
    raise SystemExit(1)
EOF
    if errorlevel 1 (
        set "EXITCODE=%ERRORLEVEL%"
        echo Failed to install llama_cpp; AI editor won't work
        exit /b %EXITCODE%
    )
)

uv run python -m ui.main %*

set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo RevoicePortable failed to start. You may be missing dependencies.
    exit /b %EXITCODE%
)

