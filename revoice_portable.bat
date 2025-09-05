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
set "LLAMA_DIR=%CD%\llama_cpp_pkg"
if exist "%LLAMA_DIR%" (
    set "PYTHONPATH=%LLAMA_DIR%;%PYTHONPATH%"
)
uv run python -c "import llama_cpp  # type: ignore"
if errorlevel 1 (
    uv pip install llama-cpp-python --target "%LLAMA_DIR%"
    if errorlevel 1 (
        set "EXITCODE=%ERRORLEVEL%"
        echo Failed to install llama-cpp-python; AI editor won't work
        exit /b %EXITCODE%
    )
    set "PYTHONPATH=%LLAMA_DIR%;%PYTHONPATH%"
    uv run python -c "import llama_cpp  # type: ignore"
    if errorlevel 1 (
        set "EXITCODE=%ERRORLEVEL%"
        echo Failed to install llama-cpp-python; AI editor won't work
        exit /b %EXITCODE%
    )
)

uv run python tools/bootstrap_portable.py || exit /b %ERRORLEVEL%
uv run python -m ui.main %*

set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo RevoicePortable failed to start. You may be missing dependencies.
    exit /b %EXITCODE%
)

