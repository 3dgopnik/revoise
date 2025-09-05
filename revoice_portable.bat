@echo off
chcp 65001 > nul
cd /d %~dp0

where uv >nul 2>&1 || (echo uv not found. Please install uv and try again. & exit /b 1)

for /f "tokens=2" %%v in ('uv python --version') do set "UV_PY=%%v"
if not "%UV_PY:~0,4%"=="3.11" (
    uv python install 3.11 || exit /b %ERRORLEVEL%
    uv venv --python 3.11 || exit /b %ERRORLEVEL%
)

uv sync --frozen --no-dev
if errorlevel 1 (
    echo Failed to install dependencies.
    echo If this relates to PyTorch, reinstall with:
    echo    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    exit /b %ERRORLEVEL%
)

if exist "%CD%\bin" (
    set "PATH=%CD%\bin;%PATH%"
)

set "HF_HOME=%CD%\hf_cache"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%"
set "TRANSFORMERS_CACHE=%HF_HOME%"

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

