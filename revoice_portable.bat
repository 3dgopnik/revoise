@echo off
chcp 65001 > nul
cd /d %~dp0

set "PATH=%CD%\bin;%PATH%"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin;%PATH%"
set "CT2_FORCE_CPU="

if not exist "%CD%\hf_cache" mkdir "%CD%\hf_cache"
set "HF_HOME=%CD%\hf_cache"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%"
set "TRANSFORMERS_CACHE=%HF_HOME%"

where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    where uv
    echo uv is required but was not found in PATH. See https://astral.sh/uv for installation instructions.
    exit /b 1
)

echo Starting RevoicePortable...
uv run python -m ui.main %*

