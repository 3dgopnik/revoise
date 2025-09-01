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

echo Starting RevoicePortable...
uv run python -m ui.main %*

