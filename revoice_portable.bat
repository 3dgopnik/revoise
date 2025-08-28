@echo off
chcp 65001 > nul
cd /d %~dp0

:: наши DLL (cuDNN) и ffmpeg
set PATH=%CD%\bin;%PATH%

:: CUDA 12.0 runtime (нужны cudart64_12.dll, cublas*.dll)
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin;%PATH%

:: если раньше ставили — убираем принудительный CPU
set CT2_FORCE_CPU=

:: кэш моделей HF
set HF_HOME=%CD%\hf_cache
set HUGGINGFACE_HUB_CACHE=%HF_HOME%
set TRANSFORMERS_CACHE=%HF_HOME%

echo Starting RevoicePortable...
uv run python -m ui.main %*
