@echo off
chcp 65001 > nul
cd /d %~dp0\..
set HF_HOME=%CD%\hf_cache
set HUGGINGFACE_HUB_CACHE=%HF_HOME%
set TRANSFORMERS_CACHE=%HF_HOME%
echo === Downloading STT models to models\stt ===
uv run python tools\fetch_stt_models.py %*
echo Done.
pause
