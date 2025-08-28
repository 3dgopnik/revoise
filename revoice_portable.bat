@echo off
setlocal enabledelayedexpansion
REM Add CUDA toolkit to PATH if present
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin" (
  set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin;%PATH%"
  echo CUDA 12.0 Toolkit found and added to PATH.
) else (
  echo Warning: CUDA 12.0 Toolkit not found at C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin. Continuing without it.
)
REM Activate uv env and run UI
where uv >nul 2>nul
if errorlevel 1 (
  echo uv not found. Install from https://docs.astral.sh/uv/
  pause
  exit /b 1
)
uv sync
uv run python -m ui.main_window
