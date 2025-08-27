@echo off
setlocal enabledelayedexpansion
REM Activate uv env and run UI
where uv >nul 2>nul
if errorlevel 1 (
  echo uv not found. Install from https://docs.astral.sh/uv/
  pause
  exit /b 1
)
uv sync
uv run python -m ui.main_window
