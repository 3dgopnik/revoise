#!/usr/bin/env bash
set -e
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install: https://docs.astral.sh/uv/"
  exit 1
fi
uv sync --frozen --no-dev
uv run python tools/bootstrap_portable.py
uv run python -m ui.main
