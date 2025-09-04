from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model_manager import ensure_model

# Cache for resolved model paths
_MODEL_PATH_CACHE: dict[tuple[str, str], Path] = {}


def _auto_download_enabled() -> bool:
    try:
        with open("config.json", encoding="utf-8") as fh:
            data = json.load(fh)
        return bool(data.get("auto_download_models", True))
    except Exception:
        return True


def get_model_path(
    name: str,
    category: str,
    *,
    parent: Any | None = None,
    auto_download: bool | None = None,
) -> Path:
    key = (name, category)
    path = _MODEL_PATH_CACHE.get(key)
    if path is None:
        if auto_download is None:
            auto_download = _auto_download_enabled()
        path = ensure_model(name, category, parent=parent, auto_download=auto_download)
        _MODEL_PATH_CACHE[key] = path
    return path


def clear_cache() -> None:
    _MODEL_PATH_CACHE.clear()


__all__ = ["get_model_path", "clear_cache"]
