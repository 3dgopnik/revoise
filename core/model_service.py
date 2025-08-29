from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from .model_manager import ensure_model

_MODEL_PATH_CACHE: Dict[Tuple[str, str], Path] = {}


def get_model_path(
    name: str,
    category: str,
    *,
    parent: Any | None = None,
    auto_download: bool = False,
) -> Path:
    key = (name, category)
    path = _MODEL_PATH_CACHE.get(key)
    if path is None:
        path = ensure_model(name, category, parent=parent, auto_download=auto_download)
        _MODEL_PATH_CACHE[key] = path
    return path


def clear_cache() -> None:
    _MODEL_PATH_CACHE.clear()


__all__ = ["get_model_path", "clear_cache"]
