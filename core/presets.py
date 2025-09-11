from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_presets(dir_path: Path) -> dict[str, dict[str, Any]]:
    """Load `.rvpreset` files from *dir_path*.

    Each preset file should contain a JSON object with optional keys:
    ``rate``, ``pitch``, ``style`` and ``preset``.
    The file name (without extension) is used as the preset name.
    Invalid files are ignored.
    """
    presets: dict[str, dict[str, Any]] = {}
    if not dir_path.exists():
        return presets
    for path in dir_path.glob("*.rvpreset"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                presets[path.stem] = data
        except Exception:
            continue
    return presets
