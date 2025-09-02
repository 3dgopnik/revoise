from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable

from .tts_adapters import synthesize_beep as _synthesize_beep
from .tts_adapters import synthesize_silero

logger = logging.getLogger(__name__)



def synthesize_beep(text: str, speaker: str, sr: int) -> bytes:
    """Fallback beep synthesis."""
    return _synthesize_beep(sample_rate=sr)


registry: dict[str, Callable[[str, str, int], bytes]] = {
    "silero": synthesize_silero,
    "beep": synthesize_beep,
}


def get_engine(name: str | None = None) -> Callable[[str, str, int], bytes]:
    """Resolve TTS engine by name or configuration."""
    if name is None or not name:
        name = os.getenv("TTS_ENGINE")
        if not name:
            try:
                with open("config.json", encoding="utf-8") as fh:
                    cfg = json.load(fh)
                name = cfg.get("tts", {}).get("engine")
            except Exception:
                name = None
        name = name or "silero"
    key = name.lower()
    try:
        return registry[key]
    except KeyError:
        return registry["beep"]
