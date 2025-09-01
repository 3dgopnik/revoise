from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from collections.abc import Callable

import numpy as np

from .tts_adapters import BeepTTS, SileroTTS

logger = logging.getLogger(__name__)


def synthesize_silero(text: str, speaker: str, sr: int) -> np.ndarray:
    """Synthesize speech via Silero TTS."""
    return SileroTTS(Path(__file__).resolve().parent.parent).tts(text, speaker, sr=sr)


def synthesize_beep(text: str, speaker: str, sr: int) -> np.ndarray:
    """Fallback beep synthesis."""
    return BeepTTS().tts(text, speaker, sr=sr)


registry: dict[str, Callable[[str, str, int], np.ndarray]] = {
    "silero": synthesize_silero,
    "beep": synthesize_beep,
}


def get_engine(name: str | None = None) -> Callable[[str, str, int], np.ndarray]:
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
