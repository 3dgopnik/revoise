from __future__ import annotations

import json
import logging
import os
import wave
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import numpy as np

from .tts_adapters import SileroTTS, synthesize_beep as _synthesize_beep

logger = logging.getLogger(__name__)



def synthesize_beep(text: str, speaker: str, sr: int) -> bytes:
    """Fallback beep synthesis."""
    return _synthesize_beep(sample_rate=sr)


def synthesize_silero(text: str, speaker: str, sr: int) -> bytes:
    wav = SileroTTS(Path(__file__).resolve().parent.parent).tts(text, speaker, sr)
    pcm16 = (np.clip(wav, -1.0, 1.0) * 32767).astype("<i2")
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


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
