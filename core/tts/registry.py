from __future__ import annotations

import json
import os

from .engines import BeepEngine, SileroEngine, TTSEngineBase, VibeVoiceEngine

registry: dict[str, type[TTSEngineBase]] = {
    "silero": SileroEngine,
    "beep": BeepEngine,
    "vibevoice": VibeVoiceEngine,
}

_loaded: dict[str, TTSEngineBase] = {}


def register_engine(name: str, engine_cls: type[TTSEngineBase]) -> None:
    registry[name.lower()] = engine_cls


def get_engine(name: str | None = None) -> TTSEngineBase:
    if not name:
        name = os.getenv("TTS_ENGINE")
        if not name:
            try:
                with open("config.json", encoding="utf-8") as fh:
                    cfg = json.load(fh)
                name = cfg.get("tts_engine") or cfg.get("tts", {}).get("engine")
            except Exception:
                name = None
        name = name or "silero"
    key = name.lower()
    cls = registry.get(key, registry["beep"])
    if key not in _loaded:
        engine = cls()
        engine.load()
        _loaded[key] = engine
    return _loaded[key]


def available_engines() -> list[str]:
    return sorted(registry.keys())


def health_check() -> dict[str, bool]:
    return {name: cls.is_available() for name, cls in registry.items()}
