"""Configuration helpers for Revoise."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EngineSettings:
    """Common parameters for TTS engines."""

    model: str | None = None
    device: str = "cpu"
    attention_backend: str = "sdpa"
    quantization: str = "none"
    voices: list[str] = field(default_factory=list)


@dataclass
class TTSSection:
    """TTS configuration block."""

    default_engine: str = "silero"
    vibevoice: EngineSettings = field(default_factory=EngineSettings)
    silero: EngineSettings = field(default_factory=EngineSettings)


@dataclass
class Config:
    """Root configuration object."""

    tts: TTSSection = field(default_factory=TTSSection)


def load_config(path: str | Path = "config.json") -> Config:
    """Load configuration from ``path``.

    Only the ``tts`` section is parsed into dataclasses; unknown keys are
    ignored. Missing values fall back to sensible defaults.
    """

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    tts_raw = raw.get("tts", {})
    tts_cfg = TTSSection(
        default_engine=tts_raw.get("default_engine", "silero"),
        vibevoice=EngineSettings(**tts_raw.get("vibevoice", {})),
        silero=EngineSettings(**tts_raw.get("silero", {})),
    )

    return Config(tts=tts_cfg)


__all__ = ["Config", "TTSSection", "EngineSettings", "load_config"]
