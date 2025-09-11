from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from ...pkg_installer import ensure_package
from ...tts_adapters import SILERO_VOICES, SileroTTS
from .base import TTSEngineBase


class SileroEngine(TTSEngineBase):
    """Silero TTS wrapper."""

    def __init__(self, language: str = "ru", auto_download: bool = True) -> None:
        self.language = language
        self.auto_download = auto_download
        self._impl: SileroTTS | None = None

    def load(self) -> None:
        try:
            ensure_package("torch", "torch is required for Silero TTS.")
            ensure_package("torchaudio", "torchaudio is required for Silero TTS.")
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency handling
            raise RuntimeError(str(exc)) from exc
        self._impl = SileroTTS(auto_download=self.auto_download, language=self.language)

    def synthesize(
        self,
        text: str,
        speaker: str,
        sample_rate: int,
        rate: float | None = None,
        pitch: float | None = None,
        style: str | None = None,
        preset: str | None = None,
        **_: Any,
    ) -> np.ndarray:
        if self._impl is None:
            self.load()
        assert self._impl is not None
        if rate is not None and not (0.5 <= rate <= 2.0):
            raise ValueError("rate must be between 0.5 and 2.0")
        if pitch is not None and not (-10.0 <= pitch <= 10.0):
            raise ValueError("pitch must be between -10.0 and 10.0")
        wav = self._impl.tts(
            text,
            speaker,
            sr=sample_rate,
            rate=rate,
            pitch=pitch,
            style=style,
            preset=preset,
        )
        return np.asarray(wav, dtype=np.float32)

    def unload(self) -> None:
        self._impl = None

    @classmethod
    def supports_language(cls, lang: str) -> bool:
        return lang in SILERO_VOICES

    @classmethod
    def is_available(cls) -> bool:
        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("torchaudio") is not None
            and importlib.util.find_spec("omegaconf") is not None
        )
