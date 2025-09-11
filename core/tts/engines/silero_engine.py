from __future__ import annotations

import importlib.util

import numpy as np

from ...tts_adapters import SILERO_VOICES, SileroTTS
from .base import TTSEngineBase


class SileroEngine(TTSEngineBase):
    """Silero TTS wrapper."""

    def __init__(self, language: str = "ru", auto_download: bool = True) -> None:
        self.language = language
        self.auto_download = auto_download
        self._impl: SileroTTS | None = None

    def load(self) -> None:
        self._impl = SileroTTS(auto_download=self.auto_download, language=self.language)

    def synthesize(self, text: str, speaker: str, sample_rate: int) -> np.ndarray:
        if self._impl is None:
            self.load()
        assert self._impl is not None
        wav = self._impl.tts(text, speaker, sr=sample_rate)
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
            and importlib.util.find_spec("omegaconf") is not None
        )
