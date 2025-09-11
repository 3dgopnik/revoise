from __future__ import annotations

from typing import Any

import numpy as np

from ...tts_adapters import BeepTTS
from .base import TTSEngineBase


class BeepEngine(TTSEngineBase):
    """Simple engine that generates a beep sound."""

    def __init__(self) -> None:
        self._impl: BeepTTS | None = None

    def load(self) -> None:
        self._impl = BeepTTS()

    def synthesize(self, text: str, speaker: str, sample_rate: int, **kwargs: Any) -> np.ndarray:
        if self._impl is None:
            self.load()
        assert self._impl is not None
        return self._impl.tts(text, speaker, sr=sample_rate)

    def unload(self) -> None:
        self._impl = None

    @classmethod
    def supports_language(cls, lang: str) -> bool:  # noqa: D401 - short
        return True

    @classmethod
    def is_available(cls) -> bool:  # noqa: D401 - short
        return True
