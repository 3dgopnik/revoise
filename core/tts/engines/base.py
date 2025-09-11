from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class TTSEngineBase(ABC):
    """Abstract base class for TTS engines."""

    @abstractmethod
    def load(self) -> None:
        """Load engine resources."""

    @abstractmethod
    def synthesize(self, text: str, speaker: str, sample_rate: int, **kwargs: Any) -> np.ndarray:
        """Synthesize *text* with *speaker* at *sample_rate* and return waveform."""

    @abstractmethod
    def unload(self) -> None:
        """Release any held resources."""

    @classmethod
    @abstractmethod
    def supports_language(cls, lang: str) -> bool:
        """Return ``True`` if the engine supports *lang*."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return ``True`` if the engine's dependencies are available."""
