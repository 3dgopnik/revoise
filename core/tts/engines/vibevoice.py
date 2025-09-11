from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import soundfile as sf

from .base import TTSEngineBase


@dataclass
class VibeVoiceOptions:
    """Options for :class:`VibeVoiceEngine`."""

    executable: str = "vibe-voice"
    model_path: str | None = None


class VibeVoiceEngine(TTSEngineBase):
    """Thin wrapper around the external ``vibe-voice`` binary."""

    def __init__(self, options: VibeVoiceOptions | None = None) -> None:
        self.options = options or VibeVoiceOptions()

    def load(self) -> None:  # noqa: D401 - simple
        pass

    def synthesize(self, text: str, speaker: str, sample_rate: int, **kwargs: Any) -> np.ndarray:
        cmd = [
            self.options.executable,
            "--text",
            text,
            "--speaker",
            speaker,
            "--sample-rate",
            str(sample_rate),
            "--output",
            "-",
        ]
        if self.options.model_path:
            cmd += ["--model-path", self.options.model_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
        data, _ = sf.read(BytesIO(result.stdout), dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        return data

    def unload(self) -> None:  # noqa: D401 - simple
        pass

    @classmethod
    def supports_language(cls, lang: str) -> bool:  # noqa: D401 - simple
        return True

    @classmethod
    def is_available(cls) -> bool:  # noqa: D401 - simple
        return shutil.which("vibe-voice") is not None
