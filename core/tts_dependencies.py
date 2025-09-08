from __future__ import annotations

from collections.abc import Mapping, Sequence

from .pkg_installer import ensure_package

TTS_DEPENDENCIES: Mapping[str, Sequence[str]] = {
    "silero": ["torch", "omegaconf"],
    "coqui_xtts": ["TTS"],
    "gtts": ["gTTS"],
}


def ensure_tts_dependencies(engine: str) -> None:
    """Ensure that required packages for a TTS engine are installed and importable."""

    deps = TTS_DEPENDENCIES.get(engine, [])
    for pkg in deps:
        ensure_package(pkg, f"{pkg} is required for {engine}.")

