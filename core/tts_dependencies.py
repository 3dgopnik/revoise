from __future__ import annotations

import logging
from collections.abc import Mapping

from .pkg_installer import ensure_package

TTS_DEPENDENCIES: Mapping[str, dict[str, dict[str, list[str] | str]]] = {
    "silero": {
        "torch": {
            "install_args": ["--index-url", "https://download.pytorch.org/whl/cu118"],
            "fallback_args": ["--index-url", "https://download.pytorch.org/whl/cpu"],
        },
        "omegaconf": {},
    },
    "coqui_xtts": {
        "TTS": {},
    },
    "gtts": {
        "gtts": {"package": "gTTS"},
    },
}

logger = logging.getLogger(__name__)


def ensure_tts_dependencies(engine: str) -> None:
    """Ensure that required packages for a TTS engine are installed and importable."""
    deps = TTS_DEPENDENCIES.get(engine, {})
    for module_name, options in deps.items():
        ensure_package(
            module_name,
            package=options.get("package"),
            install_args=options.get("install_args"),
            fallback_args=options.get("fallback_args"),
        )
