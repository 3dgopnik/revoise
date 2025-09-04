from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Mapping

TTS_DEPENDENCIES: Mapping[str, dict[str, str]] = {
    "silero": {
        "torch": "pip install torch --index-url https://download.pytorch.org/whl/cpu",
        "omegaconf": "pip install omegaconf",
    },
    "coqui_xtts": {
        "TTS": "pip install TTS",
    },
    "gtts": {
        "gtts": "pip install gTTS",
    },
}


def ensure_tts_dependencies(engine: str) -> None:
    """Ensure that required packages for a TTS engine are installed and importable.

    For example, ``ensure_tts_dependencies("silero")`` installs ``torch`` and ``omegaconf``.
    """
    deps = TTS_DEPENDENCIES.get(engine, {})
    for module_name, install_cmd in deps.items():
        try:
            importlib.import_module(module_name)
            continue
        except ModuleNotFoundError as exc:
            try:
                subprocess.run(
                    [sys.executable, "-m", *install_cmd.split()],
                    check=True,
                    capture_output=True,
                )
                importlib.import_module(module_name)
            except Exception:
                raise RuntimeError(
                    f"{engine} requires the '{module_name}' package. Install it via `{install_cmd}`"
                ) from exc
