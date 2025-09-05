from __future__ import annotations

import importlib
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

TTS_DEPENDENCIES: Mapping[str, dict[str, str]] = {
    "silero": {
        "torch": "uv pip install torch --index-url https://download.pytorch.org/whl/cu118",
        "omegaconf": "uv pip install omegaconf",
    },
    "coqui_xtts": {
        "TTS": "uv pip install TTS",
    },
    "gtts": {
        "gtts": "uv pip install gTTS",
    },
}

TTS_PKG_DIR = Path(os.getenv("REVOISE_TTS_PKG_DIR", ".portable_pkgs"))
if str(TTS_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(TTS_PKG_DIR))


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
                TTS_PKG_DIR.mkdir(parents=True, exist_ok=True)
                if module_name == "torch":
                    try:
                        subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "uv",
                                "pip",
                                "install",
                                "torch",
                                "--index-url",
                                "https://download.pytorch.org/whl/cu118",
                                f"--target={TTS_PKG_DIR}",
                            ],
                            check=True,
                            capture_output=True,
                        )
                    except Exception:
                        try:
                            subprocess.run(
                                [
                                    sys.executable,
                                    "-m",
                                    "uv",
                                    "pip",
                                    "install",
                                    "torch",
                                    "--index-url",
                                    "https://download.pytorch.org/whl/cpu",
                                    f"--target={TTS_PKG_DIR}",
                                ],
                                check=True,
                                capture_output=True,
                            )
                        except Exception:
                            raise RuntimeError(
                                "Failed to install torch. Check your Python version, reinstall the torch CPU package, or clear the uv cache."
                            ) from exc
                else:
                    cmd_parts = install_cmd.split()
                    insert_at = cmd_parts.index("install") + 1
                    cmd_parts.insert(insert_at, f"--target={TTS_PKG_DIR}")
                    subprocess.run(
                        [sys.executable, "-m", *cmd_parts],
                        check=True,
                        capture_output=True,
                    )
                if str(TTS_PKG_DIR) not in sys.path:
                    sys.path.insert(0, str(TTS_PKG_DIR))
                importlib.import_module(module_name)
            except Exception:
                raise RuntimeError(
                    f"{engine} requires the '{module_name}' package. Install it via `{install_cmd}`"
                ) from exc
