from __future__ import annotations

import importlib
import importlib.util
import logging
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

logger = logging.getLogger(__name__)


def ensure_uv() -> None:
    """Ensure the uv CLI is installed."""
    try:
        import uv  # noqa: F401
    except ModuleNotFoundError:
        try:
            subprocess.run(
                [sys.executable, "-m", "ensurepip", "--upgrade"],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "uv"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as err:
            logger.error(
                "Failed to install uv\nstdout:%s\nstderr:%s",
                err.stdout,
                err.stderr,
            )


def ensure_tts_dependencies(engine: str) -> None:
    """Ensure that required packages for a TTS engine are installed and importable.

    For example, ``ensure_tts_dependencies("silero")`` installs ``torch`` and ``omegaconf``.
    """
    ensure_uv()
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
                            text=True,
                        )
                    except subprocess.CalledProcessError as gpu_err:
                        logger.error(
                            "Failed to install torch (GPU)\nstdout:%s\nstderr:%s",
                            gpu_err.stdout,
                            gpu_err.stderr,
                        )
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
                                text=True,
                            )
                        except subprocess.CalledProcessError as cpu_err:
                            logger.error(
                                "Failed to install torch (CPU)\nstdout:%s\nstderr:%s",
                                cpu_err.stdout,
                                cpu_err.stderr,
                            )
                            raise RuntimeError(
                                "Failed to install torch. Check your Python version, reinstall the torch CPU package, or clear the uv cache."
                            ) from gpu_err
                else:
                    cmd_parts = install_cmd.split()
                    insert_at = cmd_parts.index("install") + 1
                    cmd_parts.insert(insert_at, f"--target={TTS_PKG_DIR}")
                    cmd = [sys.executable, "-m", *cmd_parts]
                    if module_name == "omegaconf":
                        for attempt in range(2):
                            try:
                                subprocess.run(cmd, check=True, capture_output=True, text=True)
                                break
                            except subprocess.CalledProcessError as err:
                                logger.error(
                                    "Failed to install omegaconf (attempt %s)\nstdout:%s\nstderr:%s",
                                    attempt + 1,
                                    err.stdout,
                                    err.stderr,
                                )
                                if attempt == 1:
                                    raise RuntimeError(
                                        f"{engine} requires the '{module_name}' package. Install it via `{install_cmd}`"
                                    ) from err
                    else:
                        subprocess.run(cmd, check=True, capture_output=True, text=True)
                if str(TTS_PKG_DIR) not in sys.path:
                    sys.path.insert(0, str(TTS_PKG_DIR))
                importlib.invalidate_caches()
                if importlib.util.find_spec(module_name) is None:
                    raise RuntimeError(
                        f"{engine} requires the '{module_name}' package. Install it via `{install_cmd}`"
                    )
            except Exception:
                raise RuntimeError(
                    f"{engine} requires the '{module_name}' package. Install it via `{install_cmd}`"
                ) from exc
