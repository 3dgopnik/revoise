from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from collections.abc import Sequence

logger = logging.getLogger(__name__)


def ensure_uv() -> None:
    """Ensure the uv CLI is installed."""
    try:  # pragma: no cover - optional dependency
        import uv  # noqa: F401

        return
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        pass
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
    except subprocess.CalledProcessError as err:  # pragma: no cover - error path
        logger.error("Failed to install uv\nstdout:%s\nstderr:%s", err.stdout, err.stderr)
        raise


def ensure_package(
    module_name: str,
    *,
    package: str | None = None,
    install_args: Sequence[str] | None = None,
    fallback_args: Sequence[str] | None = None,
) -> None:
    """Ensure *module_name* is importable, installing via uv if missing."""
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError:
        pass

    ensure_uv()
    pkg = package or module_name
    cmd = [sys.executable, "-m", "uv", "pip", "install", pkg]
    if install_args:
        cmd.extend(install_args)

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as err:
        if fallback_args:
            fallback_cmd = [
                sys.executable,
                "-m",
                "uv",
                "pip",
                "install",
                pkg,
                *fallback_args,
            ]
            try:
                subprocess.run(fallback_cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as err2:
                logger.error(
                    "Failed to install %s\nstdout:%s\nstderr:%s",
                    pkg,
                    err2.stdout,
                    err2.stderr,
                )
                raise RuntimeError(f"Failed to install package '{pkg}'") from err2
        else:
            logger.error(
                "Failed to install %s\nstdout:%s\nstderr:%s",
                pkg,
                err.stdout,
                err.stderr,
            )
            raise RuntimeError(f"Failed to install package '{pkg}'") from err

    importlib.invalidate_caches()
    importlib.import_module(module_name)
