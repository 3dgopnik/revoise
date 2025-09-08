from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
import subprocess
import sys
from pathlib import Path


def ensure_uv() -> None:
    """Ensure the uv CLI is installed."""
    try:  # pragma: no cover - optional dependency
        import uv  # noqa: F401

        return
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        pass
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


_CONFIG_CACHE: dict[str, bool] | None = None


def _module_from_spec(pkg_spec: str) -> str:
    """Derive an importable module name from *pkg_spec*."""
    name = pkg_spec.split("==", 1)[0].split("[", 1)[0]
    return name.replace("-", "_").replace(".", "_").lower()


def _pin_preference() -> bool:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        try:
            with open("config.json", encoding="utf-8") as fh:
                data = json.load(fh)
            _CONFIG_CACHE = {"pin": bool(data.get("preferences", {}).get("pin_dependencies", False))}
        except Exception:
            _CONFIG_CACHE = {"pin": False}
    return _CONFIG_CACHE["pin"]


def ensure_package(pkg_spec: str, message: str, ask_to_pin: bool | None = None) -> None:
    """Ensure *pkg_spec* is installed and importable.

    Prompts the user to install the package and optionally pin it to
    ``requirements.txt``. The import name is derived from ``pkg_spec``.
    """

    if ask_to_pin is None:
        ask_to_pin = _pin_preference()

    module_name = _module_from_spec(pkg_spec)
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError:
        pass

    print(message)
    install = input(f"Install {pkg_spec}? [Y/n]: ").strip().lower()
    if install not in {"", "y", "yes"}:
        raise ModuleNotFoundError(module_name)

    ensure_uv()
    subprocess.run([sys.executable, "-m", "uv", "pip", "install", pkg_spec], check=True)

    importlib.invalidate_caches()
    importlib.import_module(module_name)

    if not ask_to_pin:
        return

    pin = input("Pin to requirements.txt? [y/N]: ").strip().lower()
    if pin in {"y", "yes"}:
        pkg_name = pkg_spec.split("==", 1)[0]
        try:
            version = metadata.version(pkg_name)
            line = f"{pkg_name}=={version}\n"
        except metadata.PackageNotFoundError:  # pragma: no cover - fallback path
            line = f"{pkg_spec}\n"
        req_file = Path("requirements.txt")
        req_file.parent.mkdir(parents=True, exist_ok=True)
        with req_file.open("a", encoding="utf-8") as fh:
            fh.write(line)

