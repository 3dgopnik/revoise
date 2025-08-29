from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.request import urlopen


class DownloadError(Exception):
    """Raised when a model download fails."""


def download_model(url: str, target: Path) -> None:
    """Download a model from ``url`` to ``target``.

    The function logs progress information and writes the file to ``target``.
    """
    logging.info("Downloading model from %s to %s", url, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            with open(target, "wb") as file:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent = downloaded / total * 100
                        print(f"\r{percent:6.2f}% downloaded", end="")
        if total:
            print()
        logging.info("Model downloaded to %s", target)
    except Exception as exc:  # pragma: no cover - safety net
        logging.info("Download failed: %s", exc)
        raise DownloadError(str(exc)) from exc


def ensure_model(name: str, category: str) -> Path:
    """Ensure that a model exists locally, downloading if necessary.

    Parameters
    ----------
    name:
        Model file name.
    category:
        Model category used inside the ``models`` folder.

    Returns
    -------
    Path
        Path to the model file on disk.
    """
    models_dir = Path("models") / category
    model_path = models_dir / name
    if model_path.exists():
        logging.info("Model '%s' found locally at %s", name, model_path)
        return model_path

    config_path = Path("config.json")
    config: dict = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)

    cached = (
        config.get("models", {})
        .get(category, {})
        .get(name)
    )
    if cached:
        cached_path = Path(cached)
        if cached_path.exists():
            logging.info("Model '%s' found locally at %s", name, cached_path)
            return cached_path

    while True:
        consent = input(
            f"Model '{name}' is missing. Download it? [y/N]: "
        ).strip().lower()
        if consent not in {"y", "yes"}:
            raise FileNotFoundError(f"Model '{name}' is missing.")

        url = input("Enter download URL: ").strip()
        logging.info("Download of model '%s' started", name)
        try:
            download_model(url, model_path)
        except DownloadError as exc:
            print(f"Download failed: {exc}")
            retry = input("Retry download? [y/N]: ").strip().lower()
            if retry in {"y", "yes"}:
                continue
            raise
        else:
            logging.info("Download of model '%s' completed", name)
            config.setdefault("models", {}).setdefault(category, {})[name] = str(model_path)
            with open(config_path, "w", encoding="utf-8") as file:
                json.dump(config, file, indent=2)
            return model_path
