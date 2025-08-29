from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.request import urlopen

try:  # pragma: no cover - GUI import guard
    from PySide6.QtWidgets import QMessageBox, QWidget
except Exception:  # pragma: no cover - fallback for headless environments

    class QMessageBox:  # type: ignore[override]
        Yes = 1
        No = 0
        Retry = 2
        Cancel = 3

        @staticmethod
        def question(*args, **kwargs):  # pragma: no cover - fallback behaviour
            raise RuntimeError("Qt is not available")

        @staticmethod
        def warning(*args, **kwargs):  # pragma: no cover - fallback behaviour
            raise RuntimeError("Qt is not available")

    class QWidget:  # type: ignore[override]
        pass


from .model_sources import MODEL_SOURCES


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


def ensure_model(name: str, category: str, *, parent: QWidget | None = None) -> Path:
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

    cached = config.get("models", {}).get(category, {}).get(name)
    if cached:
        cached_path = Path(cached)
        if cached_path.exists():
            logging.info("Model '%s' found locally at %s", name, cached_path)
            return cached_path

    url = MODEL_SOURCES.get(category, {}).get(name)
    if url is None:
        raise FileNotFoundError(f"No source URL for model '{name}' in category '{category}'")

    while True:
        consent = QMessageBox.question(
            parent,
            "Download model",
            f"Model '{name}' is missing. Download it?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if consent != QMessageBox.Yes:
            raise FileNotFoundError(f"Model '{name}' is missing.")

        logging.info("Download of model '%s' started", name)
        try:
            download_model(url, model_path)
        except DownloadError as exc:
            logging.info("Download failed: %s", exc)
            retry = QMessageBox.warning(
                parent,
                "Download failed",
                f"Failed to download model '{name}': {exc}",
                QMessageBox.Retry | QMessageBox.Cancel,
                QMessageBox.Retry,
            )
            if retry == QMessageBox.Retry:
                continue
            raise
        else:
            logging.info("Download of model '%s' completed", name)
            config.setdefault("models", {}).setdefault(category, {})[name] = str(model_path)
            with open(config_path, "w", encoding="utf-8") as file:
                json.dump(config, file, indent=2)
            return model_path
