from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict
from urllib.request import urlopen

if TYPE_CHECKING:  # pragma: no cover - typing-only definitions

    class QMessageBox:
        Yes: int
        No: int
        Retry: int
        Cancel: int

        @staticmethod
        def question(*args: Any, **kwargs: Any) -> int: ...

        @staticmethod
        def warning(*args: Any, **kwargs: Any) -> int: ...

    class QWidget: ...
else:  # pragma: no cover - GUI import guard
    try:
        from PySide6.QtWidgets import QMessageBox, QWidget
    except Exception:  # pragma: no cover - fallback for headless environments

        class QMessageBox:
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

        class QWidget:
            pass


MODEL_REGISTRY_PATH = Path("models") / "model_registry.json"


def load_model_registry() -> Dict[str, Dict[str, Any]]:
    """Load model registry mapping categories to model metadata."""
    with open(MODEL_REGISTRY_PATH, encoding="utf-8") as file:
        return json.load(file)


MODEL_REGISTRY = load_model_registry()


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


def list_models(category: str) -> Dict[str, Dict[str, Any]]:
    """Return model metadata for a category."""
    models = MODEL_REGISTRY.get(category, {})
    result: Dict[str, Dict[str, Any]] = {}
    for model_name, data in models.items():
        if isinstance(data, list):
            result[model_name] = {"urls": data}
        else:
            result[model_name] = data
    return result


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

    entry = MODEL_REGISTRY.get(category, {}).get(name)
    if isinstance(entry, dict):
        urls = entry.get("urls", [])
    else:
        urls = entry
    if not urls:
        raise FileNotFoundError(f"No source URLs for model '{name}' in category '{category}'")

    consent = QMessageBox.question(
        parent,
        "Download model",
        f"Model '{name}' is missing. Download it?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    if consent != QMessageBox.Yes:
        raise FileNotFoundError(f"Model '{name}' is missing.")

    for url in urls:
        logging.info("Download of model '%s' from %s started", name, url)
        try:
            download_model(url, model_path)
        except DownloadError as exc:
            logging.info("Download failed: %s", exc)
            QMessageBox.warning(
                parent,
                "Download failed",
                f"Failed to download model '{name}' from {url}: {exc}",
                QMessageBox.Retry,
                QMessageBox.Retry,
            )
            continue
        else:
            logging.info("Download of model '%s' completed", name)
            config.setdefault("models", {}).setdefault(category, {})[name] = str(model_path)
            with open(config_path, "w", encoding="utf-8") as file:
                json.dump(config, file, indent=2)
            return model_path

    raise DownloadError(f"Failed to download model '{name}' from all registered sources")
