"""Prefetch STT models for offline use."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.model_manager import DownloadError, list_models  # noqa: E402,I001
from core.model_service import get_model_path  # noqa: E402,I001


def fetch_models(models: Iterable[str]) -> bool:
    """Download the provided STT models."""

    ok = True
    for name in models:
        try:
            path = get_model_path(name, "stt", auto_download=True)
        except FileNotFoundError as exc:
            logging.error("%s: not found in registry (%s)", name, exc)
            ok = False
        except DownloadError as exc:  # pragma: no cover - network errors
            logging.error("%s: download failed: %s", name, exc)
            ok = False
        else:
            logging.info("%s: ready at %s", name, path)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch STT models")
    available = sorted(list_models("stt"))
    parser.add_argument(
        "model",
        nargs="+",
        choices=available if available else None,
        help="Model names to prefetch (e.g. base, small, medium, large-v3)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not fetch_models(args.model):
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()

