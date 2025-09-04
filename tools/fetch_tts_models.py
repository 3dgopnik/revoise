"""Prefetch TTS models for offline use."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.model_manager import DownloadError, ensure_model, list_models


def fetch(models: list[str]) -> None:
    for name in models:
        target = ROOT / "models" / "tts" / name
        cached = target.exists()
        try:
            path = ensure_model(name, "tts", auto_download=True)
        except DownloadError as exc:  # pragma: no cover - network errors
            logging.error("%s: download failed: %s", name, exc)
            continue
        action = "cached" if cached else "downloaded"
        logging.info("%s: %s at %s", name, action, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch TTS models")
    available = sorted(list_models("tts"))
    parser.add_argument(
        "--engine",
        action="append",
        choices=available,
        help="Engine to prefetch",
    )
    parser.add_argument("--all", action="store_true", help="Fetch all TTS models")
    args = parser.parse_args()

    if args.all:
        engines = available
    elif args.engine:
        engines = args.engine
    else:  # pragma: no cover - argument parsing safeguard
        parser.error("Specify --engine or --all")

    logging.basicConfig(level=logging.INFO)
    fetch(list(engines))


if __name__ == "__main__":  # pragma: no cover
    main()

