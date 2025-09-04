"""Prefetch TTS models for offline use."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.model_manager import DownloadError, ensure_model, list_models  # noqa: E402


def fetch(models: list[str]) -> None:
    ok = True
    for name in models:
        if name == "silero":
            import torch

            hub_dir = Path(torch.hub.get_dir())
            cache_dir = hub_dir / "snakers4_silero-models_master"
            cached_before = cache_dir.exists()
            try:
                torch.hub.load(
                    repo_or_dir="snakers4/silero-models",
                    model="silero_tts",
                    language="ru",
                    speaker="v4_ru",
                    trust_repo=True,
                    force_reload=False,
                )
            except Exception as exc:  # pragma: no cover - network errors
                logging.error("silero: download failed: %s", exc)
                ok = False
                continue
            action = "cached" if cached_before else "downloaded"
            logging.info("silero: %s at %s", action, cache_dir)
            continue
        target = ROOT / "models" / "tts" / name
        cached = target.exists()
        try:
            path = ensure_model(name, "tts", auto_download=True)
        except DownloadError as exc:  # pragma: no cover - network errors
            logging.error("%s: download failed: %s", name, exc)
            ok = False
            continue
        action = "cached" if cached else "downloaded"
        logging.info("%s: %s at %s", name, action, path)
    if not ok:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch TTS models")
    available = sorted({*list_models("tts"), "silero"})
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

