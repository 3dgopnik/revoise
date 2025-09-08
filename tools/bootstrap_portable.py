"""Prepare portable environment by fetching models and dependencies."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("REVOISE_TTS_PKG_DIR", str(ROOT / ".portable_pkgs"))

from core.model_manager import ensure_model, list_models  # noqa: E402
from core.tts_dependencies import ensure_tts_dependencies  # noqa: E402
from tools.fetch_tts_models import fetch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap portable resources")
    available_stt = sorted(list_models("stt"))
    parser.add_argument(
        "--stt",
        action="append",
        choices=available_stt,
        help="STT model to prefetch",
    )
    parser.add_argument(
        "--all-stt",
        action="store_true",
        help="Fetch all STT models",
    )
    args = parser.parse_args()

    tts_models = sorted({*list_models("tts"), "silero"})
    for engine in (*tts_models, "gtts"):
        try:
            ensure_tts_dependencies(engine)
        except RuntimeError as exc:  # pragma: no cover - optional deps
            if engine == "silero":
                print(f"Failed to prepare silero dependencies: {exc}", file=sys.stderr)
                raise SystemExit(1) from exc
            print(f"Skipping {engine} dependencies: {exc}")

    fetch(tts_models)
    stt_models: list[str] = []
    if args.all_stt:
        stt_models = available_stt
    elif args.stt:
        stt_models = args.stt
    for name in stt_models:
        ensure_model(name, "stt", auto_download=True)


if __name__ == "__main__":  # pragma: no cover
    main()
