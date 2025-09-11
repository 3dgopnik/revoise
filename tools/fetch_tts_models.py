"""Prefetch TTS models for offline use."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import lazily to keep dependencies optional
from core.model_manager import DownloadError, ensure_model, list_models  # noqa: E402,I001


SILERO_LANG_MODELS = {"ru": "v4_ru", "en": "v3_en", "de": "v3_de"}
VIBEVOICE_REPOS = {
    "1.5b": "vibe-voice/vibevoice-1_5b",
    "large": "vibe-voice/vibevoice-large",
    "7b": "vibe-voice/vibevoice-large",
}


def fetch_registry(models: Iterable[str]) -> None:
    ok = True
    for name in models:
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


def fetch_silero(language: str) -> None:
    if importlib.util.find_spec("torch") is None:
        logging.warning("silero-%s: torch not installed, skipping", language)
        return
    import torch

    torch_home = ROOT / "models" / "torch_hub"
    torch_home.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_home))
    hub_dir = Path(torch.hub.get_dir())
    cache_dir = hub_dir / "snakers4_silero-models_master"
    cached_before = cache_dir.exists()
    try:
        torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=language,
            speaker=SILERO_LANG_MODELS.get(language, "v4_ru"),
            trust_repo=True,
            force_reload=False,
        )
    except Exception as exc:  # pragma: no cover - network errors
        logging.error("silero-%s: download failed: %s", language, exc)
        raise SystemExit(1) from exc
    action = "cached" if cached_before else "downloaded"
    logging.info("silero-%s: %s at %s", language, action, cache_dir)


def fetch_vibevoice(model: str) -> None:
    if importlib.util.find_spec("huggingface_hub") is None:
        logging.error(
            "vibevoice-%s: huggingface_hub not installed. Run `uv pip install huggingface_hub`.",
            model,
        )
        raise SystemExit(1)
    from huggingface_hub import snapshot_download

    try:
        repo_id = VIBEVOICE_REPOS[model]
    except KeyError as exc:
        logging.error("vibevoice: unknown model '%s'", model)
        raise SystemExit(1) from exc
    target = ROOT / "models" / "tts" / f"vibevoice-{model}"
    cached = target.exists()
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=target,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:  # pragma: no cover - network errors
        logging.error("vibevoice-%s: download failed: %s", model, exc)
        raise SystemExit(1) from exc
    action = "cached" if cached else "downloaded"
    logging.info("vibevoice-%s: %s at %s", model, action, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch TTS models")
    sub = parser.add_subparsers(dest="cmd", required=True)

    reg = sub.add_parser("registry", help="Fetch models from registry")
    available = sorted(list_models("tts"))
    reg.add_argument("--engine", action="append", choices=available, help="Model to prefetch")
    reg.add_argument("--all", action="store_true", help="Fetch all registry models")

    sil = sub.add_parser("silero", help="Fetch Silero voice pack")
    sil.add_argument("--language", default="ru", help="Language code (ru, en, de)")

    vib = sub.add_parser("vibevoice", help="Fetch VibeVoice weights")
    vib.add_argument("--model", choices=sorted(VIBEVOICE_REPOS), default="1.5b")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.cmd == "registry":
        if args.all:
            fetch_registry(available)
        elif args.engine:
            fetch_registry(args.engine)
        else:  # pragma: no cover - argument parsing safeguard
            reg.error("Specify --engine or --all")
    elif args.cmd == "silero":
        fetch_silero(args.language)
    elif args.cmd == "vibevoice":
        fetch_vibevoice(args.model)


if __name__ == "__main__":  # pragma: no cover
    main()

