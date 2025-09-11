import importlib
import logging
import sys
import types
from pathlib import Path

from tools import fetch_tts_models


def test_fetch_vibevoice_downloads(monkeypatch, tmp_path):
    calls: dict[str, str] = {}

    def fake_snapshot_download(repo_id, local_dir, local_dir_use_symlinks, resume_download):
        calls["repo_id"] = repo_id
        calls["local_dir"] = str(local_dir)
        Path(local_dir).mkdir(parents=True, exist_ok=True)

    module = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: module if name == "huggingface_hub" else None,
    )

    fetch_tts_models.fetch_vibevoice("1.5b")
    assert calls["repo_id"] == "vibe-voice/vibevoice-1_5b"


def test_fetch_silero_downloads_language(monkeypatch, tmp_path):
    calls: list[dict] = []

    dummy_torch = types.SimpleNamespace(
        hub=types.SimpleNamespace(
            set_dir=lambda p: None,
            get_dir=lambda: tmp_path,
            load=lambda **kw: calls.append(kw),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)
    monkeypatch.setitem(sys.modules, "torchaudio", types.ModuleType("torchaudio"))
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace() if name in ("torch", "torchaudio") else None,
    )

    fetch_tts_models.fetch_silero("en")
    assert calls and calls[0]["language"] == "en"


def test_fetch_silero_missing_torch(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    fetch_tts_models.fetch_silero("ru")
    assert "torch not installed" in caplog.text

