import json
from pathlib import Path

import pytest

from core.model_manager import DownloadError, ensure_model, download_model


def test_ensure_model_existing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_path = Path("models/tts/existing.bin")
    model_path.parent.mkdir(parents=True)
    model_path.write_text("data")
    assert ensure_model("existing.bin", "tts") == model_path


def test_ensure_model_from_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_path = tmp_path / "cached.bin"
    model_path.write_text("cached")
    config = {"models": {"tts": {"cached.bin": str(model_path)}}}
    Path("config.json").write_text(json.dumps(config))
    assert ensure_model("cached.bin", "tts") == model_path


def test_ensure_model_download(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.bin"
    source.write_text("payload")
    url = source.as_uri()
    inputs = iter(["y", url])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    model_path = ensure_model("downloaded.bin", "tts")
    assert model_path.exists()
    stored = json.loads(Path("config.json").read_text())
    assert stored["models"]["tts"]["downloaded.bin"] == str(model_path)


def test_user_declines_download(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inputs = iter(["n"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    with pytest.raises(FileNotFoundError):
        ensure_model("missing.bin", "tts")


def test_download_failure(tmp_path):
    target = tmp_path / "file.bin"
    with pytest.raises(DownloadError):
        download_model("file:///nonexistent.bin", target)
