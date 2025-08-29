import json
from pathlib import Path

import pytest

from core import model_manager
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
    monkeypatch.setattr(
        model_manager,
        "MODEL_SOURCES",
        {"tts": {"downloaded.bin": url}},
    )

    class MsgBox:
        Yes = 1
        No = 0
        Retry = 2
        Cancel = 3

        @staticmethod
        def question(*args, **kwargs):
            return MsgBox.Yes

        @staticmethod
        def warning(*args, **kwargs):
            return MsgBox.Cancel

    monkeypatch.setattr(model_manager, "QMessageBox", MsgBox)

    model_path = ensure_model("downloaded.bin", "tts")
    assert model_path.exists()
    stored = json.loads(Path("config.json").read_text())
    assert stored["models"]["tts"]["downloaded.bin"] == str(model_path)


def test_user_declines_download(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.bin"
    source.write_text("payload")
    url = source.as_uri()
    monkeypatch.setattr(
        model_manager,
        "MODEL_SOURCES",
        {"tts": {"missing.bin": url}},
    )

    class MsgBox:
        Yes = 1
        No = 0
        Retry = 2
        Cancel = 3

        @staticmethod
        def question(*args, **kwargs):
            return MsgBox.No

        @staticmethod
        def warning(*args, **kwargs):
            return MsgBox.Cancel

    monkeypatch.setattr(model_manager, "QMessageBox", MsgBox)

    with pytest.raises(FileNotFoundError):
        ensure_model("missing.bin", "tts")


def test_download_failure(tmp_path):
    target = tmp_path / "file.bin"
    with pytest.raises(DownloadError):
        download_model("file:///nonexistent.bin", target)
