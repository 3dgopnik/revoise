import json
import sys
import types
from pathlib import Path

import pytest

from core import model_manager
from core.model_manager import DownloadError, download_model, ensure_model


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
        "MODEL_REGISTRY",
        {"tts": {"downloaded.bin": [url]}},
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
        "MODEL_REGISTRY",
        {"tts": {"missing.bin": [url]}},
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


def test_ensure_model_fallback_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    good_source = tmp_path / "good.bin"
    good_source.write_text("payload")
    bad_url = (tmp_path / "missing.bin").as_uri()
    good_url = good_source.as_uri()
    monkeypatch.setattr(
        model_manager,
        "MODEL_REGISTRY",
        {"tts": {"combo.bin": [bad_url, good_url]}},
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
            return MsgBox.Retry

    monkeypatch.setattr(model_manager, "QMessageBox", MsgBox)

    model_path = ensure_model("combo.bin", "tts")
    assert model_path.exists()


def test_ensure_model_dict_entry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "payload.bin"
    source.write_text("data")
    url = source.as_uri()
    monkeypatch.setattr(
        model_manager,
        "MODEL_REGISTRY",
        {"tts": {"payload.bin": {"urls": [url], "description": "dummy"}}},
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
    model_path = ensure_model("payload.bin", "tts")
    assert model_path.exists()


def test_download_failure(tmp_path):
    target = tmp_path / "file.bin"
    with pytest.raises(DownloadError):
        download_model("file:///nonexistent.bin", target)


def test_ensure_model_downloads_stt_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    files = ["model.bin", "config.json", "tokenizer.json"]
    for fn in files:
        (repo / fn).write_text("data")
    base_url = repo.as_uri() + "/"
    monkeypatch.setattr(
        model_manager,
        "MODEL_REGISTRY",
        {"stt": {"dummy": {"base_urls": [base_url], "files": files}}},
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

    model_dir = ensure_model("dummy", "stt")
    assert model_dir.is_dir()
    for fn in files:
        assert (model_dir / fn).exists()


def test_whispermodel_loads_from_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    files = ["model.bin", "config.json", "tokenizer.json"]
    for fn in files:
        (repo / fn).write_text("data")
    base_url = repo.as_uri() + "/"
    monkeypatch.setattr(
        model_manager,
        "MODEL_REGISTRY",
        {"stt": {"dummy": {"base_urls": [base_url], "files": files}}},
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

    from core import pipeline

    monkeypatch.setattr(pipeline, "ensure_model", model_manager.ensure_model)
    monkeypatch.setattr(pipeline, "FWHISPER", None)
    monkeypatch.setattr(pipeline, "MODEL_PATH_CACHE", {})

    class DummyWhisperModel:
        def __init__(self, model_dir, *args, **kwargs):
            self.model_dir = model_dir

        def transcribe(self, *args, **kwargs):
            return [], None

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = DummyWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")

    pipeline.transcribe_whisper(audio, model_size="dummy")

    expected_dir = Path("models/stt/dummy")
    assert pipeline.FWHISPER.model_dir == str(expected_dir)
