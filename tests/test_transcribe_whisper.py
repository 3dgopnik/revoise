import sys
import types

import pytest

from core import model_service, pipeline
from core.model_manager import DownloadError


def test_transcribe_whisper_download_refused(tmp_path, monkeypatch):
    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = object
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    captured: dict[str, bool] = {}

    def fake_get_model_path(name, category, *, parent=None, auto_download=False):
        captured["auto_download"] = auto_download
        raise FileNotFoundError("missing")

    monkeypatch.setattr(model_service, "get_model_path", fake_get_model_path)
    monkeypatch.setattr(pipeline, "FWHISPER", None)
    monkeypatch.setattr(model_service, "_MODEL_PATH_CACHE", {})

    with pytest.raises(RuntimeError, match="download was declined"):
        pipeline.transcribe_whisper(tmp_path / "dummy.wav")

    assert captured["auto_download"] is True


def test_transcribe_whisper_loads_existing_model(tmp_path, monkeypatch):
    dummy_path = tmp_path / "dummy"

    class DummyWhisperModel:
        def __init__(self, model_path, *args, **kwargs):
            self.model_path = model_path

        def transcribe(self, *args, **kwargs):
            return [], None

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = DummyWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    monkeypatch.setattr(model_service, "get_model_path", lambda name, category, **kwargs: dummy_path)
    monkeypatch.setattr(pipeline, "FWHISPER", None)
    monkeypatch.setattr(model_service, "_MODEL_PATH_CACHE", {})

    result = pipeline.transcribe_whisper(tmp_path / "sample.wav")

    assert isinstance(pipeline.FWHISPER, DummyWhisperModel)
    assert pipeline.FWHISPER.model_path == str(dummy_path)
    assert isinstance(result, list) and result == []


def test_transcribe_whisper_uses_model_cache(tmp_path, monkeypatch):
    dummy_path = tmp_path / "dummy"

    class DummyWhisperModel:
        def __init__(self, model_path, *args, **kwargs):
            self.model_path = model_path

        def transcribe(self, *args, **kwargs):
            return [], None

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = DummyWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    calls = {"count": 0}

    def fake_ensure_model(name, category, **kwargs):
        calls["count"] += 1
        return dummy_path

    monkeypatch.setattr(model_service, "ensure_model", fake_ensure_model)
    monkeypatch.setattr(model_service, "_MODEL_PATH_CACHE", {})

    monkeypatch.setattr(pipeline, "FWHISPER", None)
    pipeline.transcribe_whisper(tmp_path / "first.wav")

    monkeypatch.setattr(pipeline, "FWHISPER", None)
    pipeline.transcribe_whisper(tmp_path / "second.wav")

    assert calls["count"] == 1


def test_transcribe_whisper_download_failure(tmp_path, monkeypatch):
    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = object
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    captured: dict[str, bool] = {}

    def fake_get_model_path(name, category, *, parent=None, auto_download=False):
        captured["auto_download"] = auto_download
        raise DownloadError("failed")

    monkeypatch.setattr(model_service, "get_model_path", fake_get_model_path)
    monkeypatch.setattr(pipeline, "FWHISPER", None)
    monkeypatch.setattr(model_service, "_MODEL_PATH_CACHE", {})

    with pytest.raises(RuntimeError, match="download failed"):
        pipeline.transcribe_whisper(tmp_path / "dummy.wav")

    assert captured["auto_download"] is True

