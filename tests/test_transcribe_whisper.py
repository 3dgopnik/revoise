import sys
import types

import pytest

from core import pipeline


def test_transcribe_whisper_download_refused(tmp_path, monkeypatch):
    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = object
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    def fake_ensure_model(name, category):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(pipeline, "ensure_model", fake_ensure_model)
    monkeypatch.setattr(pipeline, "FWHISPER", None)

    with pytest.raises(RuntimeError, match="download was declined"):
        pipeline.transcribe_whisper(tmp_path / "dummy.wav")


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

    monkeypatch.setattr(pipeline, "ensure_model", lambda name, category: dummy_path)
    monkeypatch.setattr(pipeline, "FWHISPER", None)

    result = pipeline.transcribe_whisper(tmp_path / "sample.wav")

    assert isinstance(pipeline.FWHISPER, DummyWhisperModel)
    assert pipeline.FWHISPER.model_path == str(dummy_path)
    assert isinstance(result, list) and result == []
