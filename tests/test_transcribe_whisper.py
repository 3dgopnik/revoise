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
