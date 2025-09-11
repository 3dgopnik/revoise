import types

import numpy as np
import pytest

from core.tts.engines.silero_engine import SileroEngine


def test_silero_engine_param_mapping(monkeypatch):
    captured = {}

    def fake_tts(text, speaker, sr, **kwargs):  # type: ignore[unused-argument]
        captured.update(kwargs)
        return np.zeros(1, dtype=np.float32)

    def fake_load(self):
        self._impl = types.SimpleNamespace(tts=fake_tts)

    monkeypatch.setattr(SileroEngine, "load", fake_load)
    engine = SileroEngine()
    wav = engine.synthesize(
        "hi",
        "baya",
        16000,
        rate=1.1,
        pitch=0.2,
        style="foo.wav",
        preset="bar",
    )
    assert wav.shape[0] == 1
    assert captured["rate"] == 1.1
    assert captured["pitch"] == 0.2
    assert captured["style"] == "foo.wav" or captured.get("style_wav") == "foo.wav"
    assert captured["preset"] == "bar"


def test_silero_engine_rate_validation(monkeypatch):
    def fake_load(self):
        self._impl = types.SimpleNamespace(tts=lambda *a, **k: np.zeros(1, dtype=np.float32))

    monkeypatch.setattr(SileroEngine, "load", fake_load)
    engine = SileroEngine()
    with pytest.raises(ValueError):
        engine.synthesize("hi", "baya", 16000, rate=0.1)
