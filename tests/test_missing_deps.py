import builtins
import importlib.util as imp_util
import sys
from pathlib import Path

import numpy as np
import pytest

from core import pipeline
from core.pipeline import synth_chunk
from core.tts_adapters import BeepTTS, CoquiXTTS, SileroTTS


def test_silero_ensure_model_missing_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as excinfo:
        SileroTTS(Path("."))._ensure_model()
    assert "pip install torch --index-url https://download.pytorch.org/whl/cpu" in str(
        excinfo.value
    )


def test_coqui_ensure_model_missing_tts(monkeypatch):
    monkeypatch.delitem(sys.modules, "TTS", raising=False)
    monkeypatch.delitem(sys.modules, "TTS.api", raising=False)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("TTS"):
            raise ModuleNotFoundError("No module named 'TTS'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="TTS"):
        CoquiXTTS(Path("."))._ensure_model()


def _setup_synth(monkeypatch, fallback_array: np.ndarray):
    monkeypatch.setattr(pipeline, "run", lambda cmd: None)
    storage: dict[str, np.ndarray] = {}

    def fake_write(path, data, sr):
        storage["data"] = np.array(data, dtype=np.float32)
        storage["sr"] = sr

    def fake_read(path, dtype):
        return storage["data"], storage["sr"]

    monkeypatch.setattr(pipeline.sf, "write", fake_write)
    monkeypatch.setattr(pipeline.sf, "read", fake_read)
    monkeypatch.setattr(BeepTTS, "tts", lambda self, text, speaker, sr: fallback_array)


def test_synth_chunk_fallback_silero(monkeypatch, tmp_path):
    original_find_spec = imp_util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "torch":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(imp_util, "find_spec", fake_find_spec)
    expected = np.array([0.1, -0.1], dtype=np.float32)
    _setup_synth(monkeypatch, expected)
    wav = synth_chunk("ffmpeg", "hi", 16000, "spk", tmp_path, "silero")
    np.testing.assert_array_equal(wav, expected)


def test_synth_chunk_fallback_coqui(monkeypatch, tmp_path):
    original_find_spec = imp_util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in {"TTS", "torch"}:
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(imp_util, "find_spec", fake_find_spec)
    expected = np.array([0.2, -0.2], dtype=np.float32)
    _setup_synth(monkeypatch, expected)
    wav = synth_chunk("ffmpeg", "hi", 16000, "spk", tmp_path, "coqui_xtts")
    np.testing.assert_array_equal(wav, expected)
