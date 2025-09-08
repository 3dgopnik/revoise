import builtins
import importlib
import importlib.util as imp_util
import logging
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from core import pipeline, pkg_installer
from core.pipeline import synth_chunk
from core.tts_adapters import BeepTTS, CoquiXTTS, SileroTTS
from core.tts_dependencies import ensure_tts_dependencies


def test_silero_ensure_model_missing_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setitem(sys.modules, "omegaconf", types.ModuleType("omegaconf"))
    original_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    SileroTTS._model = None
    with pytest.raises(RuntimeError) as excinfo:
        SileroTTS()._ensure_model()
    assert "uv pip install torch --index-url https://download.pytorch.org/whl/cpu" in str(
        excinfo.value
    )
    assert calls and "torch" in " ".join(calls[0])


def test_silero_missing_omegaconf(monkeypatch):
    monkeypatch.delitem(sys.modules, "omegaconf", raising=False)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    original_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "omegaconf":
            raise ModuleNotFoundError("No module named 'omegaconf'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="uv pip install omegaconf"):
        ensure_tts_dependencies("silero")
    assert calls and "omegaconf" in " ".join(calls[0])


def test_coqui_ensure_model_missing_tts(monkeypatch):
    monkeypatch.delitem(sys.modules, "TTS", raising=False)
    monkeypatch.delitem(sys.modules, "TTS.api", raising=False)
    original_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name.startswith("TTS"):
            raise ModuleNotFoundError("No module named 'TTS'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="uv pip install TTS"):
        CoquiXTTS(Path("."))._ensure_model()
    assert calls and "TTS" in calls[0]


def _setup_synth(monkeypatch, fallback_array: np.ndarray):
    monkeypatch.setattr(pipeline, "run", lambda cmd: None)
    storage: dict[str, np.ndarray] = {}

    def fake_write(path, data, sr):
        storage["data"] = np.array(data, dtype=np.float32).reshape(-1)
        storage["sr"] = sr

    def fake_read(path, dtype):
        return storage["data"], storage["sr"]

    monkeypatch.setattr(pipeline.sf, "write", fake_write)
    monkeypatch.setattr(pipeline.sf, "read", fake_read)
    monkeypatch.setattr(BeepTTS, "tts", lambda self, text, speaker, sr: fallback_array)


def test_synth_chunk_fallback_silero(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    def fake_ensure(*args, **kwargs):
        raise ModuleNotFoundError("torch")

    monkeypatch.setattr(pkg_installer, "ensure_package", fake_ensure)

    expected = np.array([0.1, -0.1], dtype=np.float32)
    _setup_synth(monkeypatch, expected)
    wav, reason = synth_chunk(
        "ffmpeg", "hi", 16000, "spk", tmp_path, "silero", allow_beep_fallback=True
    )
    np.testing.assert_array_equal(wav, expected)
    assert reason


def test_synth_chunk_fallback_silero_warns(monkeypatch, tmp_path, caplog):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    def fake_ensure(*args, **kwargs):
        raise ModuleNotFoundError("torch")

    monkeypatch.setattr(pkg_installer, "ensure_package", fake_ensure)

    _setup_synth(monkeypatch, np.array([0.1, -0.1], dtype=np.float32))
    with caplog.at_level(logging.INFO):
        synth_chunk("ffmpeg", "hi", 16000, "spk", tmp_path, "silero", allow_beep_fallback=True)
    assert "fallback=true" in caplog.text


def test_synth_chunk_fallback_coqui(monkeypatch, tmp_path):
    original_find_spec = imp_util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in {"TTS", "torch"}:
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(imp_util, "find_spec", fake_find_spec)

    def fake_ensure(*args, **kwargs):
        raise ModuleNotFoundError("TTS")

    monkeypatch.setattr(pkg_installer, "ensure_package", fake_ensure)

    expected = np.array([0.2, -0.2], dtype=np.float32)
    _setup_synth(monkeypatch, expected)
    wav, reason = synth_chunk(
        "ffmpeg", "hi", 16000, "spk", tmp_path, "coqui_xtts", allow_beep_fallback=True
    )
    np.testing.assert_array_equal(wav, expected)
    assert reason
