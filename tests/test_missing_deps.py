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

from core import model_manager, pipeline, pkg_installer
import core.tts_adapters as tts_adapters
from core.pipeline import synth_chunk
from core.tts_adapters import BeepTTS, CoquiXTTS, SileroTTS
import core.tts_dependencies as tts_deps
from core.tts_dependencies import ensure_tts_dependencies


_PKG_INSTALLER_SPEC = importlib.util.spec_from_file_location(
    "_pkg_installer_real",
    Path(__file__).resolve().parents[1] / "core" / "pkg_installer.py",
)
assert _PKG_INSTALLER_SPEC.loader is not None
_REAL_PKG_INSTALLER = importlib.util.module_from_spec(_PKG_INSTALLER_SPEC)
_PKG_INSTALLER_SPEC.loader.exec_module(_REAL_PKG_INSTALLER)


def test_ensure_package_auto_installs_without_prompt(monkeypatch):
    real_installer = _REAL_PKG_INSTALLER
    pkg_spec = "example-pkg"
    module_name = real_installer._module_from_spec(pkg_spec)
    original_import = importlib.import_module
    state = {"installed": False}

    def fake_import(name, *args, **kwargs):
        if name == module_name:
            if state["installed"]:
                return types.ModuleType(module_name)
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(real_installer.importlib, "import_module", fake_import)
    monkeypatch.setattr(real_installer, "ensure_uv", lambda: None)
    monkeypatch.setattr(real_installer, "_auto_install_preference", lambda: True)
    monkeypatch.setattr(real_installer, "_pin_preference", lambda: False)
    monkeypatch.setattr(real_installer, "_CONFIG_CACHE", None)

    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        state["installed"] = True
        return subprocess.CompletedProcess(cmd, 0)

    def fail_input(*_args, **_kwargs):  # pragma: no cover - defensive, should not trigger
        raise AssertionError("input should not be called when auto-installing")

    monkeypatch.setattr(real_installer.subprocess, "run", fake_run)
    monkeypatch.setattr(builtins, "input", fail_input)

    real_installer.ensure_package(pkg_spec, "Need example package")

    assert calls == [[sys.executable, "-m", "uv", "pip", "install", pkg_spec]]
    imported = real_installer.importlib.import_module(module_name)
    assert isinstance(imported, types.ModuleType)


def test_silero_ensure_model_missing_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setitem(sys.modules, "omegaconf", types.ModuleType("omegaconf"))
    monkeypatch.setitem(sys.modules, "torchaudio", types.ModuleType("torchaudio"))
    original_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    real_installer = _REAL_PKG_INSTALLER
    monkeypatch.setattr(pkg_installer, "ensure_package", real_installer.ensure_package)
    monkeypatch.setattr(tts_adapters, "ensure_package", real_installer.ensure_package)
    monkeypatch.setattr(real_installer, "ensure_uv", lambda: None)
    monkeypatch.setattr(real_installer, "_auto_install_preference", lambda: True)
    monkeypatch.setattr(real_installer, "_CONFIG_CACHE", None)

    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(real_installer.subprocess, "run", fake_run)

    SileroTTS._model = None
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        SileroTTS()._ensure_model()
    assert "uv pip install torch" in " ".join(map(str, excinfo.value.cmd))
    assert calls and "torch" in " ".join(calls[0])


def test_silero_missing_omegaconf(monkeypatch):
    monkeypatch.delitem(sys.modules, "omegaconf", raising=False)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "torchaudio", types.ModuleType("torchaudio"))
    original_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "omegaconf":
            raise ModuleNotFoundError("No module named 'omegaconf'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    real_installer = _REAL_PKG_INSTALLER
    monkeypatch.setattr(pkg_installer, "ensure_package", real_installer.ensure_package)
    monkeypatch.setattr(tts_adapters, "ensure_package", real_installer.ensure_package)
    monkeypatch.setattr(tts_deps, "ensure_package", real_installer.ensure_package)
    monkeypatch.setattr(real_installer, "ensure_uv", lambda: None)
    monkeypatch.setattr(real_installer, "_auto_install_preference", lambda: True)
    monkeypatch.setattr(real_installer, "_CONFIG_CACHE", None)

    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(real_installer.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        ensure_tts_dependencies("silero")
    assert "uv pip install omegaconf" in " ".join(map(str, excinfo.value.cmd))
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
    real_installer = _REAL_PKG_INSTALLER
    monkeypatch.setattr(pkg_installer, "ensure_package", real_installer.ensure_package)
    monkeypatch.setattr(tts_adapters, "ensure_package", real_installer.ensure_package)
    monkeypatch.setattr(real_installer, "ensure_uv", lambda: None)
    monkeypatch.setattr(real_installer, "_auto_install_preference", lambda: True)
    monkeypatch.setattr(real_installer, "_CONFIG_CACHE", None)

    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(real_installer.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        CoquiXTTS(Path("."))._ensure_model()
    assert "uv pip install TTS" in " ".join(map(str, excinfo.value.cmd))
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
    stub_msg = type("Msg", (), {"warning": staticmethod(lambda *args, **kwargs: None)})
    monkeypatch.setattr(model_manager, "QMessageBox", stub_msg)
    monkeypatch.setattr(pipeline, "QMessageBox", stub_msg)

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
    stub_msg = type("Msg", (), {"warning": staticmethod(lambda *args, **kwargs: None)})
    monkeypatch.setattr(model_manager, "QMessageBox", stub_msg)
    monkeypatch.setattr(pipeline, "QMessageBox", stub_msg)

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
    stub_msg = type("Msg", (), {"warning": staticmethod(lambda *args, **kwargs: None)})
    monkeypatch.setattr(model_manager, "QMessageBox", stub_msg)
    monkeypatch.setattr(pipeline, "QMessageBox", stub_msg)

    expected = np.array([0.2, -0.2], dtype=np.float32)
    _setup_synth(monkeypatch, expected)
    wav, reason = synth_chunk(
        "ffmpeg", "hi", 16000, "spk", tmp_path, "coqui_xtts", allow_beep_fallback=True
    )
    np.testing.assert_array_equal(wav, expected)
    assert reason
