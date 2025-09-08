import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = types.ModuleType("torch")
hub = types.ModuleType("torch.hub")
hub.download_url_to_file = lambda *a, **k: None
hub.set_dir = lambda *a, **k: None
torch.hub = hub
pkg = types.ModuleType("torch.package")
pkg.PackageImporter = type("PackageImporter", (), {})
torch.package = pkg
torch.__version__ = "0.0"
torch.set_num_threads = lambda *a, **k: None
torch.device = lambda *a, **k: None
sys.modules["torch"] = torch
sys.modules["torch.hub"] = hub
sys.modules["torch.package"] = pkg
sys.modules["omegaconf"] = types.ModuleType("omegaconf")

from core import pipeline, pkg_installer  # noqa: E402
from core.tts_adapters import SileroTTS  # noqa: E402


class DummyModel:
    def apply_tts(self, *args, **kwargs):
        return np.zeros(1, dtype=np.float32)

    def to(self, *args, **kwargs):
        return self


def test_silero_download_and_cache(monkeypatch, tmp_path, caplog):
    hub_dir = tmp_path / "hub"
    cache_dir = hub_dir / "snakers4_silero-models_master"
    torch.hub.get_dir = lambda: str(hub_dir)
    calls = {"online": 0, "offline": 0}

    def online_load(*args, **kwargs):
        calls["online"] += 1
        cache_dir.mkdir(parents=True, exist_ok=True)
        return DummyModel(), "hi"

    torch.hub.load = online_load
    SileroTTS._model = None
    with caplog.at_level(logging.INFO):
        _, status = SileroTTS(auto_download=True)._ensure_model(return_status=True)
    assert calls["online"] == 1
    assert status == "downloaded"
    assert str(cache_dir) in caplog.text

    def offline_load(*args, **kwargs):
        calls["offline"] += 1
        if not cache_dir.exists():
            raise RuntimeError("missing")
        return DummyModel(), "hi"

    torch.hub.load = offline_load
    SileroTTS._model = None
    _, status = SileroTTS(auto_download=False)._ensure_model(return_status=True)
    assert calls["offline"] == 1
    assert status == "cached"


def test_silero_no_cache(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    torch.hub.get_dir = lambda: str(hub)

    def fake_load(*args, **kwargs):
        raise RuntimeError("missing")

    torch.hub.load = fake_load

    SileroTTS._model = None
    tts = SileroTTS(auto_download=False)
    with pytest.raises(RuntimeError, match="Auto-download models"):
        tts.tts("hi", "baya", sr=16000)


def test_check_engine_available_no_cache(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    torch.hub.get_dir = lambda: str(hub)

    def fake_load(*args, **kwargs):
        raise RuntimeError("missing")

    torch.hub.load = fake_load
    monkeypatch.setattr(pkg_installer, "ensure_package", lambda *a, **k: None)

    SileroTTS._model = None
    with pytest.raises(pipeline.TTSEngineError, match="Auto-download models"):
        pipeline.check_engine_available("silero", auto_download_models=False)
