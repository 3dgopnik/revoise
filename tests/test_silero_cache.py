import numpy as np
import pytest
import sys
import types
from pathlib import Path

torch = types.ModuleType("torch")
hub = types.ModuleType("torch.hub")
hub.download_url_to_file = lambda *a, **k: None
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.tts_adapters import SileroTTS


class DummyModel:
    def apply_tts(self, *args, **kwargs):
        return np.zeros(1, dtype=np.float32)

    def to(self, *args, **kwargs):
        return self


def test_silero_download_and_cache(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    cache_dir = hub / "snakers4_silero-models_master"
    torch.hub.get_dir = lambda: str(hub)
    calls = {"n": 0}

    def fake_load(*args, **kwargs):
        calls["n"] += 1
        cache_dir.mkdir(parents=True, exist_ok=True)
        return DummyModel(), "hi"

    torch.hub.load = fake_load

    SileroTTS._model = None
    SileroTTS(tmp_path, auto_download=True).tts("hi", "baya", sr=16000)
    SileroTTS._model = None
    SileroTTS(tmp_path, auto_download=False).tts("hi", "baya", sr=16000)
    assert calls["n"] == 2


def test_silero_no_cache(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    torch.hub.get_dir = lambda: str(hub)

    def fake_load(*args, **kwargs):
        raise RuntimeError("missing")

    torch.hub.load = fake_load

    SileroTTS._model = None
    tts = SileroTTS(tmp_path, auto_download=False)
    with pytest.raises(RuntimeError, match="Auto-download models"):
        tts.tts("hi", "baya", sr=16000)

