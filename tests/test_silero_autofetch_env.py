import os
import sys
import types

import numpy as np
import pytest

from core.tts_adapters import SileroTTS


class DummyModel:
    def apply_tts(self, *args, **kwargs):
        return np.zeros(1, dtype=np.float32)

    def to(self, *args, **kwargs):
        return self


@pytest.mark.parametrize("original", [None, "0"])
def test_env_restored(monkeypatch, tmp_path, original):
    hub_dir = tmp_path / "hub"
    cache_dir = hub_dir / "snakers4_silero-models_master"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "src/silero/model").mkdir(parents=True, exist_ok=True)
    (cache_dir / "src/silero/model/v4_ru.pt").touch()

    torch = types.SimpleNamespace(
        __version__="0.0",
        set_num_threads=lambda *a, **k: None,
        hub=types.SimpleNamespace(
            set_dir=lambda *a, **k: None,
            get_dir=lambda: str(hub_dir),
            load=lambda *a, **k: (DummyModel(), "ok"),
        ),
        device=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchaudio", types.ModuleType("torchaudio"))

    if original is None:
        monkeypatch.delenv("TORCH_HUB_DISABLE_AUTOFETCH", raising=False)
    else:
        monkeypatch.setenv("TORCH_HUB_DISABLE_AUTOFETCH", original)

    SileroTTS._models = {}
    SileroTTS(auto_download=False)._ensure_model(auto_download=False)

    if original is None:
        assert "TORCH_HUB_DISABLE_AUTOFETCH" not in os.environ
    else:
        assert os.environ["TORCH_HUB_DISABLE_AUTOFETCH"] == original


def test_env_set_for_cached_autodownload_true(monkeypatch, tmp_path):
    hub_dir = tmp_path / "hub"
    cache_dir = hub_dir / "snakers4_silero-models_master"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "src/silero/model").mkdir(parents=True, exist_ok=True)
    (cache_dir / "src/silero/model/v4_ru.pt").touch()

    def load(*args, **kwargs):
        assert os.environ.get("TORCH_HUB_DISABLE_AUTOFETCH") == "1"
        return DummyModel(), "ok"

    torch = types.SimpleNamespace(
        __version__="0.0",
        set_num_threads=lambda *a, **k: None,
        hub=types.SimpleNamespace(
            set_dir=lambda *a, **k: None,
            get_dir=lambda: str(hub_dir),
            load=load,
        ),
        device=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchaudio", types.ModuleType("torchaudio"))

    monkeypatch.delenv("TORCH_HUB_DISABLE_AUTOFETCH", raising=False)

    SileroTTS._models = {}
    SileroTTS(auto_download=True)._ensure_model(auto_download=True)

    assert "TORCH_HUB_DISABLE_AUTOFETCH" not in os.environ
