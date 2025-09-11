import ssl
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


@pytest.mark.parametrize("flag", [None, "1"])
def test_no_ssl_verify(monkeypatch, tmp_path, flag):
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

    if flag is None:
        monkeypatch.delenv("NO_SSL_VERIFY", raising=False)
    else:
        monkeypatch.setenv("NO_SSL_VERIFY", flag)

    original_ctx = ssl._create_default_https_context
    monkeypatch.setattr(ssl, "_create_default_https_context", original_ctx, raising=False)

    SileroTTS._models = {}
    SileroTTS(auto_download=False)._ensure_model(auto_download=False)

    if flag == "1":
        assert ssl._create_default_https_context == ssl._create_unverified_context
    else:
        assert ssl._create_default_https_context is original_ctx
