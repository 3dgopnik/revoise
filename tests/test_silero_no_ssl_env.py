import ssl
import ssl
import sys
import types
from typing import Any

import numpy as np
import pytest

from core.tts_adapters import SileroTTS, set_ssl_verification


class DummyModel:
    def apply_tts(self, *args, **kwargs):
        return np.zeros(1, dtype=np.float32)

    def to(self, *args, **kwargs):
        return self


@pytest.mark.parametrize(
    "mode, expected_unverified",
    [
        ("default", False),
        ("env_off", True),
        ("toggle_off", True),
    ],
)
def test_no_ssl_verify(monkeypatch, tmp_path, mode, expected_unverified):
    hub_dir = tmp_path / "hub"
    cache_dir = hub_dir / "snakers4_silero-models_master"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "src/silero/model").mkdir(parents=True, exist_ok=True)
    (cache_dir / "src/silero/model/v4_ru.pt").touch()

    context_during_load: dict[str, Any] = {}

    def fake_load(*args, **kwargs):
        context_during_load["value"] = ssl._create_default_https_context
        return DummyModel(), "ok"

    torch = types.SimpleNamespace(
        __version__="0.0",
        set_num_threads=lambda *a, **k: None,
        hub=types.SimpleNamespace(
            set_dir=lambda *a, **k: None,
            get_dir=lambda: str(hub_dir),
            load=fake_load,
        ),
        device=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchaudio", types.ModuleType("torchaudio"))

    set_ssl_verification(True)
    if mode == "env_off":
        monkeypatch.setenv("NO_SSL_VERIFY", "1")
    else:
        monkeypatch.delenv("NO_SSL_VERIFY", raising=False)
    if mode == "toggle_off":
        set_ssl_verification(False)

    original_ctx = ssl._create_default_https_context
    monkeypatch.setattr(ssl, "_create_default_https_context", original_ctx, raising=False)

    SileroTTS._models = {}
    SileroTTS._statuses = {}
    SileroTTS._speakers = {}
    SileroTTS._mode = None
    SileroTTS(auto_download=False)._ensure_model(auto_download=False)

    assert "value" in context_during_load
    expected_context = (
        ssl._create_unverified_context if expected_unverified else original_ctx
    )
    assert context_during_load["value"] is expected_context
    assert ssl._create_default_https_context is original_ctx
    set_ssl_verification(True)
