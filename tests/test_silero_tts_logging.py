import logging
import sys
import types
from pathlib import Path

import pytest

from core.tts_adapters import SileroTTS


def test_silero_logs_torch_version(monkeypatch):
    dummy_torch = types.SimpleNamespace(
        __version__="1.2.3",
        hub=types.SimpleNamespace(
            load=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")),
            get_dir=lambda: ".",
        ),
        set_num_threads=lambda *a, **k: None,
        device=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    messages: list[str] = []
    monkeypatch.setattr(logging, "info", lambda msg, *a, **k: messages.append(msg % a))

    tts = SileroTTS(Path("."), auto_download=True)
    with pytest.raises(RuntimeError, match="Silero download failed: stop"):
        tts._ensure_model(auto_download=True)

    assert any("1.2.3" in m for m in messages)
