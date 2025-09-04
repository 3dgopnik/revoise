import logging
import sys
import types
from pathlib import Path

import pytest

from core.tts_adapters import SileroTTS


def test_silero_logs_torch_version(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setitem(sys.modules, "omegaconf", types.ModuleType("omegaconf"))

    dummy_torch = types.SimpleNamespace(
        __version__="1.2.3",
        set_num_threads=lambda n: None,
        hub=types.SimpleNamespace(
            get_dir=lambda: ".",
            load=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    messages: list[str] = []

    def fake_info(msg, *args, **kwargs):
        messages.append(msg % args)

    monkeypatch.setattr(logging, "info", fake_info)

    tts = SileroTTS(Path("."))
    with pytest.raises(RuntimeError, match="stop"):
        tts._ensure_model()

    assert messages and "1.2.3" in messages[0]
