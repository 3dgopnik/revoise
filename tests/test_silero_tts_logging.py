import logging
import sys
from pathlib import Path

import pytest

import core.tts_adapters as silero
from core.tts_adapters import SileroTTS


def test_silero_logs_torch_version(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    class DummyTorch:
        __version__ = "1.2.3"

    monkeypatch.setitem(sys.modules, "torch", DummyTorch())

    messages: list[str] = []

    def fake_info(msg, *args, **kwargs):
        messages.append(msg % args)

    monkeypatch.setattr(logging, "info", fake_info)
    monkeypatch.setattr(
        silero,
        "load_silero_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )

    tts = SileroTTS(Path("."))
    with pytest.raises(RuntimeError, match="stop"):
        tts._ensure_model()

    assert messages and "1.2.3" in messages[0]
