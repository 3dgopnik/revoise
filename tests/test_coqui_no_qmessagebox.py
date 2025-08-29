from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import model_manager, tts_adapters
from core.tts_adapters import CoquiXTTS


def test_coqui_ensure_model_no_qmessagebox(monkeypatch, tmp_path: Path):
    msg_calls = {"question": 0, "warning": 0}

    class MsgBox:
        Yes = 1
        No = 0
        Retry = 2

        @staticmethod
        def question(*args, **kwargs):
            msg_calls["question"] += 1
            return MsgBox.No

        @staticmethod
        def warning(*args, **kwargs):
            msg_calls["warning"] += 1
            return MsgBox.No

    monkeypatch.setattr(model_manager, "QMessageBox", MsgBox)

    captured: dict[str, bool] = {}

    def fake_ensure_model(name, category, *, parent=None, auto_download=False):
        captured["auto_download"] = auto_download
        if not auto_download:
            model_manager.QMessageBox.question(None, "", "", 0, 0)
        return tmp_path

    monkeypatch.setattr(tts_adapters, "ensure_model", fake_ensure_model)

    api_module = types.ModuleType("api")

    class DummyTTS:
        def __init__(self, *args, **kwargs):
            pass

    api_module.TTS = DummyTTS
    tts_module = types.ModuleType("TTS")
    tts_module.api = api_module
    monkeypatch.setitem(sys.modules, "TTS", tts_module)
    monkeypatch.setitem(sys.modules, "TTS.api", api_module)

    monkeypatch.setattr(CoquiXTTS, "_model", None)
    CoquiXTTS(tmp_path)._ensure_model()

    assert captured["auto_download"] is True
    assert msg_calls["question"] == 0
    assert msg_calls["warning"] == 0
