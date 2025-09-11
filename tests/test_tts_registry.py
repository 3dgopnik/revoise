import importlib

import pytest

tts_registry = importlib.import_module("core.tts.registry")
from core.tts.engines import BeepEngine, SileroEngine, VibeVoiceEngine


def _reset_loaded():
    tts_registry._loaded.clear()


def test_get_engine_returns_correct_classes():
    _reset_loaded()
    assert isinstance(tts_registry.get_engine("silero"), SileroEngine)
    _reset_loaded()
    assert isinstance(tts_registry.get_engine("beep"), BeepEngine)
    _reset_loaded()
    assert isinstance(tts_registry.get_engine("vibevoice"), VibeVoiceEngine)


def test_get_engine_unknown_falls_back_to_beep():
    _reset_loaded()
    assert isinstance(tts_registry.get_engine("unknown"), BeepEngine)
