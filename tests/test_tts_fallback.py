import logging
import shutil

import numpy as np
import pytest

from core import pipeline
from core.tts_registry import registry


def _copy_run(cmd):
    shutil.copy(cmd[3], cmd[-1])


def test_signature_detection(tmp_path, monkeypatch):
    calls = {}

    def dummy(text, speaker, sample_rate):
        calls['sr'] = sample_rate
        return np.zeros(10, dtype=np.float32)

    monkeypatch.setattr(pipeline, 'run', _copy_run)
    monkeypatch.setattr(
        pipeline, 'check_engine_available', lambda name, auto_download_models=True: None
    )
    monkeypatch.setitem(registry, 'dummy', dummy)

    wav, reason = pipeline.synth_chunk(
        ffmpeg='ff',
        text='hello',
        sr=123,
        speaker='spk',
        tmpdir=tmp_path,
        tts_engine='dummy',
        allow_beep_fallback=False,
    )
    assert calls['sr'] == 123
    assert reason is None
    assert wav.size > 0


def test_unavailable_engine_no_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'run', _copy_run)

    def unavailable(name, auto_download_models=True):
        raise pipeline.TTSEngineError('missing')

    monkeypatch.setattr(pipeline, 'check_engine_available', unavailable)
    with pytest.raises(pipeline.TTSEngineError):
        pipeline.synth_chunk(
            'ff', 'hi', 48000, 'sp', tmp_path, 'silero', allow_beep_fallback=False
        )


def test_unavailable_engine_with_fallback(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(pipeline, 'run', _copy_run)

    def unavailable(name, auto_download_models=True):
        raise pipeline.TTSEngineError('missing dep')

    monkeypatch.setattr(pipeline, 'check_engine_available', unavailable)
    caplog.set_level(logging.INFO)
    wav, reason = pipeline.synth_chunk(
        'ff', 'hi', 48000, 'sp', tmp_path, 'silero', allow_beep_fallback=True
    )
    assert reason == 'missing dep'
    assert wav.size > 0
    assert any('fallback=true' in rec.message for rec in caplog.records)
