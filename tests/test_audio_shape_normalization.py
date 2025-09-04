import shutil

import numpy as np

import core.pipeline as pipeline


def test_audio_shape_normalization(tmp_path, monkeypatch):
    def fake_run(cmd):
        shutil.copy(cmd[3], cmd[-1])
    monkeypatch.setattr(pipeline, "run", fake_run)
    result = pipeline.synth_chunk(
        ffmpeg="ffmpeg",
        text="hello",
        sr=16000,
        speaker="spk",
        tmpdir=tmp_path,
        tts_engine="beep",
    )
    assert (tmp_path / "tts_raw.wav").exists()
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
