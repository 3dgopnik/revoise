# ruff: noqa: I001
from pathlib import Path
import re
import sys
import types

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from core import pipeline


def test_apply_edited_text_extra_lines_ignored():
    phrases = [(0.0, 1.0, "a"), (1.0, 2.0, "b")]
    edited = "[[#1]] aa\n[[#2]] bb\n[[#3]] cc"  # third line should be ignored
    res = pipeline.apply_edited_text(phrases, edited)
    assert res == [(0.0, 1.0, "aa"), (1.0, 2.0, "bb")]


def test_apply_edited_text_missing_lines_preserve_original():
    phrases = [(0.0, 1.0, "a"), (1.0, 2.0, "b"), (2.0, 3.0, "c")]
    edited = "[[#1]] aa\n[[#2]] bb"  # third phrase remains 'c'
    res = pipeline.apply_edited_text(phrases, edited)
    assert res == [(0.0, 1.0, "aa"), (1.0, 2.0, "bb"), (2.0, 3.0, "c")]


def test_revoice_video_calls_setup_and_catches(monkeypatch, tmp_path):
    dummy_video = tmp_path / "in.mp4"
    dummy_video.write_bytes(b"0")

    monkeypatch.setattr(pipeline, "ensure_ffmpeg", lambda: "ffmpeg")

    def fake_run(cmd):
        Path(cmd[-1]).touch()
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline, "run", fake_run)

    called = {}

    def fake_apply(phrases, text, use_markers=True):
        called["called"] = True
        raise ValueError("boom")

    monkeypatch.setattr(pipeline, "apply_edited_text", fake_apply)

    with pytest.raises(ValueError, match="boom"):
        pipeline.revoice_video(
            str(dummy_video),
            str(tmp_path),
            speaker="spk",
            whisper_size="small",
            device="cpu",
            edited_text="text",
            phrases_cache=[(0.0, 1.0, "a"), (1.0, 2.0, "b")],
        )

    assert called.get("called")


def test_revoice_video_out_path_includes_engine_and_speaker(monkeypatch, tmp_path):
    dummy_video = tmp_path / "in.mp4"
    dummy_video.write_bytes(b"0")

    monkeypatch.setattr(pipeline, "ensure_ffmpeg", lambda: "ffmpeg")

    def fake_run(cmd):
        Path(cmd[-1]).touch()
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline, "run", fake_run)
    monkeypatch.setattr(pipeline, "_setup", lambda *a, **k: [(0.0, 1.0, "hi")])

    def fake_synth(ffmpeg, phrases, sr, speaker, tmp, tts_engine, **kwargs):
        voice = tmp / "voice.wav"
        voice.touch()
        return voice, None

    monkeypatch.setattr(pipeline, "synth_natural", fake_synth)

    out, _ = pipeline.revoice_video(
        str(dummy_video),
        str(tmp_path),
        speaker="spk!",
        whisper_size="small",
        device="cpu",
        tts_engine="eng@1",
    )

    safe_engine = re.sub(r"[^\w.-]", "", "eng@1")
    safe_voice = re.sub(r"[^\w.-]", "", "spk!")
    assert safe_engine in out
    assert safe_voice in out
