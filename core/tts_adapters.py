from __future__ import annotations

import base64
import json
import logging
import os
import wave
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import requests
import soundfile as sf
from pydub import AudioSegment

from . import model_service
from .tts_dependencies import ensure_tts_dependencies

__all__ = [
    "CoquiXTTS",
    "SileroTTS",
    "BeepTTS",
    "YandexTTS",
    "GTTSTTS",
    "resolve_model_path",
    "load_silero_model",
    "synthesize_silero",
    "synthesize_beep",
]


def resolve_model_path(*, parent: Any | None = None) -> Path:
    """Determine path to Silero model, downloading if necessary."""

    env_path = os.getenv("SILERO_MODEL")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    cfg = Path("config.json")
    if cfg.exists():
        try:
            with open(cfg, encoding="utf-8") as f:
                data = json.load(f)
            model_path = data.get("tts", {}).get("silero", {}).get("model_path")
            if model_path:
                p = Path(model_path)
                if p.exists():
                    return p
        except Exception:
            pass

    return model_service.get_model_path("silero", "tts", parent=parent)


def load_silero_model(model_path: str) -> tuple[Any, list[str], str]:
    """Load Silero model from ``model_path``.

    Returns a tuple ``(model, speakers, mode)``.
    """

    import torch

    model = torch.jit.load(model_path, map_location="cpu")
    if hasattr(model, "eval"):
        model.eval()

    speakers: list[str] = []
    mode = ""
    try:
        with ZipFile(model_path) as zf:
            if "speakers.json" in zf.namelist():
                speakers = json.loads(zf.read("speakers.json").decode("utf-8"))
            if "metadata.json" in zf.namelist():
                meta = json.loads(zf.read("metadata.json").decode("utf-8"))
                mode = meta.get("mode", "")
    except Exception:
        pass

    return model, speakers, mode


def synthesize_silero(
    text: str,
    speaker: str | None,
    sample_rate: int,
) -> bytes:
    """Synthesize speech with Silero TTS and return WAV bytes."""

    model_path = resolve_model_path()
    model, speakers, _ = load_silero_model(str(model_path))

    resolved = speaker or os.getenv("SILERO_SPEAKER")
    if resolved is None:
        cfg = Path("config.json")
        if cfg.exists():
            try:
                with open(cfg, encoding="utf-8") as f:
                    data = json.load(f)
                resolved = data.get("tts", {}).get("silero", {}).get("speaker")
            except Exception:
                resolved = None
    resolved = resolved or "aidar"
    if speakers and resolved not in speakers:
        resolved = speakers[0]

    try:
        wav = model.apply_tts(
            text=text or "",
            speaker=resolved,
            sample_rate=sample_rate,
        )
    except TypeError:
        wav = model.apply_tts(text or "", resolved, sample_rate)

    if not isinstance(wav, np.ndarray):
        wav = np.array(wav, dtype=np.float32)
    wav = wav.astype(np.float32)
    pcm16 = (np.clip(wav, -1.0, 1.0) * 32767).astype("<i2")

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def synthesize_beep(
    sample_rate: int = 48000,
    duration: float = 0.25,
    freq: float = 880.0,
) -> bytes:
    """Generate a simple sine beep and return WAV bytes."""

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave_arr = 0.5 * np.sin(2 * np.pi * freq * t)
    pcm16 = (np.clip(wave_arr, -1.0, 1.0) * 32767).astype("<i2")

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buffer.getvalue()


# --- XTTS v2 (Coqui) ---
class CoquiXTTS:
    _model = None

    def __init__(self, root: Path):
        self.root = Path(root)

    def _ensure_model(self, parent: Any | None = None):
        if CoquiXTTS._model is None:
            ensure_tts_dependencies("coqui_xtts")
            from TTS.api import TTS

            model_dir = model_service.get_model_path("coqui_xtts", "tts", parent=parent)
            # Load locally (offline)
            CoquiXTTS._model = TTS(model_path=str(model_dir))
        return CoquiXTTS._model

    def _load_refs(self, speaker_name: str) -> list[str]:
        sp_dir = self.root / "models" / "speakers" / speaker_name
        if not sp_dir.exists():
            return []
        wavs = sorted([str(p) for p in sp_dir.glob("*.wav")])
        return wavs

    def tts(
        self, text: str, speaker: str, sr: int = 48000, *, parent: Any | None = None
    ) -> np.ndarray:
        tts = self._ensure_model(parent=parent)
        refs = self._load_refs(speaker)
        # Если рефов нет — используем встроенный голос модели (speaker=None)
        wav = tts.tts(
            text=text or "",
            speaker_wav=refs if refs else None,
            language="ru",
            split_sentences=False,
        )
        # API Coqui обычно возвращает numpy.float32 [-1..1]
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)
        # Ресемплинг при необходимости
        if sr and sr != 24000:
            # модель может выдавать 24 кГц — ресемплим через soundfile+ffmpeg в твоём пайплайне;
            # тут оставим как есть, ресемплинг у тебя уже есть в synth_natural.
            pass
        return wav.astype(np.float32)


# --- Silero TTS ---
class SileroTTS:
    _model = None

    def __init__(self, root: Path):
        self.root = Path(root)

    def _ensure_model(self, parent: Any | None = None):
        if SileroTTS._model is None:
            ensure_tts_dependencies("silero")
            import torch
            logging.info("Using torch %s for Silero TTS", torch.__version__)
            model_path = resolve_model_path(parent=parent)
            model, speakers, mode = load_silero_model(str(model_path))
            SileroTTS._model = model
            SileroTTS._speakers = speakers
            SileroTTS._mode = mode
        if SileroTTS._model is None:
            raise RuntimeError("Failed to load Silero TTS model")
        return SileroTTS._model

    def tts(
        self, text: str, speaker: str, sr: int = 48000, *, parent: Any | None = None
    ) -> np.ndarray:
        model = self._ensure_model(parent=parent)
        wav = model.apply_tts(text=text or "", speaker=speaker or "baya", sample_rate=sr)
        if isinstance(wav, np.ndarray):
            arr = wav
        else:
            arr = wav.cpu().numpy() if hasattr(wav, "cpu") else np.array(wav)
        return arr.astype(np.float32)


# --- Yandex Cloud TTS ---
class YandexTTS:
    _url = "https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis"

    def tts(
        self,
        text: str,
        speaker: str,
        sr: int = 48000,
        *,
        key: str,
        folder_id: str | None = None,
    ) -> np.ndarray:
        """Synthesize speech using Yandex Cloud TTS v3."""
        headers = {"Content-Type": "application/json"}
        # Choose auth scheme depending on whether folder_id was supplied
        if folder_id:
            headers["Authorization"] = f"Bearer {key}"
            headers["x-folder-id"] = folder_id
        else:
            headers["Authorization"] = f"Api-Key {key}"
        payload = {
            "text": text or "",
            "voice": speaker,
            "lang": "ru-RU",
            "format": "lpcm",  # 16-bit LPCM
            "sampleRateHertz": sr,
        }
        resp = requests.post(self._url, headers=headers, json=payload, stream=True)
        resp.raise_for_status()
        pieces: list[np.ndarray] = []
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            chunk = obj.get("audioChunk")
            if not chunk:
                continue
            data = chunk.get("data")
            if not data:
                continue
            raw = base64.b64decode(data)
            # Convert 16-bit PCM to float32 in -1..1 range
            pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            pieces.append(pcm)
        resp.close()
        if not pieces:
            return np.array([], dtype=np.float32)
        return np.concatenate(pieces)


# --- gTTS (Google Text-to-Speech) ---
class GTTSTTS:
    """Use gTTS to generate speech and return it as a NumPy array."""

    def tts(self, text: str, speaker: str, sr: int = 48000) -> np.ndarray:
        """Synthesize speech via gTTS and resample to the desired rate."""
        ensure_tts_dependencies("gtts")
        from gtts import gTTS

        buf = BytesIO()
        gTTS(text=text or "", lang="ru").write_to_fp(buf)
        buf.seek(0)
        seg = AudioSegment.from_file(buf, format="mp3")
        seg = seg.set_frame_rate(sr).set_channels(1)
        wav_buf = BytesIO()
        seg.export(wav_buf, format="wav")
        wav_buf.seek(0)
        data, _ = sf.read(wav_buf, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32)


# --- Простейший TTS на синусоидах ---
class BeepTTS:
    """Очень простая реализация TTS.

    Генерирует синусоиду для гласных и шум для согласных. Конечно, это не
    человеческая речь, но в тестах важно лишь наличие несущего аудио, а не
    качество синтеза.
    """

    _vowel_freqs = {
        "a": 440.0,
        "e": 660.0,
        "i": 880.0,
        "o": 550.0,
        "u": 770.0,
        "y": 720.0,
        "а": 440.0,
        "е": 660.0,
        "и": 880.0,
        "о": 550.0,
        "у": 770.0,
        "ы": 720.0,
        "э": 600.0,
        "ю": 840.0,
        "я": 620.0,
    }

    def tts(self, text: str, speaker: str, sr: int = 48000) -> np.ndarray:
        dur = 0.15  # длина одного "звука" в секундах
        base_t = np.linspace(0, dur, int(sr * dur), False)
        pieces: list[np.ndarray] = []
        for ch in text.lower():
            if ch in self._vowel_freqs:
                freq = self._vowel_freqs[ch]
                pieces.append(0.2 * np.sin(2 * np.pi * freq * base_t))
            elif ch.isspace():
                pieces.append(np.zeros(int(sr * 0.1)))
            else:
                # Согласные – белый шум
                pieces.append(0.05 * np.random.randn(len(base_t)))
        if not pieces:
            return np.zeros(int(sr * 0.3), dtype=np.float32)
        return np.concatenate(pieces).astype(np.float32)
