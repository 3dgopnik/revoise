from __future__ import annotations

import base64
import json
import logging
import os
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

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
    "synthesize_beep",
]


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
    _status: str | None = None

    def __init__(self, root: Path, auto_download: bool = True):
        self.root = Path(root)
        self.auto_download = auto_download

    def _ensure_model(
        self,
        auto_download: bool = True,
        parent: Any | None = None,
        *,
        return_status: bool = False,
    ):
        if SileroTTS._model is None:
            ensure_tts_dependencies("silero")
            import torch

            torch.set_num_threads(max(1, os.cpu_count() // 2))
            old_autofetch = os.environ.get("TORCH_HUB_DISABLE_AUTOFETCH")
            if not auto_download:
                os.environ["TORCH_HUB_DISABLE_AUTOFETCH"] = "1"
            try:
                hub_dir = Path(torch.hub.get_dir())
                cache_dir = hub_dir / "snakers4_silero-models_master"
                cached_before = cache_dir.exists()
                if not auto_download and not cached_before:
                    raise RuntimeError(
                        "Silero model cache not found. Enable 'Auto-download models' in Settings or prefetch via CLI."
                    )
                logging.info("Using torch %s for Silero TTS", torch.__version__)
                model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-models",
                    model="silero_tts",
                    language="ru",
                    speaker="v4_ru",
                    trust_repo=True,
                    force_reload=False,
                )
                model.to(torch.device("cpu"))
                SileroTTS._model = model
                SileroTTS._speakers = getattr(model, "speakers", [])
                SileroTTS._mode = "offline"
                status = "cached" if cached_before else "downloaded"
                SileroTTS._status = status
                logging.info("tts.silero ensure status=%s cache_dir=%s", status, cache_dir)
            finally:
                if old_autofetch is None:
                    os.environ.pop("TORCH_HUB_DISABLE_AUTOFETCH", None)
                else:
                    os.environ["TORCH_HUB_DISABLE_AUTOFETCH"] = old_autofetch
        model = SileroTTS._model
        return (model, SileroTTS._status) if return_status else model

    def tts(
        self, text: str, speaker: str, sr: int = 48000, *, parent: Any | None = None
    ) -> np.ndarray:
        model = self._ensure_model(
            auto_download=self.auto_download, parent=parent
        )
        wav = model.apply_tts(
            text=text or "", speaker=speaker or "baya", sample_rate=sr
        )
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
