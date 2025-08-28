# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import soundfile as sf

__all__ = ["CoquiXTTS", "KokoroTTS", "SileroTTS", "BeepTTS"]

# --- XTTS v2 (Coqui) ---
class CoquiXTTS:
    _model = None

    def __init__(self, root: Path):
        self.root = Path(root)
        self.model_dir = self.root / "models" / "tts" / "coqui_xtts"

    def _ensure_model(self):
        if CoquiXTTS._model is None:
            from TTS.api import TTS
            # Загружаем локально (без интернета)
            CoquiXTTS._model = TTS(model_path=str(self.model_dir))
        return CoquiXTTS._model

    def _load_refs(self, speaker_name: str) -> list[str]:
        sp_dir = self.root / "models" / "speakers" / speaker_name
        if not sp_dir.exists():
            return []
        wavs = sorted([str(p) for p in sp_dir.glob("*.wav")])
        return wavs

    def tts(self, text: str, speaker: str, sr: int = 48000) -> np.ndarray:
        tts = self._ensure_model()
        refs = self._load_refs(speaker)
        # Если рефов нет — используем встроенный голос модели (speaker=None)
        wav = tts.tts(
            text=text or "",
            speaker_wav=refs if refs else None,
            language="ru",
            split_sentences=False
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

# --- Kokoro (очень лёгкая) ---
class KokoroTTS:
    _model = None

    def __init__(self, root: Path):
        self.root = Path(root)
        self.model_dir = self.root / "models" / "tts" / "kokoro"

    def _ensure_model(self):
        if KokoroTTS._model is None:
            import kokoro
            KokoroTTS._model = kokoro.load(self.model_dir)  # модель из локальной папки
        return KokoroTTS._model

    def tts(self, text: str, speaker: str, sr: int = 48000) -> np.ndarray:
        model = self._ensure_model()
        wav = model.tts(text or "", language="ru")  # укажем ru; при необходимости выберем en/ja
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)
        return wav.astype(np.float32)

# --- Silero TTS ---
class SileroTTS:
    _model = None

    def __init__(self, root: Path):
        self.root = Path(root)
        self.model_dir = self.root / "models" / "tts" / "silero"

    def _ensure_model(self):
        if SileroTTS._model is None:
            import torch
            pt_files = sorted(self.model_dir.glob("*.pt"))
            if not pt_files:
                raise FileNotFoundError(f"Silero model (.pt) not found in {self.model_dir}")
            model_path = pt_files[0]
            model = torch.package.PackageImporter(str(model_path)).load_pickle("tts_models", "model")
            model.to("cpu")
            if hasattr(model, "eval"):
                model.eval()  # Some model versions do not provide eval()
            SileroTTS._model = model
        if SileroTTS._model is None:
            raise RuntimeError("Failed to load Silero TTS model")
        return SileroTTS._model

    def tts(self, text: str, speaker: str, sr: int = 48000) -> np.ndarray:
        model = self._ensure_model()
        wav = model.apply_tts(text=text or "", speaker=speaker or "baya", sample_rate=sr)
        if isinstance(wav, np.ndarray):
            arr = wav
        else:
            arr = wav.cpu().numpy() if hasattr(wav, "cpu") else np.array(wav)
        return arr.astype(np.float32)


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
