# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import soundfile as sf

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
