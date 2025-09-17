from __future__ import annotations

import base64
import json
import logging
import os
import ssl
import time
import wave
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import URLError

import numpy as np
import requests
import soundfile as sf
from pydub import AudioSegment

from . import model_service
from .model_manager import QMessageBox
from .pkg_installer import ensure_package

SILERO_LANG_MODELS = {
    "ru": "v4_ru",
    "en": "v3_en",
    "de": "v3_de",
}

SILERO_VOICES = {
    "ru": ["aidar", "baya", "kseniya", "xenia", "eugene"],
    "en": [
        "en_0",
        "en_1",
        "en_2",
        "en_3",
        "en_4",
        "en_5",
        "en_6",
        "en_7",
    ],
    "de": ["de_0", "de_1", "de_2"],
}

SILERO_PT_FILES = {
    "ru": "v4_ru.pt",
    "en": "v3_en.pt",
    "de": "v3_de.pt",
}

_SSL_VERIFY_DOWNLOADS = True


def _initial_ssl_flag() -> bool:
    verify = True
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    try:
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            raw = data.get("verify_ssl_downloads")
            if isinstance(raw, bool):
                verify = raw
            elif isinstance(raw, str):
                verify = raw.strip().lower() not in {"0", "false", "no"}
    except Exception:
        verify = True
    if os.environ.get("NO_SSL_VERIFY") == "1":
        verify = False
    if not verify:
        os.environ["NO_SSL_VERIFY"] = "1"
    return verify


_SSL_VERIFY_DOWNLOADS = _initial_ssl_flag()


def set_ssl_verification(enabled: bool) -> None:
    """Globally enable or disable SSL verification for model downloads."""

    global _SSL_VERIFY_DOWNLOADS
    _SSL_VERIFY_DOWNLOADS = bool(enabled)
    if enabled:
        os.environ.pop("NO_SSL_VERIFY", None)
    else:
        os.environ["NO_SSL_VERIFY"] = "1"


def ssl_verification_enabled() -> bool:
    """Return the current SSL verification preference."""

    if os.environ.get("NO_SSL_VERIFY") == "1":
        return False
    return _SSL_VERIFY_DOWNLOADS


__all__ = [
    "CoquiXTTS",
    "SileroTTS",
    "BeepTTS",
    "YandexTTS",
    "GTTSTTS",
    "synthesize_beep",
    "SILERO_VOICES",
    "set_ssl_verification",
    "ssl_verification_enabled",
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
            try:
                ensure_package("TTS", "TTS is required for coqui_xtts.")
                from TTS.api import TTS
            except ModuleNotFoundError as exc:  # pragma: no cover - user interaction
                QMessageBox.warning(parent, "Missing dependency", "TTS is required for coqui_xtts.")
                raise RuntimeError("TTS is required for coqui_xtts") from exc
            except ImportError as exc:  # pragma: no cover - defensive
                QMessageBox.warning(parent, "Import error", str(exc))
                raise RuntimeError(f"Failed to import TTS: {exc}") from exc

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
    _models: dict[str, Any] = {}
    _statuses: dict[str, str | None] = {}
    _speakers: dict[str, list[str]] = {}
    _mode = None

    def __init__(
        self,
        auto_download: bool = True,
        language: str = "ru",
        verify_ssl: bool | None = None,
    ):
        self.auto_download = auto_download
        self.language = language
        self._verify_ssl = verify_ssl

    def _ensure_model(
        self,
        auto_download: bool = True,
        parent: Any | None = None,
        *,
        return_status: bool = False,
    ):
        lang = self.language
        if lang not in SileroTTS._models:
            try:
                ensure_package("torch", "torch is required for Silero TTS.")
                ensure_package("torchaudio", "torchaudio is required for Silero TTS.")
                ensure_package("omegaconf", "omegaconf is required for Silero TTS.")
                import torch
            except ModuleNotFoundError as exc:  # pragma: no cover - user interaction
                QMessageBox.warning(parent, "Missing dependency", str(exc))
                raise RuntimeError(str(exc)) from exc
            except ImportError as exc:  # pragma: no cover - defensive
                QMessageBox.warning(parent, "Import error", str(exc))
                raise RuntimeError(str(exc)) from exc

            torch.set_num_threads(max(1, os.cpu_count() // 2))
            torch_home = Path("models") / "torch_hub"
            torch_home.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(torch_home))
            hub_dir = Path(torch.hub.get_dir())
            original_https_context = None
            verify_ssl = self._verify_ssl if self._verify_ssl is not None else ssl_verification_enabled()
            if os.environ.get("NO_SSL_VERIFY") == "1":
                verify_ssl = False
            if not verify_ssl:
                original_https_context = ssl._create_default_https_context
                ssl._create_default_https_context = ssl._create_unverified_context
            try:
                cache_dir = hub_dir / "snakers4_silero-models_master"
                pt_name = SILERO_PT_FILES.get(lang)
                model_path = cache_dir / "src" / "silero" / "model" / pt_name if pt_name else None
                cached_before = model_path.exists() if model_path else False
                old_autofetch = os.environ.get("TORCH_HUB_DISABLE_AUTOFETCH")
                if cached_before or not auto_download:
                    os.environ["TORCH_HUB_DISABLE_AUTOFETCH"] = "1"
                try:
                    if not auto_download and not cached_before:
                        raise RuntimeError(
                            "Silero model files not found. Enable 'Auto-download models' in Settings or prefetch via CLI."
                        )
                    logging.info("Using torch %s for Silero TTS", torch.__version__)
                    attempts = 3
                    for attempt in range(1, attempts + 1):
                        try:
                            load_kwargs = {
                                "repo_or_dir": str(cache_dir)
                                if cached_before
                                else "snakers4/silero-models",
                                "model": "silero_tts",
                                "language": lang,
                                "speaker": SILERO_LANG_MODELS.get(lang, "v4_ru"),
                                "trust_repo": True,
                                "force_reload": False,
                            }
                            if cached_before:
                                load_kwargs["source"] = "local"
                            model, _ = torch.hub.load(**load_kwargs)
                            break
                        except (URLError, ssl.SSLError) as e:
                            logging.warning("torch.hub.load failed (%s/%s): %s", attempt, attempts, e)
                            if attempt == attempts:
                                msg = (
                                    "Silero download failed: Run `python tools/fetch_tts_models.py silero --language <code>` or check internet connection. "
                                    "Check SSL_CERT_FILE, HTTPS_PROXY, or set NO_SSL_VERIFY=1 to disable SSL verification."
                                )
                                logging.debug("Silero download failed", exc_info=True)
                                try:
                                    from .pipeline import TTSEngineError  # type: ignore
                                except Exception:
                                    raise RuntimeError(msg) from e
                                raise TTSEngineError(msg) from e
                            time.sleep(1)
                        except Exception as e:
                            logging.info("torch.hub.load failed: %s", e)
                            logging.debug("Silero download failed", exc_info=True)
                            try:
                                from .pipeline import TTSEngineError  # type: ignore
                            except Exception:
                                raise RuntimeError(f"Silero download failed: {e}") from e
                            raise TTSEngineError(f"Silero download failed: {e}") from e
                    else:  # pragma: no cover - defensive
                        raise RuntimeError("Silero model load unexpectedly failed without exception")
                    model.to(torch.device("cpu"))
                    SileroTTS._models[lang] = model
                    SileroTTS._speakers[lang] = getattr(model, "speakers", [])
                    SileroTTS._mode = "offline"
                    status = "cached" if cached_before else "downloaded"
                    SileroTTS._statuses[lang] = status
                    logging.info("tts.silero ensure status=%s cache_dir=%s", status, cache_dir)
                finally:
                    if old_autofetch is None:
                        os.environ.pop("TORCH_HUB_DISABLE_AUTOFETCH", None)
                    else:
                        os.environ["TORCH_HUB_DISABLE_AUTOFETCH"] = old_autofetch
            finally:
                if original_https_context is not None:
                    ssl._create_default_https_context = original_https_context
        model = SileroTTS._models[lang]
        status = SileroTTS._statuses.get(lang)
        return (model, status) if return_status else model

    def tts(
        self,
        text: str,
        speaker: str,
        sr: int = 48000,
        *,
        parent: Any | None = None,
        rate: float | None = None,
        pitch: float | None = None,
        style: str | None = None,
        preset: str | None = None,
    ) -> np.ndarray:
        model = self._ensure_model(auto_download=self.auto_download, parent=parent)
        default_voice = SILERO_VOICES.get(self.language, ["baya"])[0]
        voice = speaker or default_voice
        kwargs: dict[str, Any] = {
            "text": text or "",
            "speaker": voice,
            "sample_rate": sr,
        }
        if rate is not None:
            kwargs["rate"] = rate
        if pitch is not None:
            kwargs["pitch"] = pitch
        if style is not None:
            kwargs["style_wav"] = style
        if preset is not None:
            kwargs["preset"] = preset
        wav = model.apply_tts(**kwargs)
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
        try:
            ensure_package("gTTS", "gTTS is required for gTTS engine.")
            from gtts import gTTS
        except ModuleNotFoundError as exc:  # pragma: no cover - user interaction
            QMessageBox.warning(None, "Missing dependency", "gTTS is required for gTTS engine.")
            raise RuntimeError("gTTS is required for gTTS") from exc
        except ImportError as exc:  # pragma: no cover - defensive
            QMessageBox.warning(None, "Import error", str(exc))
            raise RuntimeError(f"Failed to import gTTS: {exc}") from exc

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
