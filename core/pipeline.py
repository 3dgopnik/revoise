# ruff: noqa: UP006,UP007,UP022,UP035,UP045
import gc
import json
import logging
import random
import re
import shutil
import subprocess
import tempfile
import time
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
import soundfile as sf
from num2words import num2words
from tqdm import tqdm

from . import externals, model_service
from .model_manager import DownloadError, QMessageBox
from .pkg_installer import ensure_package
from .tts.registry import get_engine
from .tts_adapters import GTTSTTS, BeepTTS, CoquiXTTS, SileroTTS, YandexTTS

logger = logging.getLogger(__name__)


class TTSEngineError(RuntimeError):
    """Raised when the selected TTS engine cannot be used."""


def check_engine_available(engine_name: str, auto_download_models: bool = True) -> None:
    """Validate that the requested TTS engine can run."""
    global torch_unavailable
    try:
        if engine_name == "silero":
            ensure_package("torch", "torch is required for silero.")
            ensure_package("torchaudio", "torchaudio is required for silero.")
            ensure_package("omegaconf", "omegaconf is required for silero.")
            SileroTTS(auto_download=auto_download_models)._ensure_model(
                auto_download=auto_download_models
            )
        elif engine_name == "coqui_xtts":
            ensure_package("TTS", "TTS is required for coqui_xtts.")
            ensure_package("torch", "torch is required for coqui_xtts.")
            model_service.get_model_path("coqui_xtts", "tts")
        elif engine_name == "gtts":
            ensure_package("gTTS", "gTTS is required for gtts.")
        elif engine_name == "yandex":
            pass
        elif engine_name == "vibevoice":
            if shutil.which("vibe-voice") is None:
                raise FileNotFoundError("vibe-voice binary not found. Install VibeVoice.")
        else:
            ensure_package(engine_name, f"{engine_name} is required.")
    except ModuleNotFoundError as e:
        if "torch" in str(e).lower():
            torch_unavailable = True
            QMessageBox.warning(None, "Missing dependency", TORCH_MISSING_MSG)
            raise TTSEngineError(TORCH_MISSING_MSG) from e
        QMessageBox.warning(None, "Missing dependency", str(e))
        raise TTSEngineError(str(e)) from e
    except (
        FileNotFoundError,
        DownloadError,
        RuntimeError,
    ) as e:
        QMessageBox.warning(None, "Missing dependency", str(e))
        raise TTSEngineError(str(e)) from e


# ===================== Global =====================
FWHISPER: Any | None = None
TTS_MODEL = None
torch_unavailable = False
TORCH_MISSING_MSG = (
    "Torch is unavailable. Steps: check Python version, reinstall torch CPU, clear uv cache."
)

MULTISPACE = re.compile(r"\s+")
PAUSE_TAG = re.compile(r"\[\[\s*PAUSE\s*=\s*(\d+)\s*\]\]", re.IGNORECASE)


# ===================== Утилиты =====================
def run(cmd: List[str]):
    logger.debug("Run: %s", " ".join(cmd))
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.debug("STDOUT: %s", r.stdout)  # capture normal output
        logger.debug("STDERR: %s", r.stderr)  # capture warnings/errors
        if r.returncode != 0:
            raise RuntimeError(
                f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )
        return r
    except Exception:
        logger.exception("Run failed")
        raise


def ensure_ffmpeg() -> str:
    path, version = externals.ensure_ffmpeg()
    cfg_path = Path("config.json")
    config: dict = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as fh:
            config = json.load(fh)
    config.setdefault("externals", {})["ffmpeg"] = str(path)
    config["ffmpeg_version"] = version
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    return str(path)


# ===================== Whisper =====================
def transcribe_whisper(
    audio_wav: Path,
    language="ru",
    model_size="large-v3",
    device="cuda",
    *,
    parent: Any | None = None,
):
    global FWHISPER
    logger.debug("Starting transcribe_whisper for %s", audio_wav)
    try:
        ensure_package("faster-whisper", "faster-whisper is required for transcription.")
        from faster_whisper import WhisperModel

        need_load = (FWHISPER is None) or getattr(FWHISPER, "_name", "") != model_size
        if need_load:
            logger.info("Ensuring Whisper model %s is available", model_size)
            try:
                model_dir = model_service.get_model_path(
                    model_size,
                    "stt",
                    parent=parent,
                )
            except FileNotFoundError as exc:
                logger.error("Model download declined: %s", exc)
                raise RuntimeError("Whisper model download was declined") from exc
            except DownloadError as exc:
                logger.error("Model download failed: %s", exc)
                raise RuntimeError("Whisper model download failed") from exc
            logger.info("Loading Whisper model %s from %s on %s", model_size, model_dir, device)
            compute_type = "int8_float16" if device == "cuda" else "int8"
            FWHISPER = WhisperModel(str(model_dir), device=device, compute_type=compute_type)
            FWHISPER._name = model_size
            logger.info("Whisper model %s initialized", model_size)
        assert FWHISPER is not None
        segments, _ = FWHISPER.transcribe(
            str(audio_wav),
            language=language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        result = [(s.start, s.end, s.text.strip()) for s in segments]
        logger.debug("Finished transcribe_whisper with %d segments", len(result))
        return result
    except ModuleNotFoundError as exc:  # pragma: no cover - user interaction
        QMessageBox.warning(
            parent, "Missing dependency", "faster-whisper is required for transcription."
        )
        logger.exception("Whisper transcription failed")
        raise RuntimeError("faster-whisper is required") from exc
    except Exception:
        logger.exception("Whisper transcription failed")
        raise


# ===================== Фразы =====================
def merge_into_phrases(segments: List[Tuple[float, float, str]], max_gap=0.35, min_len=0.8):
    """Merge short segments into longer phrases.

    Segments are sorted by their start time to ensure chronological processing.

    Parameters
    ----------
    segments : List[Tuple[float, float, str]]
        Input segments as (start, end, text) tuples.
    max_gap : float, optional
        Maximum allowed pause between segments to merge them.
    min_len : float, optional
        Minimum phrase length before a gap can split phrases.

    Returns
    -------
    List[Tuple[float, float, str]]
        Merged phrases sorted by start time.
    """

    phrases: List[Tuple[float, float, str]] = []
    if not segments:
        return phrases

    # Sort segments by start time to handle unsorted input
    segments = sorted(segments, key=lambda s: s[0])

    cs, ce, ct = segments[0]
    for s, e, t in segments[1:]:
        gap = s - ce
        if gap <= max_gap or (ce - cs) < min_len:
            ce, ct = e, (ct + " " + t).strip()
        else:
            phrases.append((cs, ce, MULTISPACE.sub(" ", ct).strip()))
            cs, ce, ct = s, e, t
    phrases.append((cs, ce, MULTISPACE.sub(" ", ct).strip()))
    return phrases


def phrases_to_marked_text(phrases: List[Tuple[float, float, str]]) -> str:
    lines = []
    for i, (_, _, t) in enumerate(phrases, start=1):
        lines.append(f"[[#{i}]] {t}")
    return "\n".join(lines)


def apply_edited_text(
    phrases: List[Tuple[float, float, str]],
    edited_text: str,
    *,
    use_markers: bool = True,
) -> List[Tuple[float, float, str]]:
    """Apply user edits to phrases.

    Parameters
    ----------
    phrases:
        Original phrases with timings.
    edited_text:
        User edited text. Each line corresponds to a phrase.
    use_markers:
        Whether ``[[#i]]`` markers are present in the text.

    Notes
    -----
    If ``edited_text`` contains fewer lines than there are phrases, the
    remaining phrases keep their original text. Extra lines in
    ``edited_text`` are ignored.

    Returns
    -------
    List[Tuple[float, float, str]]
        Phrases with updated text.
    """

    lines = [ln.strip() for ln in edited_text.strip().splitlines() if ln.strip()]

    updated: List[Tuple[float, float, str]] = []
    if use_markers:
        marker_re = re.compile(r"\s*\[\[#\d+\]\]\s*(.*)")
        for phrase, line in zip_longest(phrases, lines):
            if phrase is None:
                break
            start, end, original = phrase
            if line is None:
                text = original
            else:
                m = marker_re.fullmatch(line)
                text = m.group(1).strip() if m else line
            updated.append((start, end, text))
    else:
        for phrase, line in zip_longest(phrases, lines):
            if phrase is None:
                break
            start, end, original = phrase
            text = original if line is None else line
            updated.append((start, end, text))
    return updated


def _setup(
    wav: Path,
    whisper_size: str,
    device: str,
    phrases_cache: Optional[List[Tuple[float, float, str]]],
    edited_text: Optional[str],
    *,
    use_markers: bool,
) -> List[Tuple[float, float, str]]:
    """Prepare phrases: transcribe if needed and apply edited text."""

    logger.debug(
        "Starting _setup with cache=%s edited=%s",
        phrases_cache is not None,
        bool(edited_text),
    )
    try:
        if phrases_cache is None:
            logger.info("Transcribing audio %s", wav)
            segs = transcribe_whisper(wav, language="ru", model_size=whisper_size, device=device)
            if not segs:
                raise RuntimeError("Речь не обнаружена.")
            phrases = merge_into_phrases(segs, max_gap=0.35, min_len=0.8)
            logger.debug("Merged into %d phrases", len(phrases))
        else:
            logger.debug("Using cached phrases")
            phrases = phrases_cache

        if edited_text:
            logger.debug("Applying edited text")
            phrases = apply_edited_text(phrases, edited_text, use_markers=use_markers)

        logger.debug("_setup produced %d phrases", len(phrases))
        return phrases
    except Exception:
        logger.exception("_setup failed")
        raise


def normalize_text(text: str, *, read_numbers: bool = False, spell_latin: bool = False) -> str:
    """Рудиментарная нормализация текста для TTS."""

    def _num(match: re.Match[str]) -> str:
        try:
            return num2words(int(match.group(0)), lang="ru")
        except Exception:
            return match.group(0)

    if read_numbers:
        text = re.sub(r"\d+", _num, text)
    if spell_latin:
        text = re.sub(r"[A-Za-z]+", lambda m: " ".join(list(m.group())), text)
    return text


# ===================== TTS-заглушки =====================
# Здесь остаются твои текущие реализации synth_natural и synth_chunk.
# Если используется Silero/Yandex/XTTS — они будут вызывать normalize_text с read_numbers/spell_latin.


def _synth_chunk_single(
    ffmpeg: str,
    text: str,
    sr: int,
    speaker: str,
    tmpdir: Path,
    tts_engine: str | None,
    language: str = "ru",
    read_numbers: bool = False,
    spell_latin: bool = False,
    yandex_key: str | None = None,
    yandex_voice: str | None = None,
    allow_beep_fallback: bool = False,
    auto_download_models: bool = True,
    tts_rate: float | None = None,
    tts_pitch: float | None = None,
    tts_style: str | None = None,
    tts_preset: str | None = None,
    *,
    silence_gap_ms: int = 0,
) -> tuple[np.ndarray, str | None]:
    """Internal helper that synthesizes a single phrase."""

    text = normalize_text(text, read_numbers=read_numbers, spell_latin=spell_latin)
    engine_name = (tts_engine or "silero").lower()
    logger.debug("Synthesizing chunk with engine=%s", engine_name)

    # Deterministic per-speaker seeding for consistent timbre
    seed = int(sha256(speaker.encode("utf-8")).hexdigest(), 16) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - torch may be unavailable
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
    fallback_reason: str | None = None
    try:
        check_engine_available(engine_name, auto_download_models=auto_download_models)
    except TTSEngineError as e:
        if not allow_beep_fallback:
            logger.info("tts.engine=%s fallback=false reason=\"%s\"", engine_name, e)
            raise
        logger.info("tts.engine=%s fallback=true reason=\"%s\"", engine_name, e)
        wav = BeepTTS().tts(text, speaker, sr=sr)
        model_sr = sr
        fallback_reason = str(e)
    else:
        try:
            if engine_name == "coqui_xtts":
                wav = CoquiXTTS(Path(__file__).resolve().parent.parent).tts(text, speaker, sr=24000)
                model_sr = 24000
            elif engine_name == "gtts":
                wav = GTTSTTS().tts(text, speaker, sr=sr)
                model_sr = sr
            elif engine_name == "yandex":
                if not yandex_key or not (yandex_voice or speaker):
                    raise ValueError("Yandex TTS requires yandex_key and yandex_voice")
                voice = yandex_voice or speaker
                wav = YandexTTS().tts(text, voice, sr=sr, key=yandex_key)
                model_sr = sr
            else:
                engine = get_engine(engine_name)
                wav = engine.synthesize(
                    text,
                    speaker,
                    sr,
                    rate=tts_rate,
                    pitch=tts_pitch,
                    style=tts_style,
                    preset=tts_preset,
                )
                model_sr = sr
        except (
            TypeError,
            RuntimeError,
            ImportError,
            ModuleNotFoundError,
        ) as e:
            if not allow_beep_fallback:
                logger.info("tts.engine=%s fallback=false reason=\"%s\"", engine_name, e)
                raise TTSEngineError(str(e)) from e
            logger.info("tts.engine=%s fallback=true reason=\"%s\"", engine_name, e)
            wav = BeepTTS().tts(text, speaker, sr=sr)
            model_sr = sr
            fallback_reason = str(e)

    raw = tmpdir / "tts_raw.wav"
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 1 or (wav.ndim == 2 and wav.shape[1] == 0):
        wav = wav.reshape(-1, 1)
    sf.write(raw, wav, model_sr)
    out = tmpdir / "tts.wav"
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(raw),
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-c:a",
            "pcm_s16le",
            str(out),
        ]
    )
    wav_out, _ = sf.read(out, dtype=np.float32)
    if silence_gap_ms > 0:
        gap = np.zeros(int(sr * silence_gap_ms / 1000), dtype=np.float32)
        wav_out = np.concatenate([wav_out, gap])
    if not fallback_reason:
        peak = float(np.max(np.abs(wav_out)))
        rms = float(np.sqrt(np.mean(np.square(wav_out))))
        logger.info("tts.engine=%s fallback=false peak=%.4f rms=%.4f", engine_name, peak, rms)
    logger.debug("synth_chunk produced %d samples", len(wav_out))
    return wav_out, fallback_reason


def synth_chunk(
    ffmpeg: str,
    text: str | Mapping[str, str],
    sr: int,
    speaker: str,
    tmpdir: Path,
    tts_engine: str | None,
    language: str = "ru",
    read_numbers: bool = False,
    spell_latin: bool = False,
    yandex_key: str | None = None,
    yandex_voice: str | None = None,
    allow_beep_fallback: bool = False,
    auto_download_models: bool = True,
    tts_rate: float | None = None,
    tts_pitch: float | None = None,
    tts_style: str | None = None,
    tts_preset: str | None = None,
    *,
    silence_gap_ms: int = 0,
) -> tuple[Any, str | None]:
    """Generate audio for one or multiple language variants."""
    global torch_unavailable

    def _beep_response(txt: str | Mapping[str, str]) -> Any:
        if isinstance(txt, Mapping):
            return {lang: BeepTTS().tts(v, speaker, sr=sr) for lang, v in txt.items()}
        return BeepTTS().tts(txt, speaker, sr=sr)

    if torch_unavailable:
        logger.warning("GPU acceleration unavailable")
        return _beep_response(text), TORCH_MISSING_MSG

    try:
        if isinstance(text, Mapping):
            wavs: dict[str, np.ndarray] = {}
            fallback_reason: str | None = None
            for lang, variant in text.items():
                wav, reason = _synth_chunk_single(
                    ffmpeg,
                    variant,
                    sr,
                    speaker,
                    tmpdir,
                    tts_engine,
                    language=lang,
                    read_numbers=read_numbers,
                    spell_latin=spell_latin,
                    yandex_key=yandex_key,
                    yandex_voice=yandex_voice,
                    allow_beep_fallback=allow_beep_fallback,
                    auto_download_models=auto_download_models,
                    tts_rate=tts_rate,
                    tts_pitch=tts_pitch,
                    tts_style=tts_style,
                    tts_preset=tts_preset,
                    silence_gap_ms=silence_gap_ms,
                )
                wavs[lang] = wav
                if reason and not fallback_reason:
                    fallback_reason = reason
            return wavs, fallback_reason

        return _synth_chunk_single(
            ffmpeg,
            text,
            sr,
            speaker,
            tmpdir,
            tts_engine,
            language=language,
            read_numbers=read_numbers,
            spell_latin=spell_latin,
            yandex_key=yandex_key,
            yandex_voice=yandex_voice,
            allow_beep_fallback=allow_beep_fallback,
            auto_download_models=auto_download_models,
            tts_rate=tts_rate,
            tts_pitch=tts_pitch,
            tts_style=tts_style,
            tts_preset=tts_preset,
            silence_gap_ms=silence_gap_ms,
        )
    except TTSEngineError as e:
        if torch_unavailable:
            logger.warning("GPU acceleration unavailable")
            return _beep_response(text), str(e)
        raise


def synth_natural(
    ffmpeg: str,
    phrases: list[tuple[float, float, str]],
    sr: int,
    speaker: str,
    tmpdir: Path,
    tts_engine: str | None,
    language: str = "ru",
    yandex_key: str | None = None,
    yandex_voice: str | None = None,
    min_gap_sec: float = 0.30,
    overall_speed: float = 1.0,
    read_numbers: bool = False,
    spell_latin: bool = False,
    speed_jitter: float = 0.03,
    allow_beep_fallback: bool = False,
    auto_download_models: bool = True,
    tts_rate: float | None = None,
    tts_pitch: float | None = None,
    tts_style: str | None = None,
    tts_preset: str | None = None,
    *,
    silence_gap_ms: int = 0,
    autosave_minutes: float | None = None,
    checkpoint_path: Path | None = None,
    force_offload: bool = False,
) -> tuple[Path, str | None]:
    """
    Simple synthesis: calls synth_chunk for each phrase.
    """
    if not phrases:
        raise ValueError("No phrases to synthesize")
    logger.info("Starting synth_natural for %d phrases", len(phrases))
    total_dur = max(p[1] for p in phrases) + 3.0
    checkpoint = checkpoint_path or (tmpdir / "tts_autosave.npz")
    start_idx = 0
    cur_tail = 0.0
    if checkpoint.exists():
        try:
            data = np.load(checkpoint, allow_pickle=True)
            master = data["master"]
            cur_tail = float(data["cur_tail"])
            start_idx = int(data["index"])
            logger.info("Resuming from checkpoint %s at phrase %d", checkpoint, start_idx)
        except Exception:
            master = np.zeros(int(total_dur * sr), dtype=np.float32)
            logger.warning("Failed to load checkpoint, starting from scratch")
    else:
        master = np.zeros(int(total_dur * sr), dtype=np.float32)

    start_time = time.perf_counter()
    try:
        if "torch" in globals():  # pragma: no cover - torch may be absent
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        fallback_reason: str | None = None
        save_interval = (autosave_minutes or 0) * 60
        last_save = time.perf_counter()
        for idx in tqdm(range(start_idx, len(phrases)), desc="TTS", unit="phr", initial=start_idx, total=len(phrases)):
            start, _end, txt = phrases[idx]
            logger.debug("Synthesizing phrase %d", idx + 1)
            try:
                wav, reason = synth_chunk(
                    ffmpeg,
                    txt,
                    sr,
                    speaker,
                    tmpdir,
                    tts_engine,
                    language,
                    read_numbers=read_numbers,
                    spell_latin=spell_latin,
                    yandex_key=yandex_key,
                    yandex_voice=yandex_voice,
                    allow_beep_fallback=allow_beep_fallback,
                    auto_download_models=auto_download_models,
                    tts_rate=tts_rate,
                    tts_pitch=tts_pitch,
                    tts_style=tts_style,
                    tts_preset=tts_preset,
                    silence_gap_ms=silence_gap_ms,
                )
                if reason and not fallback_reason:
                    fallback_reason = reason
            except Exception:
                logger.exception("synth_chunk failed for phrase %d", idx + 1)
                raise
            place_t = max(start, cur_tail + min_gap_sec)
            s0 = int(place_t * sr)
            s1 = min(len(master), s0 + len(wav))
            if s0 < len(master):
                master[s0:s1] += wav[: (s1 - s0)]
            cur_tail = (s0 + len(wav)) / sr
            if save_interval and time.perf_counter() - last_save >= save_interval:
                np.savez(checkpoint, master=master, cur_tail=cur_tail, index=idx + 1)
                last_save = time.perf_counter()
        out_wav = tmpdir / "voice_aligned.wav"
        sf.write(out_wav, master, sr)
        logger.info("Finished synth_natural: %s", out_wav)
        return out_wav, fallback_reason
    except Exception:
        logger.exception("synth_natural failed")
        raise
    finally:
        if checkpoint.exists():
            checkpoint.unlink(missing_ok=True)  # type: ignore[attr-defined]
        elapsed = time.perf_counter() - start_time
        peak = 0.0
        try:  # pragma: no cover - torch may be absent
            import torch

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / (1024**2)
                if force_offload:
                    torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        logger.info("tts.performance peak_vram=%.1fMB elapsed=%.2fs", peak, elapsed)


# ===================== Основной пайплайн =====================
def revoice_video(
    video: str,
    outdir: str,
    speaker: str,
    whisper_size: str,
    device: str,
    sr: int = 48000,
    min_gap_ms: int = 300,
    speed_pct: int = 100,
    edited_text: str | None = None,
    phrases_cache: list[tuple[float, float, str]] | None = None,
    use_markers: bool = True,
    read_numbers: bool = False,
    spell_latin: bool = False,
    music_path: str | None = None,
    music_db: float = -18.0,
    duck_ratio: float = 8.0,
    duck_thresh: float = 0.05,
    tts_engine: str | None = None,
    language: str = "ru",
    yandex_key: str | None = None,
    yandex_voice: str | None = None,
    speed_jitter: float = 0.03,
    allow_beep_fallback: bool = False,
    auto_download_models: bool = True,
    tts_rate: float | None = None,
    tts_pitch: float | None = None,
    tts_style: str | None = None,
    tts_preset: str | None = None,
    *,
    silence_gap_ms: int = 0,
    autosave_minutes: float | None = None,
    force_offload: bool = False,
) -> tuple[str, str | None]:
    """Main revoicing function: transcribes, synthesizes speech, and mixes."""
    logger.info("Starting revoice_video for %s", video)
    ffmpeg = ensure_ffmpeg()
    safe_engine = re.sub(r"[^\w.-]", "", tts_engine or "unknown")
    safe_voice = re.sub(r"[^\w.-]", "", speaker)
    in_video = Path(video).resolve()
    out_dirp = Path(outdir).resolve()
    out_dirp.mkdir(parents=True, exist_ok=True)
    if not in_video.exists():
        raise FileNotFoundError(in_video)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        wav = tmp / "orig.wav"
        logger.debug("Extracting audio from %s", in_video)
        try:
            run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(in_video),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(sr),
                    "-acodec",
                    "pcm_s16le",
                    str(wav),
                ]
            )
        except Exception:
            logger.exception("Failed to extract audio")
            raise

        try:
            phrases = _setup(
                wav,
                whisper_size,
                device,
                phrases_cache,
                edited_text,
                use_markers=use_markers,
            )
        except ValueError as e:
            logger.exception("_setup reported invalid edited_text")
            raise ValueError(f"Invalid edited_text: {e}") from e
        except Exception:
            logger.exception("_setup failed")
            raise

        logger.info("Synthesizing voice track")
        try:
            voice_wav, fb_reason = synth_natural(
                ffmpeg,
                phrases,
                sr,
                speaker,
                tmp,
                tts_engine,
                language,
                yandex_key=yandex_key,
                yandex_voice=yandex_voice,
                min_gap_sec=max(0, min_gap_ms) / 1000.0,
                overall_speed=np.clip(speed_pct / 100.0, 0.8, 1.2),
                read_numbers=read_numbers,
                spell_latin=spell_latin,
                speed_jitter=speed_jitter,
                allow_beep_fallback=allow_beep_fallback,
                auto_download_models=auto_download_models,
                tts_rate=tts_rate,
                tts_pitch=tts_pitch,
                tts_style=tts_style,
                tts_preset=tts_preset,
                silence_gap_ms=silence_gap_ms,
                autosave_minutes=autosave_minutes,
                checkpoint_path=out_dirp / "tts_autosave.npz",
                force_offload=force_offload,
            )
        except Exception:
            logger.exception("synth_natural failed in revoice_video")
            raise

        # Replace audio if no background music is provided
        if not music_path or not Path(music_path).exists():
            logger.info("Muxing video without background music")
            out_video = out_dirp / f"{in_video.stem}_{safe_engine}_{safe_voice}.mp4"
            try:
                run(
                    [
                        ffmpeg,
                        "-y",
                        "-i",
                        str(in_video),
                        "-i",
                        str(voice_wav),
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:v",
                        "copy",
                        "-shortest",
                        str(out_video),
                    ]
                )
            except Exception:
                logger.exception("Video muxing failed")
                raise
            logger.info("revoice_video finished: %s", out_video)
            return str(out_video), fb_reason

        # Mixing voice with background music
        logger.info("Mixing voice with background music")
        out_audio = tmp / "mix.wav"
        # fmt: off
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(voice_wav),
            "-stream_loop",
            "-1",
            "-i",
            str(music_path),
            "-filter_complex",
            f"[1:a]volume={music_db}dB[bg];",
            f"[bg][0:a]sidechaincompress=threshold={duck_thresh}:ratio={duck_ratio}:attack=20:release=300[mduck];",
            "[mduck][0:a]amix=inputs=2:duration=first:dropout_transition=200,volume=1.0[out]",
            "-map", "[out]",
            "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", str(out_audio)
        ]
        # fmt: on
        try:
            run(cmd)
        except Exception:
            logger.exception("Audio mixing failed")
            raise

        out_video = out_dirp / f"{in_video.stem}_{safe_engine}_{safe_voice}.mp4"
        try:
            run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(in_video),
                    "-i",
                    str(out_audio),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-shortest",
                    str(out_video),
                ]
            )
        except Exception:
            logger.exception("Final muxing failed")
            raise
        logger.info("revoice_video finished: %s", out_video)
        return str(out_video), fb_reason
