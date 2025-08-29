# -*- coding: utf-8 -*-
import logging
import re
import shutil
import subprocess
import tempfile
from itertools import zip_longest
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import soundfile as sf
from num2words import num2words
from tqdm import tqdm

from .model_manager import ensure_model
from .tts_adapters import BeepTTS, CoquiXTTS, SileroTTS, YandexTTS, GTTSTTS

logger = logging.getLogger(__name__)

# ===================== Global =====================
MODEL_PATH_CACHE: dict[tuple[str, str], Path] = {}
FWHISPER: Any | None = None
TTS_MODEL = None

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
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    local = Path(__file__).resolve().parent.parent / "bin" / "ffmpeg.exe"
    if local.exists():
        return str(local)
    raise RuntimeError("ffmpeg не найден. Положите ffmpeg.exe в bin/ или добавьте в PATH.")


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
        from faster_whisper import WhisperModel

        need_load = (FWHISPER is None) or getattr(FWHISPER, "_name", "") != model_size
        if need_load:
            logger.info("Ensuring Whisper model %s is available", model_size)
            try:
                cache_key = (model_size, "stt")
                model_path = MODEL_PATH_CACHE.get(cache_key)
                if model_path is None:
                    model_path = ensure_model(model_size, "stt", parent=parent)
                    MODEL_PATH_CACHE[cache_key] = model_path
            except FileNotFoundError as exc:
                logger.error("Model download declined: %s", exc)
                raise RuntimeError("Whisper model download was declined") from exc
            logger.info("Loading Whisper model %s from %s on %s", model_size, model_path, device)
            compute_type = "int8_float16" if device == "cuda" else "int8"
            FWHISPER = WhisperModel(str(model_path), device=device, compute_type=compute_type)
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


def synth_chunk(
    ffmpeg: str,
    text: str,
    sr: int,
    speaker: str,
    tmpdir: Path,
    tts_engine: str,
    read_numbers: bool = False,
    spell_latin: bool = False,
    yandex_key: Optional[str] = None,
    yandex_voice: Optional[str] = None,
) -> np.ndarray:
    """Generate an audio fragment for a single phrase."""

    text = normalize_text(text, read_numbers=read_numbers, spell_latin=spell_latin)
    engine = (tts_engine or "beep").lower()
    logger.debug("Synthesizing chunk with engine=%s", engine)
    try:
        if engine == "beep":
            wav = BeepTTS().tts(text, speaker, sr=sr)
            model_sr = sr
        elif engine == "silero":
            wav = SileroTTS(Path(__file__).resolve().parent.parent).tts(text, speaker, sr=sr)
            model_sr = sr
        elif engine == "coqui_xtts":
            wav = CoquiXTTS(Path(__file__).resolve().parent.parent).tts(text, speaker, sr=24000)
            model_sr = 24000
        elif engine == "gtts":
            wav = GTTSTTS().tts(text, speaker, sr=sr)
            model_sr = sr
        elif engine == "yandex":
            if not yandex_key or not (yandex_voice or speaker):
                raise ValueError("Yandex TTS requires yandex_key and yandex_voice")
            voice = yandex_voice or speaker
            wav = YandexTTS().tts(text, voice, sr=sr, key=yandex_key)
            model_sr = sr
        else:
            raise ValueError(f"Unsupported TTS engine: {engine}")

        raw = tmpdir / "tts_raw.wav"
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
        logger.debug("synth_chunk produced %d samples", len(wav_out))
        return wav_out
    except Exception:
        logger.exception("synth_chunk failed")
        raise


def synth_natural(
    ffmpeg: str,
    phrases: List[Tuple[float, float, str]],
    sr: int,
    speaker: str,
    tmpdir: Path,
    tts_engine: str,
    yandex_key: Optional[str] = None,
    yandex_voice: Optional[str] = None,
    min_gap_sec: float = 0.30,
    overall_speed: float = 1.0,
    read_numbers: bool = False,
    spell_latin: bool = False,
    speed_jitter: float = 0.03,
) -> Path:
    """
    Simple synthesis: calls synth_chunk for each phrase.
    """
    logger.info("Starting synth_natural for %d phrases", len(phrases))
    total_dur = phrases[-1][1] + 3.0
    master = np.zeros(int(total_dur * sr), dtype=np.float32)
    cur_tail = 0.0
    try:
        for i, (start, end, txt) in enumerate(tqdm(phrases, desc="TTS", unit="phr"), start=1):
            logger.debug("Synthesizing phrase %d", i)
            try:
                wav = synth_chunk(
                    ffmpeg,
                    txt,
                    sr,
                    speaker,
                    tmpdir,
                    tts_engine,
                    read_numbers=read_numbers,
                    spell_latin=spell_latin,
                    yandex_key=yandex_key,
                    yandex_voice=yandex_voice,
                )
            except Exception:
                logger.exception("synth_chunk failed for phrase %d", i)
                raise
            place_t = max(start, cur_tail + min_gap_sec)
            s0 = int(place_t * sr)
            s1 = min(len(master), s0 + len(wav))
            if s0 < len(master):
                master[s0:s1] += wav[: (s1 - s0)]
            cur_tail = (s0 + len(wav)) / sr
        out_wav = tmpdir / "voice_aligned.wav"
        sf.write(out_wav, master, sr)
        logger.info("Finished synth_natural: %s", out_wav)
        return out_wav
    except Exception:
        logger.exception("synth_natural failed")
        raise


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
    edited_text: Optional[str] = None,
    phrases_cache: Optional[List[Tuple[float, float, str]]] = None,
    use_markers: bool = True,
    read_numbers: bool = False,
    spell_latin: bool = False,
    music_path: Optional[str] = None,
    music_db: float = -18.0,
    duck_ratio: float = 8.0,
    duck_thresh: float = 0.05,
    tts_engine: str = "silero",
    yandex_key: Optional[str] = None,
    yandex_voice: Optional[str] = None,
    speed_jitter: float = 0.03,
) -> str:
    """Main revoicing function: transcribes, synthesizes speech, and mixes."""
    logger.info("Starting revoice_video for %s", video)
    ffmpeg = ensure_ffmpeg()
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
            voice_wav = synth_natural(
                ffmpeg,
                phrases,
                sr,
                speaker,
                tmp,
                tts_engine,
                yandex_key=yandex_key,
                yandex_voice=yandex_voice,
                min_gap_sec=max(0, min_gap_ms) / 1000.0,
                overall_speed=np.clip(speed_pct / 100.0, 0.8, 1.2),
                read_numbers=read_numbers,
                spell_latin=spell_latin,
                speed_jitter=speed_jitter,
            )
        except Exception:
            logger.exception("synth_natural failed in revoice_video")
            raise

        # Replace audio if no background music is provided
        if not music_path or not Path(music_path).exists():
            logger.info("Muxing video without background music")
            out_video = out_dirp / f"{in_video.stem}_revoiced.mp4"
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
            return str(out_video)

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

        out_video = out_dirp / f"{in_video.stem}_revoiced.mp4"
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
        return str(out_video)
