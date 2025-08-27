# -*- coding: utf-8 -*-
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
from itertools import zip_longest

import numpy as np
import soundfile as sf
from tqdm import tqdm

# ===================== Глобальные =====================
FWHISPER = None
TTS_MODEL = None

MULTISPACE = re.compile(r"\s+")
PAUSE_TAG = re.compile(r"\[\[\s*PAUSE\s*=\s*(\d+)\s*\]\]", re.IGNORECASE)

# ===================== Утилиты =====================
def run(cmd: List[str]):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    return r

def ensure_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe: 
        return exe
    local = Path(__file__).resolve().parent.parent / "bin" / "ffmpeg.exe"
    if local.exists():
        return str(local)
    raise RuntimeError("ffmpeg не найден. Положите ffmpeg.exe в bin/ или добавьте в PATH.")

# ===================== Whisper =====================
def transcribe_whisper(audio_wav: Path, language="ru", model_size="large-v3", device="cuda"):
    global FWHISPER
    from faster_whisper import WhisperModel
    need_load = (FWHISPER is None) or getattr(FWHISPER, "_name", "") != model_size
    if need_load:
        compute_type = "int8_float16" if device == "cuda" else "int8"
        FWHISPER = WhisperModel(model_size, device=device, compute_type=compute_type)
        FWHISPER._name = model_size
    segments, _ = FWHISPER.transcribe(
        str(audio_wav), language=language, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300}
    )
    return [(s.start, s.end, s.text.strip()) for s in segments]

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

    phrases = []
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

def phrases_to_marked_text(phrases: List[Tuple[float,float,str]]) -> str:
    lines=[]
    for i,(_,_,t) in enumerate(phrases, start=1):
        lines.append(f"[[#{i}]] {t}")
    return "\n".join(lines)


def apply_edited_text(phrases: List[Tuple[float, float, str]], edited_text: str,
                      *, use_markers: bool = True) -> List[Tuple[float, float, str]]:
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


def _setup(wav: Path, whisper_size: str, device: str,
           phrases_cache: Optional[List[Tuple[float, float, str]]],
           edited_text: Optional[str], *, use_markers: bool) -> List[Tuple[float, float, str]]:
    """Prepare phrases: transcribe if needed and apply edited text."""

    if phrases_cache is None:
        segs = transcribe_whisper(wav, language="ru", model_size=whisper_size, device=device)
        if not segs:
            raise RuntimeError("Речь не обнаружена.")
        phrases = merge_into_phrases(segs, max_gap=0.35, min_len=0.8)
    else:
        phrases = phrases_cache

    if edited_text:
        phrases = apply_edited_text(phrases, edited_text, use_markers=use_markers)

    return phrases

# ===================== TTS-заглушки =====================
# Здесь остаются твои текущие реализации synth_natural и synth_chunk.
# Если используется Silero/Yandex/XTTS — они будут вызывать normalize_text с read_numbers/spell_latin.

def synth_chunk(ffmpeg: str, text: str, sr: int, speaker: str,
                tmpdir: Path, tts_engine: str,
                read_numbers: bool = False, spell_latin: bool = False) -> np.ndarray:
    """
    Заглушка TTS — тут должна быть твоя реальная логика Silero/XTTS/Kokoro.
    Для краткости пример: генерируем тишину нужной длительности.
    """
    dur = max(0.3, len(text.split()) * 0.3)
    return np.zeros(int(sr * dur), dtype=np.float32)

def synth_natural(ffmpeg: str, phrases: List[Tuple[float,float,str]], sr: int,
                  speaker: str, tmpdir: Path, tts_engine: str,
                  min_gap_sec: float = 0.30, overall_speed: float = 1.0,
                  read_numbers: bool = False, spell_latin: bool = False,
                  speed_jitter: float = 0.03) -> Path:
    """
    Простейший синтез: вызывает synth_chunk на каждую фразу.
    """
    total_dur = phrases[-1][1] + 3.0
    master = np.zeros(int(total_dur * sr), dtype=np.float32)
    cur_tail = 0.0
    for i, (start, end, txt) in enumerate(tqdm(phrases, desc="TTS", unit="phr")):
        wav = synth_chunk(ffmpeg, txt, sr, speaker, tmpdir, tts_engine,
                          read_numbers=read_numbers, spell_latin=spell_latin)
        place_t = max(start, cur_tail + min_gap_sec)
        s0 = int(place_t * sr)
        s1 = min(len(master), s0 + len(wav))
        if s0 < len(master):
            master[s0:s1] += wav[:(s1 - s0)]
        cur_tail = (s0 + len(wav)) / sr
    out_wav = tmpdir / "voice_aligned.wav"
    sf.write(out_wav, master, sr)
    return out_wav

# ===================== Основной пайплайн =====================
def revoice_video(video: str, outdir: str, speaker: str, whisper_size: str, device: str,
                  sr: int = 48000, min_gap_ms: int = 300,
                  speed_pct: int = 100, edited_text: Optional[str] = None,
                  phrases_cache: Optional[List[Tuple[float,float,str]]] = None,
                  use_markers: bool = True,
                  read_numbers: bool = False, spell_latin: bool = False,
                  music_path: Optional[str] = None, music_db: float = -18.0,
                  duck_ratio: float = 8.0, duck_thresh: float = 0.05,
                  tts_engine: str = "silero",
                  yandex_key: Optional[str] = None, yandex_voice: Optional[str] = None,
                  speed_jitter: float = 0.03) -> str:
    """
    Основная функция переозвучки: распознает, генерирует речь, микширует.
    """
    ffmpeg = ensure_ffmpeg()
    in_video = Path(video).resolve()
    out_dirp = Path(outdir).resolve()
    out_dirp.mkdir(parents=True, exist_ok=True)
    if not in_video.exists():
        raise FileNotFoundError(in_video)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        wav = tmp / "orig.wav"
        run([ffmpeg, "-y", "-i", str(in_video), "-vn", "-ac", "1", "-ar", str(sr),
             "-acodec", "pcm_s16le", str(wav)])

        try:
            phrases = _setup(wav, whisper_size, device, phrases_cache, edited_text,
                             use_markers=use_markers)
        except ValueError as e:
            raise ValueError(f"Invalid edited_text: {e}") from e

        voice_wav = synth_natural(
            ffmpeg, phrases, sr, speaker, tmp, tts_engine,
            min_gap_sec=max(0, min_gap_ms)/1000.0,
            overall_speed=np.clip(speed_pct/100.0, 0.8, 1.2),
            read_numbers=read_numbers, spell_latin=spell_latin,
            speed_jitter=speed_jitter
        )

        # Если нет музыки — просто подменяем звук
        if not music_path or not Path(music_path).exists():
            out_video = out_dirp / f"{in_video.stem}_revoiced.mp4"
            run([ffmpeg, "-y", "-i", str(in_video), "-i", str(voice_wav),
                 "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-shortest", str(out_video)])
            return str(out_video)

        # Микширование с музыкой
        out_audio = tmp / "mix.wav"
        cmd = [
            ffmpeg, "-y",
            "-i", str(voice_wav),
            "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex",
            f"[1:a]volume={music_db}dB[bg];"
            f"[bg][0:a]sidechaincompress=threshold={duck_thresh}:ratio={duck_ratio}:attack=20:release=300[mduck];"
            f"[mduck][0:a]amix=inputs=2:duration=first:dropout_transition=200,volume=1.0[out]",
            "-map", "[out]",
            "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", str(out_audio)
        ]
        run(cmd)

        out_video = out_dirp / f"{in_video.stem}_revoiced.mp4"
        run([ffmpeg, "-y", "-i", str(in_video), "-i", str(out_audio),
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-shortest", str(out_video)])
        return str(out_video)
