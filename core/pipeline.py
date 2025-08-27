from dataclasses import dataclass
from typing import List
from pathlib import Path

@dataclass
class Segment:
    start: float
    end: float
    original_text: str
    edited_text: str

def extract_audio(input_video: Path, output_wav: Path, ffmpeg_path: Path) -> None:
    """ffmpeg -i input -vn -ac 1 -ar 16000 -y output.wav"""
    # TODO: implement subprocess call
    pass

def run_stt(audio_wav: Path) -> List[Segment]:
    """Run faster-whisper and return segments"""
    # TODO: call faster_whisper
    return []

def synthesize_tts(segments: List[Segment], engine: str, models_dir: Path, out_dir: Path) -> Path:
    """Join synthesized segments into a WAV track"""
    # TODO: implement via selected TTS engine adapter
    return out_dir / "voice.wav"

def mux_video(original_video: Path, voice_wav: Path, music_wav: Path, out_video: Path, ffmpeg_path: Path) -> None:
    """Sidechain ducking and muxing via ffmpeg"""
    # TODO: implement ducking and mux
    pass
