import shutil

import numpy as np
import pytest

from core.script_parser import parse_script, split_chunks
from core.tts.engines.vibevoice import VibeVoiceEngine

try:
    import torch
except Exception:  # pragma: no cover - torch may be missing
    torch = None


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available() or shutil.which("vibe-voice") is None,
    reason="vibe-voice binary and CUDA GPU required",
)
def test_vibevoice_long_dialogue():
    before = torch.cuda.memory_allocated(0)
    lines = [f"Speaker {1 if i % 2 == 0 else 2}: слово{i}" for i in range(750)]
    script = "\n".join(lines)
    parsed = parse_script(script)
    chunks = split_chunks(parsed, min_sec=30, max_sec=60)
    engine = VibeVoiceEngine()
    engine.load()
    wavs = []
    for chunk in chunks:
        text = " ".join(line.text for line in chunk)
        wavs.append(engine.synthesize(text, speaker="0", sample_rate=48000))
    audio = np.concatenate(wavs)
    assert audio.size > 0
    engine.unload()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated(0)
    assert after <= before
