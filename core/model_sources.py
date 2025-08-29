from __future__ import annotations

"""Mapping of official model sources.

The mapping groups models by category (tts, stt, llm) and provides
URLs to download the model archives or binaries from official
HuggingFace or GitHub releases.
"""

MODEL_SOURCES: dict[str, dict[str, str]] = {
    "tts": {
        # Coqui XTTS v2 model archive on HuggingFace
        "coqui_xtts": "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.zip",
        # Silero Russian TTS model from GitHub releases
        "silero": "https://github.com/snakers4/silero-models/releases/download/v0.4/silero_tts_ru_v3.pt",
    },
    "stt": {
        # Faster-Whisper large-v3 model weights on HuggingFace
        "large-v3": "https://huggingface.co/guillaumekln/faster-whisper-large-v3/resolve/main/model.bin",
    },
    "llm": {
        # Example lightweight instruction-tuned model on HuggingFace
        "qwen2.5": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/Qwen2.5-0.5B-Instruct-q4_k_m.gguf",
    },
}
