# Модели и веса

Папка может быть пустой. Положите веса сюда (или укажите пути в `config.example.json`).

## Whisper / faster-whisper
- Поместите модели в `models/whisper` (ggml/gguf для whisper.cpp или файлы для faster-whisper).

## TTS
- Silero, Coqui XTTS-v2, Dia, Kokoro, MARS5, Orpheus — каждая в своей подпапке (`models/tts/<engine>`).
- Некоторые модели требуют GPU и значительную VRAM.

## LipSync
- Позже: wav2lip в `models/lipsync`.
