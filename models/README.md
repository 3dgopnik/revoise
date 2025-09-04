# Модели и веса

Папка может быть пустой. Положите веса сюда (или укажите пути в `config.example.json`).

## Реестр моделей

Официальные ссылки на веса перечислены в `model_registry.json`.
Файл содержит категории (`tts`, `stt`, `llm`) и для каждой
модели — массив URL-ов по приоритету (основной, зеркала).

```json
{
  "tts": {
    "coqui_xtts": [
      "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.zip"
    ]
  }
}
```

Дополнить реестр можно, добавив новую модель или зеркало в нужный раздел.

## Whisper / faster-whisper
- Поместите модели в `models/whisper` (ggml/gguf для whisper.cpp или файлы для faster-whisper).

## TTS
- Coqui XTTS-v2, Dia, MARS5, Orpheus — каждая в своей подпапке (`models/tts/<engine>`).
- Silero скачивается автоматически через `torch.hub`.
- Некоторые модели требуют GPU и значительную VRAM.
- Референсы спикеров для Coqui XTTS кладите в `models/speakers/<имя_спикера>/` (wav-файлы).

## LipSync
- Позже: wav2lip в `models/lipsync`.
