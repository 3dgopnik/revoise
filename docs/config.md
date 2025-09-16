# Конфигурация

1. Скопируйте `config.example.json` в `config.json`.
2. Основные ключи:
   - `llm.family` — семейство локальной LLM.
   - `llm.model_path` — путь к файлу модели.
   - `llm.auto_download` — загружать модель автоматически.
   - `tts.default_engine` — движок TTS по умолчанию.
   - `tts.<engine>.model` — имя модели TTS.
   - `tts.<engine>.device` — устройство (`cpu` или `cuda`).
   - `tts.<engine>.attention_backend` — бэкенд внимания (`sdpa`, `flash` и т.д.).
   - `tts.<engine>.quantization` — режим квантования.
   - `tts.<engine>.voices` — доступные пресеты голосов.
   - `tts.silence_gap_ms` — пауза между фразами (мс).
   - `tts.autosave_minutes` — автосохранение чекпоинтов.
   - `tts.force_offload` — выгружать модель и логировать пик VRAM.
   - `preferences.pin_dependencies` — предлагать фиксировать зависимости.
   - `use_imageio_ffmpeg` — автоматически установить `imageio-ffmpeg`.
   - `externals.ffmpeg` — путь к собственному бинарю FFmpeg.
