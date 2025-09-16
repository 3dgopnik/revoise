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

## Настройки UI (`config.json`)

Интерфейс `ui.main` использует корневой `config.json` для хранения API‑ключей
и пользовательских предпочтений. При наличии модуля `cryptography` файл
`config.key` генерируется автоматически, а значения `yandex_key` и
`chatgpt_key` шифруются; иначе ключи сохраняются в base64. Если `config.json`
отсутствует, UI создаёт его с настройками по умолчанию.

Поддерживаемые поля:

- `yandex_key` — API‑ключ Яндекс TTS (строка, пустая по умолчанию).
- `chatgpt_key` — ключ OpenAI ChatGPT, используемый в редакторе текста (строка).
- `allow_beep_fallback` — воспроизводить короткий сигнал при отсутствии озвучки
  (логическое значение, `false` по умолчанию).
- `auto_download_models` — разрешить автоматическую загрузку моделей и ресурсов
  (логическое значение, `true` по умолчанию).
- `out_dir` — абсолютный путь к папке для сохранения результатов (`output`
  внутри репозитория по умолчанию).
- `language` — язык интерфейса и подсказок, код ISO (`ru` по умолчанию).
- `preset` — выбранный пресет настроек озвучки; `"None"` сохраняет ручной выбор.
- `whisper_model` — название модели распознавания речи (`"base"` по умолчанию).
- `speed_pct` — темп синтеза в процентах (целое число, `100` по умолчанию).
- `min_gap_ms` — минимальная пауза между репликами в миллисекундах (`350`).
- `read_numbers` — произносить числа полностью вместо цифр (`false`).
- `spell_latin` — произносить латиницу посимвольно (`false`).

Пример файла с настройками по умолчанию:

```json
{
  "yandex_key": "",
  "chatgpt_key": "",
  "allow_beep_fallback": false,
  "auto_download_models": true,
  "out_dir": "/абсолютный/путь/к/output",
  "language": "ru",
  "preset": "None",
  "whisper_model": "base",
  "speed_pct": 100,
  "min_gap_ms": 350,
  "read_numbers": false,
  "spell_latin": false
}
```
