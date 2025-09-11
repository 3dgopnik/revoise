# Revoice — TTS/STT Pipeline и UI

> Мини-пайплайн: извлечь аудио → faster-whisper → правка → TTS → ffmpeg-mux.
> Цель — «установил и запустил»: UI на PySide6, офлайн TTS приоритет, Live-режим опционально.

## 📌 Оглавление
1. [Usage](#usage)
2. [Сборка окружения (uv)](#сборка-окружения-uv)
3. [Новая оболочка UI (Qt / PySide6)](#новая-оболочка-ui-qt--pyside6)
4. [Редактор текста и таймингов](#редактор-текста-и-таймингов)
5. [TTS-движки (офлайн/онлайн)](#tts-движки-офлайнонлайн)
6. [Звук: музыка, SFX, постпроцесс](#звук-музыка-sfx-постпроцесс)
7. [Лёгкий видеоредактор (таймлайн)](#лёгкий-видеоредактор-таймлайн)
8. [Live режим (реалтайм ассистент)](#live-режим-реалтайм-ассистент)
9. [«Было → стало» и сравнение](#было--стало-и-сравнение)
10. [Пресеты диктора и качества](#пресеты-диктора-и-качества)
11. [Дорожная карта по спринтам](#дорожная-карта-по-спринтам)
12. [Риски и контроль качества](#риски-и-контроль-качества)

---
 
## Usage

Для запуска UI используйте один из скриптов:

```bash
./run_ui.sh   # Linux/macOS
run_ui.bat    # Windows
```

Оба выполняют `uv run python -m ui.main`.

## Сборка окружения (uv)
- Требуется Python 3.10–3.12 и [uv](https://docs.astral.sh/uv/).
- Установка зависимостей:
```bash
uv pip install -r requirements.txt
# или
uv pip install .
```
- Запуск UI напрямую:
```bash
uv run python -m ui.main
```
(см. раздел [Usage](#usage) для скриптов).
- Каталог `models/` может быть пустым; ссылки на веса перечислены в `models/model_registry.json`.
- Пропущенные Python-зависимости устанавливаются автоматически в `.venv`,
  а недостающие модели скачиваются в `models/` при первом использовании.
- Проверки качества:
  Перед запуском установите dev-зависимости (ruff, mypy, pytest):
  ```bash
  uv pip install --extra dev .
  uv run ruff check .
  uv run ruff format --check .
  uv run mypy .
  uv run pytest -q
  ```
- Обновление списка зависимостей (после ленивой установки):
```bash
uv run python tools/freeze_reqs.py
```

### Config

Копируйте `config.example.json` в `config.json` и при необходимости правьте.
Доступные ключи:

- `llm.family` — семейство локальной LLM.
- `llm.model_path` — путь к файлу модели, если она уже скачана.
- `llm.auto_download` — автоматически загружать недостающую LLM.
- `tts.default_engine` — движок TTS по умолчанию.
- `tts.<engine>.model` — имя модели TTS.
- `tts.<engine>.device` — устройство (`cpu` или `cuda`).
- `tts.<engine>.attention_backend` — бэкенд внимания (`sdpa`, `flash` и т.д.).
- `tts.<engine>.quantization` — режим квантования.
- `tts.<engine>.voices` — список доступных пресетов голосов.
- `tts.silence_gap_ms` — вставка паузы между фразами (мс).
- `tts.autosave_minutes` — как часто сохранять чекпоинты синтеза.
- `tts.force_offload` — освобождать VRAM по завершении и логировать пик.
- `preferences.pin_dependencies` — предлагать фиксировать версии в `requirements.txt`.
- `use_imageio_ffmpeg` — использовать пакет `imageio-ffmpeg` для автоматической установки FFmpeg.
- `externals.ffmpeg` — путь к бинарю FFmpeg, если хотите использовать свой экземпляр.

---

## Новая оболочка UI (Qt / PySide6)
- Главное окно: входное видео, выбор TTS, музыка/SFX, рендер
- Редактор текста: таблица сегментов (start/end/original/edited) + diff-подсветка
- Аудиомиксер: голос, фон, SFX
- Хоткеи: Ctrl+O/S/E, Ctrl+Z/Y, Ctrl+C/X/V, Ctrl+A, Ctrl+Enter, Space
- Контекстное меню ПКМ

---

## Редактор текста и таймингов
- Импорт/экспорт JSON/CSV/SRT, вставка из буфера, сброс правок
- Ритм-якоря, автопунктуация, VAD-паузы (350–450 ms), `speed_jitter`
- Парсер сценариев проверяет строки вида `Speaker N:` (до 4 дикторов) и разбивает текст на чанки 30–120 с
- AI правка текста: кнопка **AI Edit** с выбором редактора Qwen или VibeVoice

---

## TTS-движки (офлайн/онлайн)
**Офлайн (приоритет):** Silero, Coqui XTTS-v2, Dia, MARS5, Orpheus
**Онлайн/гибрид:** Hume, gTTS, Gemini, OpenAI TTS, Minimax, Chatterbox, VibeVoice
Сравнение: https://huggingface.co/spaces/TTS-AGI/TTS-Arena

> Референсы спикеров для Coqui XTTS кладите в `models/speakers/<имя>`.

### Silero requirements
Silero TTS requires `torch` and `torchaudio`. Install them to avoid a BeepTTS fallback:
```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```
If these packages are missing, the pipeline falls back to BeepTTS with an audible beep.

If `models/torch_hub/snakers4_silero-models_master` exists, Silero loads from the cache and sets `TORCH_HUB_DISABLE_AUTOFETCH=1` to prevent network access.

### Manual TTS model fetch
Download models for offline use:
```bash
# Silero voice pack
uv run python tools/fetch_tts_models.py silero --language en

# VibeVoice weights
uv run python tools/fetch_tts_models.py vibevoice --model 1.5b

# Models from registry (e.g. Coqui XTTS)
uv run python tools/fetch_tts_models.py registry --engine coqui_xtts
```
Downloads use local cache and show progress bars. If a fetch fails with SSL
errors, ensure your system certificates are installed or set `SSL_CERT_FILE` to a
valid bundle.

### TTS dependencies
Missing Python packages are installed automatically into `.venv` for TTS engines:

- Silero: `torch` and `torchaudio` (`uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`)
- Coqui XTTS: `TTS`
- gTTS: `gTTS`

Если установка не удалась, движок пропускается. Чтобы включить его позднее,
установите пакет вручную, например: `uv pip install TTS`.


- gTTS: выберите движок `gtts` в UI. Сервис не предлагает голоса и требует интернет.
- VibeVoice: выберите движок `vibevoice` в UI. Требует установленного бинаря `vibe-voice` (см. [docs/tts_vibevoice.md](docs/tts_vibevoice.md)).

### TTS CLI
Переменные окружения:
- `TTS_ENGINE` — движок (по умолчанию `silero`; доступны `vibevoice`, `beep`)
- `SILERO_SPEAKER` — имя диктора

```bash
uv run python -m ui.main --say "text"                     # default engine
TTS_ENGINE=silero   uv run python -m ui.main --say "Hello"
TTS_ENGINE=vibevoice uv run python -m ui.main --say "Hello"
# output: output/tts_test.wav
```

---

## Звук: музыка, SFX, постпроцесс
- Фон: −18 dB, sidechain ducking (без `makeup=0`), «поднимать на паузах»
- Голос: лёгкий компрессор, де-клик, нормализация
- SFX: ручной слой или генерация (ElevenLabs SFX)

---

## Лёгкий видеоредактор (таймлайн)
- Обрезка/трим/скорость, логотип/заставка
- Дорожки: Video • Voice • Music • SFX, зум таймлайна, маркеры
- Экспорт: H.264/H.265, SRT/впечённые субтитры

---

## Live режим (реалтайм ассистент)
- Интеграция: https://github.com/Mozer/talk-llama-fast
- STT: whisper.cpp, LLM: Mistral 7B (Q5_0.gguf), TTS: XTTSv2 streaming

---

## «Было → стало» и сравнение
- Два плеера (оригинал/новая озвучка) + таблица параметров
- Экспорт отчёта (HTML/CSV) со списком изменённых сегментов и таймингов

---

## Пресеты диктора и качества
- По умолчанию: speed 0.98–1.02, pause 350–450 ms, `speed_jitter=0.03–0.05`
- Стили: Нейтральный, Объясняющий, Рекламный, Дружелюбный
- Кнопка «Сохранить мой пресет» → `.rvpreset`
- `.rvpreset` из папки `presets/` загружаются при старте и дают быстрый выбор rate/pitch/style/preset

---

## Дорожная карта по спринтам
- Спринт A — Qt-MVP (UI, редактор с diff, ducking, .rvproj)
- Спринт B — Ритм/стабильность (якоря, постпроцесс, экспорт SRT/CSV)
- Спринт C — TTS-пул (Dia/MARS5), предпрослушка
- Спринт D — Таймлайн (NLE-лайт)
- Спринт E — Live режим (talk-llama-fast)
- Спринт F — Облако (Hume, Gemini, OpenAI TTS), SFX-генерация

---

## Риски и контроль качества
- VRAM/веса: проверять видеопамять, предлагать лёгкие модели
- RU дикция: не использовать EN/ZH zero-shot для RU по умолчанию
- Лицензии/контент: предупреждать при клонировании голосов
- Логи и профилирование производительности

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest -q
```

---

## Быстрый старт
```bash
# 1) Установить uv и Python 3.10–3.12
# 2) Установить зависимости
uv pip install -r requirements.txt
# 3) Скачать модель TTS (пример: VibeVoice 1.5B)
uv run python tools/fetch_tts_models.py vibevoice --model 1.5b
# 4) Запустить UI
uv run python -m ui.main
# 5) Переключить движок из CLI
TTS_ENGINE=silero   uv run python -m ui.main --say "Hello"
TTS_ENGINE=vibevoice uv run python -m ui.main --say "Hello"
# 6) (Опционально) imageio-ffmpeg или свой ffmpeg в ./bin либо PATH
```

## Лицензия
MIT
