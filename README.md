# RevoicePortable — TTS/STT Pipeline и UI

> Мини-портативный пайплайн: извлечь аудио → faster-whisper → правка → TTS → ffmpeg-mux.  
> Цель — «распаковал и запустил»: UI на PySide6, офлайн TTS приоритет, Live-режим опционально.

## 📌 Оглавление
1. [Архитектура и папка portable](#архитектура-и-папка-portable)  
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

## Архитектура и папка portable
```
/RevoicePortable/
  /bin/ffmpeg.exe
  /models/
    /whisper/
    /tts/ (silero, coqui_xtts, dia, mars5, vibevoice, chatterbox)
    /lipsync/
  /sfx/
  /input/
  /output/
  /ui/
  /core/
  /projects/
  /logs/
  uv.lock
  pyproject.toml
```
 Папка `models/` может быть пустой; см. `models/README.md` для загрузки весов.
 Реестр официальных ссылок находится в `models/model_registry.json`.
 Добавляйте новые модели или зеркала, дописывая массив URL-ов в нужную категорию (`tts`, `stt`, `llm`).
Видео для обработки кладите в папку `input`, готовые ролики появляются в `output` (например, `D:\revoise\input` и `D:\revoise\output`). Файлы называются `<имя>_<движок>_<голос>.mp4`.

---

## Сборка окружения (uv)
- Требуется Python 3.10–3.12 и [uv](https://docs.astral.sh/uv/).
- Установка зависимостей и запуск UI:
```bash
uv sync --all-extras --frozen
uv run python -m ui.main
```
- Проверки качества:
```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest -q
```

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

### TTS dependencies
Missing Python packages are installed automatically into `.venv` for TTS engines:

- Silero: `torch` and `torchaudio` (`uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`)
- Coqui XTTS: `TTS`
- gTTS: `gTTS`

Если установка не удалась, движок пропускается. Чтобы включить его позднее,
установите пакет вручную, например: `uv pip install TTS`.


- gTTS: выберите движок `gtts` в UI. Сервис не предлагает голоса и требует интернет.

### TTS CLI
Переменные окружения:
- `TTS_ENGINE` — движок (по умолчанию `silero`)
- `SILERO_SPEAKER` — имя диктора

```bash
uv run python -m ui.main --say "text"
# появится output/tts_test.wav
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
uv sync --all-extras --frozen
# 3) Запустить UI
uv run python -m ui.main
# 4) (Опционально) положить ffmpeg в ./bin или поставить в PATH
```

## Лицензия
MIT
