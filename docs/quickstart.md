# Быстрый старт

1. Установите Python 3.10–3.12 и [uv](https://docs.astral.sh/uv/).
2. Установите зависимости:
   ```bash
   uv pip install -r requirements.txt
   ```
3. Запустите пользовательский интерфейс:
   ```bash
   ./run_ui.sh      # Linux/macOS
   run_ui.bat       # Windows
   ```
   Скрипты выполняют `uv run python -m ui.main`.
4. Для тестовой озвучки используйте CLI:
   ```bash
   uv run python -m ui.main --say "Привет"
   ```
   Результат сохранится в `output/tts_test.wav`.
5. (Опционально) заранее скачайте модели распознавания речи:
   ```bash
   uv run python tools/fetch_stt_models.py base small
   ```
   На Windows воспользуйтесь `tools\fetch_stt_models.bat base small`.
