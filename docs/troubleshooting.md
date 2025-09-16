# Устранение неполадок

## Автоустановка зависимостей
При запуске UI выполняется проверка `preflight()`, которая устанавливает `faster-whisper`
и зависимости движка Silero (`torch`, `omegaconf`). При успешной установке в логах
появляется запись с пометкой об установке; при ошибке приложение покажет окно с
подробностями и завершится, чтобы избежать некорректного состояния.

## Silero: отсутствует torch
Установите `torch` и `torchaudio`:
```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Иначе движок заменится на BeepTTS.

## Ошибки SSL или прокси
Отключите опцию «Проверять SSL-сертификаты при загрузке моделей» в настройках UI
и повторите попытку. Если используете CLI или предпочтителен ручной запуск,
можно временно задать переменные окружения:
```bash
export HTTPS_PROXY=http://proxy:port
export NO_SSL_VERIFY=1
```
Disable the "Verify SSL certificates" checkbox in the Settings dialog when a
corporate proxy replaces certificates; for CLI workflows fall back to the
environment variables above.

## Qt: libGL.so.1 not found
Установите системную библиотеку `libgl1` (или аналог) перед запуском UI.
