# Устранение неполадок

## Silero: отсутствует torch
Установите `torch` и `torchaudio`:
```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Иначе движок заменится на BeepTTS.

## Ошибки SSL или прокси
Перед загрузкой моделей задайте:
```bash
export HTTPS_PROXY=http://proxy:port
export NO_SSL_VERIFY=1
```

## Qt: libGL.so.1 not found
Установите системную библиотеку `libgl1` (или аналог) перед запуском UI.
