# Roadmap

## EN

### Borealis ASR integration — 📝 planned
- Model confirmation
- Offline deployment (CUDA, transformers with `trust_remote_code=True`)
- API adapter (`POST /v1/audio/transcriptions`, `POST /v1/audio/chunked`)
- Generation parameters (`max_new_tokens≈350`, `top_p=0.9`, `top_k=50`, `temperature=0.2`)
- Provider registration in config
- Post-processing
- Quality expectations
- Realtime and batch support
- Logging and checkpoint/feature_extractor version pinning

## RU

### Интеграция Borealis ASR — 📝 planned
- Подтверждение модели
- Офлайн-размещение (CUDA, transformers с `trust_remote_code=True`)
- API-адаптер (`POST /v1/audio/transcriptions`, `POST /v1/audio/chunked`)
- Параметры генерации (`max_new_tokens≈350`, `top_p=0.9`, `top_k=50`, `temperature=0.2`)
- Регистрация провайдера в конфиге
- Постпроцессинг
- Ожидания по качеству
- Поддержка realtime/batch
- Логирование и фиксация версий чекпоинта/feature_extractor

## Sprints

- Sprint 1 — Packaging & CI hardening — 📝 planned
- Sprint 2 — TTS pool finalization — 📝 planned
- Sprint 3 — Borealis ASR — 📝 planned
- Sprint 4 — LLM connectors & live mode — 📝 planned
