# Development plan

## Executive summary
- UI with PySide6 and offline-first TTS pipeline is runnable via `uv` scripts.
- Plugin architecture supports Silero and VibeVoice; config controls engine, device, attention backend and quantization.
- AI Edit uses local Qwen or VibeVoice API.
- Documentation covers UI flows, timeline and VibeVoice setup.
- Main risks: VRAM usage, Russian diction quality, licensing of voices.
- Quick wins: add pre-commit config, torch guidance for Silero, offline STT instructions.

## Status board
| Area | Item | Status | Source |
|---|---|---|---|
| UI | Main window, editor, mixer | Done | README.md |
| UI | Timeline NLE guide screenshots | Planned | TODO.md |
| TTS | Silero and VibeVoice engines | Done | CHANGELOG.md |
| TTS | VibeVoice integration tasks | In progress | TODO.md |
| LLM | `llm` config block and AI Edit | Done | README.md |
| LLM | Tests for editors | Planned | TODO.md |
| ASR | Borealis integration | Planned | ROADMAP.md |
| Packaging | `requirements.txt`, `freeze_reqs.py` | Done | CHANGELOG.md |
| Packaging | pre-commit and one-shot script | Planned | TODO.md |
| CI | Ruff on core/ui/tests | Done | CHANGELOG.md |
| CI | VibeVoice GPU test | Planned | TODO.md |
| Docs | UI and timeline guides | Done | CHANGELOG.md |
| Docs | Offline STT download guide | Planned | TODO.md |

## Borealis ASR integration
| Step | Outcome | Effort |
|---|---|---|
| Model confirmation | transformers + CUDA model (`trust_remote_code=True`) | M |
| Offline deployment | local download and GPU setup | L |
| API adapters | `POST /v1/audio/transcriptions`, `POST /v1/audio/chunked` | M |
| Generation params | defaults: `max_new_tokens≈350`, `top_p=0.9`, `top_k=50`, `temperature=0.2` | S |
| Provider registration | config and model registry entry | M |
| Post-processing | text normalization, VAD, punctuation | M |
| Quality tests | WER/SER targets, sample audio tests | M |
| Realtime & batch | streaming and file jobs | L |
| Version logging | log and pin checkpoint & feature extractor versions | S |
| Telemetry | structured logs with params and versions | S |

## LLM integrations
| Provider | Use cases | Config keys | Notes |
|---|---|---|---|
| Qwen (local) | AI Edit offline | `llm.family`, `llm.model_path`, `llm.auto_download` | large model, add tests |
| OpenAI ChatGPT | cloud fallback | to add: `openai.api_key`, model selector | cost and API limits |
| Yandex | regional alternative | to add: `yandex.api_key` | availability depends on region |
| Live mode (Llama.cpp/Mistral) | real-time assistant | `live_mode.enabled`, `talk_llama_fast_path` | package binary, measure latency |

## TTS integrations
### Offline engines
- Silero: document torch/torchaudio install and manual model fetch.
- Coqui XTTS-v2, Dia, MARS5, Orpheus: store weights under `models/tts/<engine>` and register.

### Online or hybrid engines
- VibeVoice: install binary, support model variants (300M/1.5B/Q8), attention backends (`sdpa`, `flash`, `torch`), quantization and VRAM logging, chunking long scripts, `tts.autosave_minutes`, `tts.force_offload`.
- gTTS: install `gTTS` package, requires internet.
- Hume, Gemini, OpenAI TTS, Minimax, Chatterbox: implement API adapters and key management.

## Quality and ops
- Add unit tests for editors, torch fallback and `ensure_package` logic.
- Run core tests on CPU and optional GPU suite for VibeVoice and Borealis.
- Keep CI with Ruff/Mypy/Pytest; add pre-commit hooks and one-shot dev script.
- Use `uv` with pinned `requirements.txt` regenerated via `tools/freeze_reqs.py`.

## Roadmap (8–12 weeks)
| Sprint | Focus | Deliverables |
|---|---|---|
| 1 (weeks 1–2) | Packaging & CI hardening | pre-commit config, `dev_checks.sh`, torch guidance |
| 2 (weeks 3–4) | TTS pool finalization | VibeVoice quantization & logging, offline STT docs |
| 3 (weeks 5–6) | Borealis ASR | model to quality tests (steps 1–7 above) |
| 4 (weeks 7–8) | LLM connectors & live mode | ChatGPT/Yandex adapters, live mode packaging, timeline screenshots |

## Definition of Done
- `ruff`, `mypy`, `pytest` green.
- Docs and changelog updated.
- Models pinned with logged versions.
- Reproducible install via `requirements.txt` and `uv`.

## Risk register
| Risk | Mitigation |
|---|---|
| High VRAM usage | quantization, `tts.force_offload`, document hardware needs |
| Russian diction quality | default RU models and human review |
| Licensing and voice cloning | user warnings and terms of use |
| Online API instability | provide offline fallbacks and error handling |
| Reproducibility | log and pin model versions |
