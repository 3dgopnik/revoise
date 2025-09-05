# TODO

- Add progress display and retry logic to `tools/bootstrap_portable.py`.
- Allow selecting a subset of TTS engines in `revoice_portable` scripts.
- Add guidance for installing optional 'torch' dependency for Silero engine.
- Add shfmt configuration to enforce shell script style.
- Cover `bootstrap_portable.py` with basic tests for STT model selection.
- Notify users in the UI when `QwenEditor` is unavailable.
- Add optional cleanup of Hugging Face cache directory on exit.
- Add unit tests for torch installation fallback logic in `ensure_tts_dependencies`.
