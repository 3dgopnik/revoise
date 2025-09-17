# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `tools/fetch_stt_models.py` script with Windows wrapper for prefetching Whisper models.
- Whisper `base` speech-to-text model registered alongside existing downloads and documented default configuration.
- VibeEditor for text editing via VibeVoice API.
- AI Edit dialog can invoke Qwen or VibeVoice editors.
- AI edit dialog for custom instructions and language selection.
- Initial changelog.
- Interactive package installer with optional requirements pinning.
- UI preflight ensures `faster-whisper` and Silero dependencies before launch and surfaces failures via dialog.
- Unified `ensure_model` downloader with progress, retries and optional SHA256 validation.
- `requirements.txt` capturing essential runtime dependencies.
- `tools/freeze_reqs.py` script to regenerate pinned requirements.
- Configurable LLM block (`llm.family`, `llm.model_path`, `llm.auto_download`).
- Top-level `tts_engine` and `preferences.pin_dependencies` settings.
- `preferences.auto_install_packages` toggle to skip interactive dependency installation prompts.
- Manifest-driven FFmpeg downloader storing path and version in config.
- GUI dialog for external binary downloads with cancelable progress bar.
- Optional `imageio-ffmpeg` integration via `use_imageio_ffmpeg` and `externals.ffmpeg` settings for automatic FFmpeg setup.
- `run_ui.sh` and `run_ui.bat` helper scripts for launching the UI with uv.
- Pluggable TTS engine framework with Silero and VibeVoice engines.
- Silero engine parameters for rate, pitch, style and preset with `.rvpreset` presets and UI selection.
- Structured `tts` configuration block with per-engine settings and dataclass loader.
- VibeVoice model weights and Silero locale packs registered in model registry.
- `fetch_tts_models.py` subcommands for VibeVoice and Silero with progress bars and caching.
- Script parser validating `Speaker N:` lines and chunking text.
- Deterministic per-speaker seeding and optional `silence_gap_ms` insertion.
- Autosave checkpoints with crash/OOM resume support.
- Optional GPU offload with peak VRAM/time logging.
- Tests for dialogue parsing, Russian text segmentation, and TTS engine registry with GPU VibeVoice integration test.
- VibeVoice engine selectable in UI with dynamic speaker listing and missing binary warning.
- Support for `NO_SSL_VERIFY=1` to disable SSL certificate verification during Silero downloads.
- Settings dialog toggle to enable or disable SSL certificate verification for model downloads, persisted in `config.json` and applied during TTS initialization.
- Regression test covering UI config defaults when `config.json` is absent.
- `revoice-check` QA runner CLI to execute uv-based installation, Ruff, MyPy, and pytest with aggregated reporting.

### Changed
- Install TTS dependencies into .venv using shared pkg_installer.
- Lazily import `llama-cpp` in `QwenEditor` and ensure required model files.
- Replace `ensure_tts_dependencies` calls with direct `ensure_package` usage and lazy heavy imports.
- Advanced TTS options moved to a dedicated Settings dialog and main UI simplified.
- Model loading now routes through the new `ensure_model` helper.
- Model downloads now support optional progress callbacks and debug-level logging updates.
- Trim default `project.dependencies` to essential packages only.
- Rename project package to `revoice`.
- CI installs dependencies with `uv pip install -r requirements.txt` and runs Ruff on `core`, `ui`, and `tests`.
- `tts_registry` now reads `tts_engine` from the top level of `config.json`.
- TTS registry moved to `core.tts.registry` and all imports updated.
- Silero engine now checks for `torchaudio` alongside `torch`.
- Configuration now uses `tts.default_engine` instead of top-level `tts_engine`.
- Silero TTS now uses the local torch hub cache when available and disables autofetch to avoid network access.
- Silero TTS now verifies cached `.pt` files and loads directly from the local cache, prompting for manual download when missing.
- Silero download errors now reference `SSL_CERT_FILE`, `HTTPS_PROXY`, and `NO_SSL_VERIFY` for troubleshooting.
- Silero download failure hint now points to `python tools/fetch_tts_models.py silero --language <code>`.
- Silero TTS now marks cached loads as `source="local"` to keep torch hub on the offline path.

### Removed
- Removed legacy bootstrap and launcher scripts.
- Removed `faster-whisper` and `omegaconf` from core dependencies.
- Removed portable mode.
- Removed lazy install support.
- Deprecated `speed_jitter` parameter and related UI controls.

### Fixed
- Tests mock package installs and model downloads, dropping portable bootstrap assumptions.
- Sorted standard library imports in `tests/test_coqui_no_qmessagebox.py`.
- Sorted standard library imports in `tests/test_edited_text.py`.
- CI now sets up a uv virtual environment before installing dependencies.
- `ui.config.load_config` now always returns preset and Whisper defaults even without `config.json`.
- UI config loader now returns the complete default tuple (including preset) when `config.json` is missing.
- Restored the default SSL context after Silero downloads when `NO_SSL_VERIFY=1` to avoid global side effects.

### Docs
- Documented manual STT model download workflow in README and quickstart.
- Documented UI `config.json` keys, defaults, and storage behaviour.
- Documented installation with `uv pip`, lazy dependency/model downloads,
  launch via `uv run python -m ui.main`, and `tools/freeze_reqs.py`.
- Clarified dev dependency installation for quality checks.
- Explain manual Silero model fetch.
- Describe offline mode with manual Silero fetch and disabled auto-download.
- Track VibeVoice TTS integration in TODO.
- Explain `.rvpreset` preset loading and selection.
- Document structured TTS configuration keys.
- Mention script parser, autosave and offload options in README.
- Added VibeVoice and timeline guides; expanded README with uv quick start, model fetch commands, and engine toggle examples.
- Document VibeVoice binary and model installation.
- Documented troubleshooting with `SSL_CERT_FILE`, `HTTPS_PROXY`, and `NO_SSL_VERIFY`.
- Documented the SSL verification toggle in configuration and troubleshooting guides.
- Explained manual Silero archive placement and proxy variables for corporate networks.
- Added UI guide and described new Settings dialog.
- Documented automatic package installation preference in configuration guide.
- Added Borealis ASR integration roadmap.
- Added quickstart, CLI, configuration, troubleshooting, and development plan guides.
- Documented the `revoice-check` workflow in README and CLI guide, including report formats and storage options.
