# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- AI edit dialog for custom instructions and language selection.
- Initial changelog.
- Interactive package installer with optional requirements pinning.
- Unified `ensure_model` downloader with progress, retries and optional SHA256 validation.
- `requirements.txt` capturing essential runtime dependencies.
- `tools/freeze_reqs.py` script to regenerate pinned requirements.
- Configurable LLM block (`llm.family`, `llm.model_path`, `llm.auto_download`).
- Top-level `tts_engine` and `preferences.pin_dependencies` settings.
- Manifest-driven FFmpeg downloader storing path and version in config.
- GUI dialog for external binary downloads with cancelable progress bar.

### Changed
- Install TTS dependencies into .venv using shared pkg_installer.
- Lazily import `llama-cpp` in `QwenEditor` and ensure required model files.
- Replace `ensure_tts_dependencies` calls with direct `ensure_package` usage and lazy heavy imports.
- Model loading now routes through the new `ensure_model` helper.
- Trim default `project.dependencies` to essential packages only.
- Rename project package to `revoice`.
- CI installs dependencies with `uv pip install -r requirements.txt` and runs Ruff on `core`, `ui`, and `tests`.
- `tts_registry` now reads `tts_engine` from the top level of `config.json`.

### Removed
- Removed legacy bootstrap and launcher scripts.
- Removed `faster-whisper` and `omegaconf` from core dependencies.
- Removed portable mode.
- Removed lazy install support.

### Fixed
- Tests mock package installs and model downloads, dropping portable bootstrap assumptions.

### Docs
- Documented installation with `uv pip`, lazy dependency/model downloads,
  launch via `uv run python -m ui.main`, and `tools/freeze_reqs.py`.
