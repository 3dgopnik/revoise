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

### Changed
- Install TTS dependencies into .venv using shared pkg_installer.
- Lazily import `llama-cpp` in `QwenEditor` and ensure required model files.
- Replace `ensure_tts_dependencies` calls with direct `ensure_package` usage and lazy heavy imports.
- Model loading now routes through the new `ensure_model` helper.

### Removed
- Removed portable bootstrap and launcher scripts.

### Docs
- Updated launch instructions to use `uv run python -m ui.main`.
