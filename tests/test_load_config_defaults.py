"""Tests for the UI configuration loader defaults."""

from __future__ import annotations

from pathlib import Path

from ui import config as ui_config


def test_load_config_returns_expected_defaults(tmp_path, monkeypatch):
    """Ensure defaults are returned when the config file is missing."""
    monkeypatch.setattr(ui_config, "BASE_DIR", Path(tmp_path))
    monkeypatch.setattr(ui_config, "CONFIG_FILE", Path(tmp_path) / "config.json")
    monkeypatch.setattr(ui_config, "KEY_FILE", Path(tmp_path) / "config.key")

    defaults = ui_config.load_config()

    expected_output_dir = str((Path(tmp_path) / "output").resolve())
    assert len(defaults) == len(ui_config.ConfigValues._fields)
    assert defaults.yandex_key == ""
    assert defaults.chatgpt_key == ""
    assert defaults.allow_beep_fallback is False
    assert defaults.auto_download_models is True
    assert defaults.verify_ssl_downloads is True
    assert defaults.auto_install_packages is True
    assert defaults.out_dir == expected_output_dir
    assert defaults.language == "ru"
    assert defaults.preset == "None"
    assert defaults.whisper_model == "base"
    assert defaults.speed_pct == 100
    assert defaults.min_gap_ms == 350
    assert defaults.read_numbers is False
    assert defaults.spell_latin is False
