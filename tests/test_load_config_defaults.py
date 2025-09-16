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
    expected_defaults = (
        "",
        "",
        False,
        True,
        expected_output_dir,
        "ru",
        "None",
        "base",
        100,
        350,
        False,
        False,
    )

    assert len(defaults) == 12
    assert defaults == expected_defaults
