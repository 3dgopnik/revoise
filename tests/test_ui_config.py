from __future__ import annotations

import json

import ui.config as config


def test_load_config_returns_full_defaults_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json", raising=False)
    monkeypatch.setattr(config, "KEY_FILE", tmp_path / "config.key", raising=False)

    result = config.load_config()

    assert isinstance(result, config.ConfigValues)
    assert result == config.DEFAULT_CONFIG
    assert len(result) == len(config.ConfigValues._fields)
    assert result.preset == "None"
    assert result.whisper_model == "base"


def test_load_config_applies_string_defaults_for_missing_keys(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_data = {
        "allow_beep_fallback": True,
        "auto_download_models": False,
        "out_dir": "custom/out",
        "language": "en",
        "speed_pct": 110,
        "min_gap_ms": 250,
        "read_numbers": True,
        "spell_latin": True,
    }
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    monkeypatch.setattr(config, "CONFIG_FILE", config_path, raising=False)
    monkeypatch.setattr(config, "KEY_FILE", tmp_path / "config.key", raising=False)

    result = config.load_config()

    assert isinstance(result, config.ConfigValues)
    assert result.preset == config.DEFAULT_CONFIG.preset
    assert result.whisper_model == config.DEFAULT_CONFIG.whisper_model
    assert result.out_dir == "custom/out"
    assert result.language == "en"
    assert result.allow_beep_fallback is True
    assert result.auto_download_models is False
    assert result.auto_install_packages is config.DEFAULT_CONFIG.auto_install_packages
    assert result.speed_pct == 110
    assert result.min_gap_ms == 250
    assert result.read_numbers is True
    assert result.spell_latin is True
    assert len(result) == len(config.ConfigValues._fields)
