"""Utility functions for loading and saving user configuration."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, NamedTuple

try:
    from cryptography.fernet import Fernet  # type: ignore
    _HAS_FERNET = True
except Exception:  # pragma: no cover - optional dependency
    Fernet = None  # type: ignore
    _HAS_FERNET = False

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config.json"
KEY_FILE = BASE_DIR / "config.key"
DEFAULT_OUT_DIR = str((BASE_DIR / "output").resolve())


class ConfigValues(NamedTuple):
    """Structured configuration values returned by :func:`load_config`."""

    yandex_key: str
    chatgpt_key: str
    allow_beep_fallback: bool
    auto_download_models: bool
    out_dir: str
    language: str
    preset: str
    whisper_model: str
    speed_pct: int
    min_gap_ms: int
    read_numbers: bool
    spell_latin: bool


DEFAULT_CONFIG = ConfigValues(
    "",
    "",
    False,
    True,
    DEFAULT_OUT_DIR,
    "ru",
    "None",
    "base",
    100,
    350,
    False,
    False,
)


def _get_str_option(data: dict[str, Any], key: str, default: str) -> str:
    """Return a string option from *data* or *default* if missing/invalid."""

    value = data.get(key, default)
    return value if isinstance(value, str) else default


def _get_int_option(data: dict[str, Any], key: str, default: int) -> int:
    """Return an integer option from *data*, falling back to *default*."""

    raw_value = data.get(key, default)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def _get_cipher() -> Fernet | None:
    """Retrieve a Fernet cipher if available, otherwise return None."""
    if not _HAS_FERNET:
        return None
    if not KEY_FILE.exists():
        KEY_FILE.write_bytes(Fernet.generate_key())
    key = KEY_FILE.read_bytes()
    return Fernet(key)


def load_config() -> ConfigValues:
    """Load API keys and user preferences from the config file."""
    default_out = DEFAULT_OUT_DIR
    if not CONFIG_FILE.exists():
        return DEFAULT_CONFIG
    data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    cipher = _get_cipher()
    yandex_enc = data.get("yandex_key", "")
    chatgpt_enc = data.get("chatgpt_key", "")
    allow_beep = bool(data.get("allow_beep_fallback", DEFAULT_CONFIG.allow_beep_fallback))
    auto_download = bool(data.get("auto_download_models", DEFAULT_CONFIG.auto_download_models))
    out_dir = _get_str_option(data, "out_dir", default_out)
    language = _get_str_option(data, "language", DEFAULT_CONFIG.language)
    preset = _get_str_option(data, "preset", DEFAULT_CONFIG.preset)
    whisper = _get_str_option(data, "whisper_model", DEFAULT_CONFIG.whisper_model)
    speed_pct = _get_int_option(data, "speed_pct", DEFAULT_CONFIG.speed_pct)
    min_gap_ms = _get_int_option(data, "min_gap_ms", DEFAULT_CONFIG.min_gap_ms)
    read_numbers = bool(data.get("read_numbers", DEFAULT_CONFIG.read_numbers))
    spell_latin = bool(data.get("spell_latin", DEFAULT_CONFIG.spell_latin))
    if cipher:
        yandex_key = cipher.decrypt(yandex_enc.encode()).decode() if yandex_enc else ""
        chatgpt_key = cipher.decrypt(chatgpt_enc.encode()).decode() if chatgpt_enc else ""
    else:
        yandex_key = base64.b64decode(yandex_enc).decode() if yandex_enc else ""
        chatgpt_key = base64.b64decode(chatgpt_enc).decode() if chatgpt_enc else ""
    return ConfigValues(
        yandex_key,
        chatgpt_key,
        allow_beep,
        auto_download,
        out_dir,
        language,
        preset,
        whisper,
        speed_pct,
        min_gap_ms,
        read_numbers,
        spell_latin,
    )


def save_config(
    yandex_key: str,
    chatgpt_key: str,
    allow_beep_fallback: bool,
    auto_download_models: bool,
    out_dir: str,
    language: str,
    preset: str,
    whisper_model: str,
    speed_pct: int,
    min_gap_ms: int,
    read_numbers: bool,
    spell_latin: bool,
) -> None:
    """Encrypt and save API keys and user preferences to the config file."""
    cipher = _get_cipher()
    if cipher:
        data = {
            "yandex_key": cipher.encrypt(yandex_key.encode()).decode() if yandex_key else "",
            "chatgpt_key": cipher.encrypt(chatgpt_key.encode()).decode() if chatgpt_key else "",
            "allow_beep_fallback": allow_beep_fallback,
            "auto_download_models": auto_download_models,
            "out_dir": out_dir,
            "language": language,
            "preset": preset,
            "whisper_model": whisper_model,
            "speed_pct": speed_pct,
            "min_gap_ms": min_gap_ms,
            "read_numbers": read_numbers,
            "spell_latin": spell_latin,
        }
    else:
        data = {
            "yandex_key": base64.b64encode(yandex_key.encode()).decode() if yandex_key else "",
            "chatgpt_key": base64.b64encode(chatgpt_key.encode()).decode() if chatgpt_key else "",
            "allow_beep_fallback": allow_beep_fallback,
            "auto_download_models": auto_download_models,
            "out_dir": out_dir,
            "language": language,
            "preset": preset,
            "whisper_model": whisper_model,
            "speed_pct": speed_pct,
            "min_gap_ms": min_gap_ms,
            "read_numbers": read_numbers,
            "spell_latin": spell_latin,
        }
    CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

