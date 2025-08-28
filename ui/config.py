"""Utility functions for loading and saving API keys configuration."""

from __future__ import annotations

import base64
import json
from pathlib import Path

try:
    from cryptography.fernet import Fernet  # type: ignore
    _HAS_FERNET = True
except Exception:  # pragma: no cover - optional dependency
    Fernet = None  # type: ignore
    _HAS_FERNET = False

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config.json"
KEY_FILE = BASE_DIR / "config.key"


def _get_cipher() -> Fernet | None:
    """Retrieve a Fernet cipher if available, otherwise return None."""
    if not _HAS_FERNET:
        return None
    if not KEY_FILE.exists():
        KEY_FILE.write_bytes(Fernet.generate_key())
    key = KEY_FILE.read_bytes()
    return Fernet(key)


def load_config() -> tuple[str, str]:
    """Load API keys from the config file, returning empty strings if absent."""
    if not CONFIG_FILE.exists():
        return "", ""
    data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    cipher = _get_cipher()
    yandex_enc = data.get("yandex_key", "")
    chatgpt_enc = data.get("chatgpt_key", "")
    if cipher:
        yandex_key = cipher.decrypt(yandex_enc.encode()).decode() if yandex_enc else ""
        chatgpt_key = cipher.decrypt(chatgpt_enc.encode()).decode() if chatgpt_enc else ""
    else:
        yandex_key = base64.b64decode(yandex_enc).decode() if yandex_enc else ""
        chatgpt_key = base64.b64decode(chatgpt_enc).decode() if chatgpt_enc else ""
    return yandex_key, chatgpt_key


def save_config(yandex_key: str, chatgpt_key: str) -> None:
    """Encrypt and save API keys to the config file."""
    cipher = _get_cipher()
    if cipher:
        data = {
            "yandex_key": cipher.encrypt(yandex_key.encode()).decode() if yandex_key else "",
            "chatgpt_key": cipher.encrypt(chatgpt_key.encode()).decode() if chatgpt_key else "",
        }
    else:
        data = {
            "yandex_key": base64.b64encode(yandex_key.encode()).decode() if yandex_key else "",
            "chatgpt_key": base64.b64encode(chatgpt_key.encode()).decode() if chatgpt_key else "",
        }
    CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
