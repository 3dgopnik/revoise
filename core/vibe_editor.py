from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)


class VibeEditor:
    """Client for the VibeVoice text editing and translation API."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = base_url or os.getenv("VIBEVOICE_API_URL", "https://api.vibevoice.ai")
        self.api_key = api_key or os.getenv("VIBEVOICE_API_KEY")

    def edit_text(self, text: str, target_languages: list[str]) -> dict[str, Any]:
        """Correct text and translate using the VibeVoice service."""
        url = f"{self.base_url.rstrip('/')}/edit"
        payload = {"text": text, "targets": target_languages}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("VibeVoice request failed: %s", exc)
            data = {"source": text}
            for lang in target_languages:
                data[lang] = text
        return data


__all__ = ["VibeEditor"]
