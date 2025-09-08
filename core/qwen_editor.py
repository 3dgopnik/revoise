from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from . import model_service
from .pkg_installer import ensure_package

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from llama_cpp import Llama

logger = logging.getLogger(__name__)


class QwenEditor:
    """Utility wrapper around a local Qwen model for text editing and translation."""

    def __init__(self) -> None:
        self._llm: Llama | None = None

    def _ensure_model(self) -> None:
        """Lazy-load the Qwen model via llama.cpp."""
        if self._llm is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                ensure_package(
                    "llama-cpp-python",
                    "llama-cpp-python is required for QwenEditor.",
                )
                from llama_cpp import Llama
            path = model_service.ensure_model("qwen2.5", "llm")
            logger.info("Loading Qwen model from %s", path)
            self._llm = Llama(model_path=str(path), n_ctx=4096, verbose=False)

    def edit_text(self, text: str, target_languages: list[str]) -> dict[str, str]:
        """Correct text, insert stress markers/pauses and translate it."""
        self._ensure_model()
        assert self._llm is not None

        prompt = (
            "You are a text editing assistant. "
            "Fix spelling and grammar in the provided Russian text, "
            "insert stress marks using '+' before stressed vowels, "
            "and insert [[PAUSE=ms]] tags for natural pauses. "
            "After correction translate the text into these languages: "
            f"{', '.join(target_languages)}. "
            "Return a JSON object with keys 'source' for the corrected text and one key per language."
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]

        logger.debug("Requesting Qwen completion for %s", target_languages)
        result = self._llm.create_chat_completion(messages=messages)
        content = result["choices"][0]["message"]["content"]
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Qwen output was not valid JSON: %s", content)
            data = {"source": text}
            for lang in target_languages:
                data[lang] = text
        return data


__all__ = ["QwenEditor"]
