"""Validate model URLs by performing HTTP HEAD requests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def validate_registry(
    registry_path: Path | None = None,
    urlopen_fn: Callable[[Request, int], Any] | None = None,
) -> None:
    """Perform HTTP HEAD requests for every URL in the model registry.

    Parameters
    ----------
    registry_path:
        Path to the model registry JSON file. Defaults to ``models/model_registry.json``.
    urlopen_fn:
        Function used to perform HTTP requests. Defaults to :func:`urllib.request.urlopen`.
    """
    if registry_path is None:
        registry_path = Path(__file__).resolve().parents[1] / "models" / "model_registry.json"

    if urlopen_fn is None:
        urlopen_fn = urlopen

    with registry_path.open("r", encoding="utf-8") as fh:
        registry = json.load(fh)

    errors: list[str] = []

    for category, models in registry.items():
        for name, data in models.items():
            if not isinstance(data, dict):
                urls = data
            elif "urls" in data:
                urls = data["urls"]
            else:
                base_urls = data.get("base_urls", [])
                files = data.get("files", [])
                urls = [base + file for base in base_urls for file in files]

            for url in urls:
                try:
                    request = Request(url, method="HEAD")
                    with urlopen_fn(request, timeout=10) as response:
                        status = response.getcode()
                        if not (200 <= status < 300):
                            errors.append(f"{category}/{name}: {url} -> {status}")
                except (HTTPError, URLError, TimeoutError) as exc:
                    errors.append(f"{category}/{name}: {url} -> {exc}")

    if errors:
        error_msg = "\n".join(errors)
        raise RuntimeError(f"Model URL validation failed:\n{error_msg}")


if __name__ == "__main__":
    validate_registry()
