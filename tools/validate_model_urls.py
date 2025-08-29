"""Validate model URLs by performing HTTP HEAD requests."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def validate_registry(registry_path: Path | None = None) -> None:
    """Perform HTTP HEAD requests for every URL in the model registry."""
    if registry_path is None:
        registry_path = Path(__file__).resolve().parents[1] / "models" / "model_registry.json"

    with registry_path.open("r", encoding="utf-8") as fh:
        registry = json.load(fh)

    errors: list[str] = []

    for category, models in registry.items():
        for name, data in models.items():
            urls = data.get("urls") if isinstance(data, dict) else data
            for url in urls:
                try:
                    request = Request(url, method="HEAD")
                    with urlopen(request, timeout=10) as response:
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
