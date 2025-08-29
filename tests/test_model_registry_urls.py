"""Tests for model URL validation."""

from __future__ import annotations

from tools.validate_model_urls import validate_registry


class _DummyResponse:
    def __enter__(self) -> _DummyResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def getcode(self) -> int:
        return 200


def _fake_urlopen(*_args: object, **_kwargs: object) -> _DummyResponse:
    return _DummyResponse()


def test_model_urls_respond_ok() -> None:
    """Model registry URLs should return 2xx on HEAD."""
    validate_registry(urlopen_fn=_fake_urlopen)
