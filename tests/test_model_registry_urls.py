"""Tests for model URL validation."""
from tools.validate_model_urls import validate_registry


def test_model_urls_respond_ok() -> None:
    """Model registry URLs should return 2xx on HEAD."""
    validate_registry()
