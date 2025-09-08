import pytest

from core import model_service


def test_ensure_model_existing(tmp_path, monkeypatch):
    target = tmp_path / "existing.bin"
    target.write_text("data")
    path = model_service.ensure_model(
        "llm", "existing.bin", tmp_path, (tmp_path / "src.bin").as_uri()
    )
    assert path == target


def test_ensure_model_download_and_sha(tmp_path, monkeypatch):
    src = tmp_path / "src.bin"
    data = b"payload"
    src.write_bytes(data)
    url = src.as_uri()
    sha = __import__("hashlib").sha256(data).hexdigest()
    monkeypatch.setattr(model_service, "_auto_download_enabled", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "y")
    dest = tmp_path / "dest"
    path = model_service.ensure_model("llm", "file.bin", dest, url, sha256=sha)
    assert path.exists() and path.read_bytes() == data


def test_ensure_model_decline(tmp_path, monkeypatch):
    src = tmp_path / "src.bin"
    src.write_text("data")
    url = src.as_uri()
    monkeypatch.setattr(model_service, "_auto_download_enabled", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "n")
    with pytest.raises(FileNotFoundError):
        model_service.ensure_model("llm", "file.bin", tmp_path, url)
