from __future__ import annotations

from pathlib import Path

from tools import fetch_stt_models


def test_fetch_stt_models_invokes_ensure_model(monkeypatch, tmp_path):
    from core import model_service

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_service, "_MODEL_REGISTRY", {
        "stt": {
            "base": {
                "files": ["model.bin"],
                "optional_files": [],
                "base_urls": ["https://example.com/"],
            }
        }
    })
    monkeypatch.setattr(model_service, "_MODEL_PATH_CACHE", {})
    monkeypatch.setattr(model_service, "_CONFIG_CACHE", {})

    calls: list[tuple[str, str, Path]] = []

    def fake_ensure_model(kind: str, file_name: str, dest_dir: Path, url: str, sha256=None):
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / file_name
        target.write_text("dummy")
        calls.append((kind, file_name, dest))
        return target

    monkeypatch.setattr(model_service, "ensure_model", fake_ensure_model)

    assert fetch_stt_models.fetch_models(["base"]) is True
    assert calls
    assert any(path.as_posix().endswith("models/stt/base") for *_rest, path in calls)

