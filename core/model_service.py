from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from .model_manager import DownloadError

# Cache for resolved model paths
_MODEL_PATH_CACHE: dict[tuple[str, str], Path] = {}
_MODEL_REGISTRY: dict[str, dict[str, Any]] | None = None
_AUTO_DOWNLOAD_OVERRIDE: bool | None = None


def _load_registry() -> dict[str, dict[str, Any]]:
    global _MODEL_REGISTRY
    if _MODEL_REGISTRY is None:
        registry_path = Path("models") / "model_registry.json"
        with registry_path.open(encoding="utf-8") as fh:
            _MODEL_REGISTRY = json.load(fh)
    return _MODEL_REGISTRY


def _auto_download_enabled() -> bool:
    if _AUTO_DOWNLOAD_OVERRIDE is not None:
        return _AUTO_DOWNLOAD_OVERRIDE
    try:
        with open("config.json", encoding="utf-8") as fh:
            data = json.load(fh)
        return bool(data.get("auto_download_models", True))
    except Exception:
        return True


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _estimate_size_mb(url: str) -> float:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as resp:
            size = resp.headers.get("Content-Length")
            if size:
                return int(size) / (1024 * 1024)
    except Exception:
        if url.startswith("file://"):
            try:
                path = Path(urllib.request.url2pathname(url[7:]))
                return path.stat().st_size / (1024 * 1024)
            except Exception:
                pass
    return 0.0


def _download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url) as resp, target.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        read = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            read += len(chunk)
            if total:
                pct = read * 100 // total
                print(f"\rDownloading {target.name}: {pct}%", end="", file=sys.stderr)
        if total:
            print(file=sys.stderr)


def ensure_model(
    kind: str,
    name: str,
    dest_dir: Path | str,
    url: str,
    sha256: str | None = None,
) -> Path:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / name
    if target.exists():
        if sha256 and _file_sha256(target) != sha256:
            target.unlink()
        else:
            return target
    size_mb = _estimate_size_mb(url)
    if not _auto_download_enabled():
        answer = (
            input(f"Download {kind} model '{name}' (~{size_mb:.1f} MB) to {target}? [y/N]: ")
            .strip()
            .lower()
        )
        if answer not in {"y", "yes"}:
            raise FileNotFoundError(f"Model '{name}' is missing.")
    for attempt in range(3):
        try:
            _download(url, target)
            break
        except Exception as exc:
            if attempt == 2:
                raise DownloadError(str(exc)) from exc
            time.sleep(1)
    if sha256 and _file_sha256(target) != sha256:
        target.unlink(missing_ok=True)
        raise DownloadError("SHA256 mismatch")
    return target


def get_model_path(
    name: str,
    category: str,
    *,
    parent: Any | None = None,
    auto_download: bool | None = None,
) -> Path:
    key = (name, category)
    cached = _MODEL_PATH_CACHE.get(key)
    if cached is not None:
        return cached
    registry = _load_registry()
    entry = registry.get(category, {}).get(name)
    if entry is None:
        raise FileNotFoundError(f"Model '{name}' not found in registry")
    global _AUTO_DOWNLOAD_OVERRIDE
    if auto_download is not None:
        _AUTO_DOWNLOAD_OVERRIDE = auto_download
    try:
        if category == "stt":
            model_dir = Path("models") / category / name
            files = entry.get("files", [])
            optional = entry.get("optional_files", [])
            bases = entry.get("base_urls", [])
            for file_name in files + optional:
                ok = False
                for base in bases:
                    url = base + file_name
                    try:
                        ensure_model(category, file_name, model_dir, url)
                        ok = True
                        break
                    except DownloadError:
                        continue
                if not ok and file_name in files:
                    raise DownloadError(f"Failed to download file '{file_name}'")
            path = model_dir
        else:
            urls = entry if isinstance(entry, list) else entry.get("urls", [])
            sha = entry.get("sha256") if isinstance(entry, dict) else None
            dest_dir = Path("models") / category
            path = None
            for url in urls:
                try:
                    path = ensure_model(category, name, dest_dir, url, sha256=sha)
                    break
                except DownloadError:
                    path = None
                    continue
            if path is None:
                raise DownloadError(f"Failed to download model '{name}'")
        _MODEL_PATH_CACHE[key] = path
        return path
    finally:
        _AUTO_DOWNLOAD_OVERRIDE = None


def clear_cache() -> None:
    _MODEL_PATH_CACHE.clear()


__all__ = ["get_model_path", "clear_cache", "ensure_model"]
