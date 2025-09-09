"""Utilities for ensuring external binaries are available."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path


def find_ffmpeg() -> Path | None:
    """Return path to ffmpeg if it is already available."""
    exe = shutil.which("ffmpeg")
    if exe:
        return Path(exe)
    local_dir = Path(__file__).resolve().parent.parent / "bin"
    for candidate in ("ffmpeg", "ffmpeg.exe"):
        path = local_dir / candidate
        if path.exists():
            return path
    return None


def download_and_extract(url: str, sha256: str, archive: Mapping[str, str], dest: Path) -> Path:
    """Download ``url`` to ``dest`` and return extracted binary path."""
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = Path(tmpdir) / Path(url).name
        with subprocess.Popen(["curl", "-L", url], stdout=subprocess.PIPE) as proc:
            if proc.stdout is None:  # pragma: no cover - defensive
                raise RuntimeError("failed to download ffmpeg")
            with open(tmp_file, "wb") as fh:
                shutil.copyfileobj(proc.stdout, fh)
        h = hashlib.sha256()
        with open(tmp_file, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != sha256:
            raise RuntimeError("ffmpeg download checksum mismatch")
        if tmp_file.suffix == ".zip":
            with zipfile.ZipFile(tmp_file) as zf:
                zf.extractall(dest)
        else:
            with tarfile.open(tmp_file, mode="r:*") as tf:
                tf.extractall(dest)
    pattern = archive.get("binary", "ffmpeg")
    matches = list(dest.glob(pattern))
    if not matches:
        raise RuntimeError("ffmpeg binary not found after extraction")
    return matches[0]


def ensure_ffmpeg(manifest_path: Path | None = None) -> tuple[Path, str]:
    """Ensure ffmpeg exists and return (path, version)."""
    existing = find_ffmpeg()
    if existing:
        path = existing
    else:
        manifest_path = manifest_path or Path("tools/externals_manifest.json")
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        system = platform.system().lower()
        entry = manifest["ffmpeg"]["windows" if system.startswith("win") else "linux"]
        path = download_and_extract(entry["url"], entry["sha256"], entry.get("archive", {}), Path("bin"))
        path.chmod(path.stat().st_mode | 0o111)
    result = subprocess.run([str(path), "-version"], stdout=subprocess.PIPE, text=True, check=True)
    version_line = result.stdout.splitlines()[0]
    version = version_line.split()[2]
    return path, version

