from __future__ import annotations

import hashlib
import os
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Mapping

from PySide6 import QtCore, QtWidgets


class DownloadWorker(QtCore.QObject):
    """Background worker that downloads and extracts an external binary."""

    progress = QtCore.Signal(int)
    finished = QtCore.Signal(Path)
    error = QtCore.Signal(str)

    def __init__(self, url: str, sha256: str, archive: Mapping[str, str]):
        super().__init__()
        self.url = url
        self.sha256 = sha256
        self.archive = archive
        self._cancel = False

    @QtCore.Slot()
    def run(self) -> None:
        try:
            dest = Path("bin")
            dest.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_file = Path(tmpdir) / Path(self.url).name
                self._download(tmp_file)
                if self._cancel:
                    return
                self._verify(tmp_file)
                self._extract(tmp_file, dest)
            pattern = self.archive.get("binary", "ffmpeg")
            matches = list(dest.glob(pattern))
            if not matches:
                raise RuntimeError("downloaded binary not found")
            path = matches[0]
            path.chmod(path.stat().st_mode | 0o111)
            self.progress.emit(100)
            self.finished.emit(path)
        except Exception as exc:  # pragma: no cover - GUI side
            self.error.emit(str(exc))

    def cancel(self) -> None:
        self._cancel = True

    def _download(self, target: Path) -> None:
        import urllib.request

        with urllib.request.urlopen(self.url) as response, open(target, "wb") as fh:
            total = int(response.getheader("Content-Length", 0))
            downloaded = 0
            while not self._cancel:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    self.progress.emit(int(downloaded * 100 / total))

    def _verify(self, file: Path) -> None:
        h = hashlib.sha256()
        with open(file, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != self.sha256:
            raise RuntimeError("checksum mismatch")

    def _extract(self, file: Path, dest: Path) -> None:
        if file.suffix == ".zip":
            with zipfile.ZipFile(file) as zf:
                zf.extractall(dest)
        else:
            with tarfile.open(file, mode="r:*") as tf:
                tf.extractall(dest)


class ExternalsDialog(QtWidgets.QDialog):
    """Dialog showing download progress for external binaries."""

    def __init__(
        self,
        url: str,
        sha256: str,
        archive: Mapping[str, str],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Downloading...")
        self._result: Path | None = None
        self._worker = DownloadWorker(url, sha256, archive)
        self._thread = QtCore.QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._progress = QtWidgets.QProgressBar(self)
        self._progress.setRange(0, 100)
        self._cancel_btn = QtWidgets.QPushButton("Cancel", self)
        self._cancel_btn.clicked.connect(self._on_cancel)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._progress)
        layout.addWidget(self._cancel_btn)

        self._thread.start()

    def _on_progress(self, value: int) -> None:
        self._progress.setValue(value)

    def _on_finished(self, path: Path) -> None:
        bin_dir = Path("bin").resolve()
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}" + os.environ.get("PATH", "")
        self._result = path
        self._thread.quit()
        self.accept()

    def _on_error(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Download failed", message)
        self._thread.quit()
        self.reject()

    def _on_cancel(self) -> None:
        self._worker.cancel()
        self._thread.quit()
        self.reject()

    def exec_with_result(self) -> Path | None:
        """Execute the dialog and return the downloaded path or ``None``."""
        self.exec()
        return self._result
