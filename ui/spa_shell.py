"""PySide6 shell embedding the web SPA via QWebEngineView."""

from __future__ import annotations

import threading

import uvicorn
from PySide6.QtCore import QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

from server.main import app as fastapi_app


def run_server() -> None:
    """Run the FastAPI server in a background thread."""
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="info")


def main() -> None:
    """Launch the Qt shell and embed the SPA."""
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    qt_app = QApplication([])
    view = QWebEngineView()
    view.setUrl(QUrl("http://127.0.0.1:8000"))
    view.setWindowTitle("Revoice")
    view.resize(1280, 720)
    view.show()
    qt_app.exec()


if __name__ == "__main__":
    main()
