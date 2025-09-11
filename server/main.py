"""FastAPI backend serving the SPA and providing WebSocket endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Revoice SPA")

_dist_path = Path(__file__).parent.parent / "web"
app.mount("/static", StaticFiles(directory=_dist_path), name="static")


@app.get("/")
async def index() -> HTMLResponse:
    """Serve the pre-built SPA entrypoint."""
    index_path = _dist_path / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


class ConnectionManager:
    """Track active WebSocket connections."""

    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: str) -> None:
        for connection in list(self.active):
            await connection.send_text(message)


progress_manager = ConnectionManager()
preview_manager = ConnectionManager()


@app.websocket("/ws/progress")
async def progress_endpoint(ws: WebSocket) -> None:
    """Broadcast textual progress updates to all listeners."""
    await progress_manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            await progress_manager.broadcast(data)
    except WebSocketDisconnect:
        progress_manager.disconnect(ws)


@app.websocket("/ws/preview")
async def preview_endpoint(ws: WebSocket) -> None:
    """Echo binary chunks to simulate fast preview buffering."""
    await preview_manager.connect(ws)
    try:
        while True:
            data = await ws.receive_bytes()
            for connection in list(preview_manager.active):
                if connection is not ws:
                    await connection.send_bytes(data)
    except WebSocketDisconnect:
        preview_manager.disconnect(ws)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
