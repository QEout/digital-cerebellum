"""
WebSocket visualization server — streams cerebellum events to the browser.

    from digital_cerebellum.viz.server import start_server
    start_server()  # opens http://localhost:8765
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from digital_cerebellum.viz.event_bus import event_bus, CerebellumEvent

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    task = asyncio.create_task(_broadcast_loop())
    yield
    task.cancel()


app = FastAPI(title="Digital Cerebellum Viz", lifespan=_lifespan)

_ws_clients: list[WebSocket] = []
_event_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=2048)


def _on_event(evt: CerebellumEvent):
    """EventBus callback — push to async queue (thread-safe from any thread)."""
    d = evt.to_dict()
    try:
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(_event_queue.put_nowait, d)
    except RuntimeError:
        try:
            _event_queue.put_nowait(d)
        except asyncio.QueueFull:
            pass


event_bus.subscribe(_on_event)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    log.info("Viz client connected (%d total)", len(_ws_clients))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        _ws_clients.remove(ws)
        log.info("Viz client disconnected (%d total)", len(_ws_clients))


async def _broadcast_loop():
    """Drain the event queue and broadcast to all WebSocket clients."""
    while True:
        evt = await _event_queue.get()
        payload = json.dumps(evt)
        dead: list[WebSocket] = []
        for ws in _ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _ws_clients.remove(ws)


# ── Mode switching (aim / tank / idle) ──

current_mode: str = "idle"
_requested_mode: str | None = None
mode_switch_event = threading.Event()
reset_event = threading.Event()


@app.post("/api/mode/{mode}")
async def switch_mode(mode: str):
    """Switch between demo modes. Stops current, starts new."""
    global _requested_mode
    if mode not in ("aim", "tank", "idle"):
        return JSONResponse({"error": "unknown mode"}, status_code=400)
    _requested_mode = mode
    mode_switch_event.set()
    return JSONResponse({"status": "ok", "mode": mode})


@app.post("/api/reset")
async def api_reset():
    """Signal the demo loop to reset the cerebellum."""
    reset_event.set()
    return JSONResponse({"status": "reset_requested"})


def consume_mode_request() -> str | None:
    """Called by demo loop to check for pending mode switch."""
    global _requested_mode
    if mode_switch_event.is_set():
        mode_switch_event.clear()
        m = _requested_mode
        _requested_mode = None
        return m
    return None


@app.get("/")
async def index():
    html_file = STATIC_DIR / "index.html"
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def start_server(host: str = "0.0.0.0", port: int = 8765):
    """Launch the viz server (blocking)."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")
