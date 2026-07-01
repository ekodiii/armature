"""server.py — Starlette HTTP server for the armature GUI.

Routes:
  GET /api/graph        → current GraphSnapshot (or {} if none loaded)
  GET /api/graphs       → {"graphs": [...names], "active": name}
  GET /api/switch?name= → switch active graph; return {"ok": true}
  GET /api/events       → SSE stream pushed on every mtime change
  GET / (and others)    → serve gui/static/ as a static site
"""

import asyncio
import json
import os
from typing import Optional

from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from gui.reader import load_registry, read_graph

# ---------------------------------------------------------------------------
# Module-level mutable state
# ---------------------------------------------------------------------------
_snapshot: dict = {}
_watch_path: Optional[str] = None
_last_mtime: Optional[float] = None
_graph_name: Optional[str] = None

# SSE clients: each is an asyncio.Queue that receives serialised JSON strings.
_sse_clients: list[asyncio.Queue] = []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _reload() -> None:
    """Re-read the YAML at _watch_path and push to all SSE queues."""
    global _snapshot, _last_mtime
    if _watch_path is None or _graph_name is None:
        return
    try:
        _snapshot = read_graph(_graph_name, _watch_path)
        _last_mtime = os.path.getmtime(_watch_path)
    except Exception:
        return
    payload = json.dumps(_snapshot)
    for q in list(_sse_clients):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


def _switch_graph(name: str) -> bool:
    """Switch the watched graph to *name*. Returns True on success."""
    global _watch_path, _graph_name
    reg = load_registry()
    path = reg["graphs"].get(name)
    if not path or not os.path.exists(path):
        return False
    _watch_path = path
    _graph_name = name
    _reload()
    return True


# ---------------------------------------------------------------------------
# Background file watcher
# ---------------------------------------------------------------------------
async def _watcher() -> None:
    global _last_mtime
    while True:
        try:
            if _watch_path is not None:
                mtime = os.path.getmtime(_watch_path)
                if mtime != _last_mtime:
                    _reload()
        except Exception:
            pass
        await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# Lifespan (Starlette ≥ 0.21 / 1.x style — replaces on_startup)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(app):
    reg = load_registry()
    active = reg.get("active")
    if active and active in reg.get("graphs", {}):
        _switch_graph(active)
    asyncio.create_task(_watcher())
    yield


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------
async def api_graph(request: Request) -> JSONResponse:
    return JSONResponse(_snapshot)


async def api_graphs(request: Request) -> JSONResponse:
    reg = load_registry()
    return JSONResponse(
        {
            "graphs": list(reg.get("graphs", {}).keys()),
            "active": _graph_name,
        }
    )


async def api_switch(request: Request) -> JSONResponse:
    name = request.query_params.get("name", "")
    ok = _switch_graph(name)
    return JSONResponse({"ok": ok})


async def api_events(request: Request) -> Response:
    queue: asyncio.Queue = asyncio.Queue(maxsize=16)
    _sse_clients.append(queue)

    async def event_stream():
        # Immediately push the current snapshot on connect.
        if _snapshot:
            yield f"data: {json.dumps(_snapshot)}\n\n".encode()
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {payload}\n\n".encode()
                except asyncio.TimeoutError:
                    # Keep-alive comment
                    yield b":\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            try:
                _sse_clients.remove(queue)
            except ValueError:
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Static directory (client assets added separately)
# ---------------------------------------------------------------------------
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

routes = [
    Route("/api/graph", api_graph),
    Route("/api/graphs", api_graphs),
    Route("/api/switch", api_switch),
    Route("/api/events", api_events),
    Mount("/", StaticFiles(directory=_STATIC_DIR, html=True)),
]

app = Starlette(
    routes=routes,
    lifespan=_lifespan,
)
