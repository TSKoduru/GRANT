"""
FastAPI app served from the AI PC.

Run with:
    uvicorn GRANT.webserver.server:app --host 0.0.0.0 --port 8000

Dashboard lives at `/`, REST endpoints under `/scan/*`.

The dashboard's HTTP traffic can hit this server directly, or be proxied
through the ESP32-S3 which hosts the WiFi AP the user connects to.
"""
from __future__ import annotations

import pathlib

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .scan_session import ScanSession

_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent
DASHBOARD_DIR = _PKG_ROOT / "dashboard"

app = FastAPI(title="GRANT scan controller")

# Static assets for the dashboard (JS, CSS, any images)
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

# One session per server process — scans are serialized
session = ScanSession()


@app.get("/")
def index() -> FileResponse:
    path = DASHBOARD_DIR / "index.html"
    if not path.exists():
        raise HTTPException(404, f"Dashboard not found at {path}")
    return FileResponse(path)


@app.post("/scan/start")
def start_scan() -> dict:
    if session.is_running():
        raise HTTPException(409, "A scan is already running")
    scan_id = session.start()
    return {"scan_id": scan_id}


@app.get("/scan/status")
def status() -> dict:
    return session.snapshot()


@app.get("/scan/heatmap")
def heatmap() -> Response:
    png = session.heatmap_png()
    if png is None:
        raise HTTPException(404, "No heatmap yet (Pillow not installed, or no frames captured)")
    return Response(content=png, media_type="image/png")


@app.get("/scan/mesh")
def mesh() -> FileResponse:
    mp = session.mesh_path()
    if mp is None or not mp.exists():
        raise HTTPException(404, "No mesh available yet")
    return FileResponse(mp, filename=mp.name, media_type="application/octet-stream")
