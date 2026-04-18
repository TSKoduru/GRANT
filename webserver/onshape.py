"""
Onshape completion webhook.

Production path: the AI PC POSTs the finished mesh to the ESP32-S3,
which then talks to Onshape over WiFi using credentials that live in
the ESP32 firmware. This keeps Onshape API keys off the AI PC.

For development we can also POST directly to Onshape from here.

Configuration (environment variables):
    ONSHAPE_WEBHOOK_URL   — ESP32 endpoint, e.g. http://esp32.local/onshape
                             (if set, we use the bridge)
    ONSHAPE_API_URL       — direct Onshape REST endpoint
    ONSHAPE_API_KEY       — Onshape API access key (for direct mode)
    ONSHAPE_API_SECRET    — Onshape API secret key (for direct mode)
    ONSHAPE_DOCUMENT_ID   — target document id (for direct mode)

Returns the Onshape document URL on success. We never raise out of this
function — a scan completing is more important than the webhook landing.
"""
from __future__ import annotations

import os
import pathlib
from typing import Optional


def notify_completion(
    scan_id: str,
    mesh_path: pathlib.Path,
) -> Optional[str]:
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError:
        return None

    # Prefer the ESP32 bridge if configured
    bridge = os.environ.get("ONSHAPE_WEBHOOK_URL")
    if bridge:
        return _post_to_bridge(bridge, scan_id, mesh_path, requests)

    # Direct mode — stubbed. Real Onshape integration requires:
    #   1. Upload .ply/.glb to the document via /documents/{did}/w/{wid}/blobelements
    #   2. Import geometry into a part studio
    #   3. Return the document URL
    # See: https://onshape-public.github.io/docs/
    return _post_direct_stub(scan_id, mesh_path, requests)


def _post_to_bridge(url: str, scan_id: str, mesh_path: pathlib.Path, requests) -> Optional[str]:
    try:
        with open(mesh_path, "rb") as f:
            r = requests.post(
                url,
                params={"scan_id": scan_id},
                files={"mesh": (mesh_path.name, f, "application/octet-stream")},
                timeout=30,
            )
        if not r.ok:
            return None
        return r.json().get("document_url")
    except Exception:
        return None


def _post_direct_stub(scan_id: str, mesh_path: pathlib.Path, requests) -> Optional[str]:
    # TODO: implement when direct-mode credentials are available.
    # Requires HMAC request signing with ONSHAPE_API_KEY/SECRET.
    return None
