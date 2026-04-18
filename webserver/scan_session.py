"""
Thread-safe wrapper around ScanOrchestrator.

The FastAPI server owns one `ScanSession` instance; starting a scan
kicks off a background thread that runs `ScanOrchestrator.run_full_scan`
while the main thread keeps answering status polls.
"""
from __future__ import annotations

import io
import pathlib
import threading
import uuid
from dataclasses import asdict, dataclass
from typing import Optional

from ..kv260.coverage_heatmap import CoverageHeatmap
from ..scan_types import CapturedView


@dataclass
class ScanSnapshot:
    scan_id: Optional[str] = None
    phase: str = "idle"             # idle | initializing | orient-1 i/n | flipping | orient-2 i/n | aligning | fusing | complete | error
    frames_captured: int = 0
    total_frames_expected: int = 0
    coverage_fraction: float = 0.0
    alignment_fitness: Optional[float] = None
    mesh_path: Optional[str] = None
    onshape_url: Optional[str] = None
    error: Optional[str] = None


class ScanSession:

    OUT_DIR = pathlib.Path("scan_output")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._snapshot = ScanSnapshot()
        self._heatmap = CoverageHeatmap()

    # ── Public API (called from the web handlers) ────────────────────

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> str:
        scan_id = uuid.uuid4().hex[:8]
        with self._lock:
            self._snapshot = ScanSnapshot(
                scan_id=scan_id,
                phase="initializing",
                # 2 orientations × (n_x + n_y) steps — matches ScanOrchestrator defaults
                total_frames_expected=2 * (12 + 12),
            )
            self._heatmap.reset()
        self._thread = threading.Thread(
            target=self._run, args=(scan_id,), daemon=True, name=f"scan-{scan_id}"
        )
        self._thread.start()
        return scan_id

    def snapshot(self) -> dict:
        with self._lock:
            return asdict(self._snapshot)

    def heatmap_png(self) -> Optional[bytes]:
        img = self._heatmap.get_heatmap_image()
        if img.size == 0:
            return None
        try:
            from PIL import Image
        except ImportError:
            return None
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        return buf.getvalue()

    def mesh_path(self) -> Optional[pathlib.Path]:
        with self._lock:
            p = self._snapshot.mesh_path
        return pathlib.Path(p) if p else None

    # ── Internal ──────────────────────────────────────────────────────

    def _on_view(self, view: CapturedView, phase: str, i: int, n: int) -> None:
        """Hook passed to ScanOrchestrator; fires per captured frame."""
        self._heatmap.update(view)
        with self._lock:
            self._snapshot.phase = f"{phase} {i + 1}/{n}"
            self._snapshot.frames_captured += 1
            self._snapshot.coverage_fraction = self._heatmap.get_fraction_covered()

    def _set_phase(self, phase: str) -> None:
        with self._lock:
            self._snapshot.phase = phase

    def _run(self, scan_id: str) -> None:
        # Imported here (not at module top) so the web server can be imported
        # and the dashboard developed on a laptop without the hardware drivers.
        try:
            import open3d as o3d

            from ..interfaces.robotic_arm import RoboticArm
            from ..interfaces.vision import VisionSystem
            from ..orchestrator import ScanOrchestrator
            from ..registration import Registration
            from .onshape import notify_completion
        except Exception as e:
            with self._lock:
                self._snapshot.phase = "error"
                self._snapshot.error = f"import failure: {e}"
            return

        try:
            self.OUT_DIR.mkdir(parents=True, exist_ok=True)

            arm = RoboticArm()
            vision = VisionSystem()
            registration = Registration()

            orch = ScanOrchestrator(
                arm=arm, vision=vision, registration=registration,
                on_view_captured=self._on_view,
            )

            # Seed the heatmap with an early centroid so it's not empty
            # during the first few captures. `run_full_scan` does its own
            # detect_object() internally; this is just a preview read.
            try:
                init_frame = vision.capture_rgbd()
                self._heatmap.set_object(vision.detect_object(init_frame))
            except Exception:
                pass  # not fatal — the orchestrator will still run

            result = orch.run_full_scan()

            mesh_path = self.OUT_DIR / f"scan_{scan_id}.ply"
            o3d.io.write_triangle_mesh(str(mesh_path), result.mesh)

            onshape_url = notify_completion(scan_id=scan_id, mesh_path=mesh_path)

            with self._lock:
                self._snapshot.phase = "complete"
                self._snapshot.mesh_path = str(mesh_path)
                self._snapshot.onshape_url = onshape_url
                self._snapshot.alignment_fitness = result.alignment_fitness

        except Exception as e:
            with self._lock:
                self._snapshot.phase = "error"
                self._snapshot.error = str(e)
