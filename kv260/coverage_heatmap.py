"""
Live coverage heatmap.

We bin camera viewing directions (vector from the object centroid to the
camera position) into a (lat, lon) grid on the unit sphere and splat a
Gaussian kernel at each new view's cell. The `get_heatmap_image()` call
returns a colored 2D image the web dashboard displays as the scan runs.

This doesn't drive the scan — the trajectory is fixed. It's purely a
visualization aid + a sponsor-prize deliverable on the KV260.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..scan_types import CapturedView, ObjectState, Pose6D


def _pose_xyz(pose: Pose6D) -> np.ndarray:
    return np.array([pose.x, pose.y, pose.z], dtype=np.float64)


class CoverageHeatmap:
    N_LAT = 36                         # rows — 5° per row
    N_LON = 72                         # cols — 5° per col
    SPLAT_RADIUS_CELLS = 3             # how wide the Gaussian footprint is
    COVERED_THRESHOLD = 0.5            # cell is "covered" at this splat intensity

    def __init__(self) -> None:
        self.heatmap = np.zeros((self.N_LAT, self.N_LON), dtype=np.float32)
        self._object_centroid: Optional[np.ndarray] = None

    def set_object(self, state: ObjectState) -> None:
        self._object_centroid = _pose_xyz(state.centroid)

    def reset(self) -> None:
        self.heatmap.fill(0.0)
        self._object_centroid = None

    def update(self, view: CapturedView) -> None:
        if self._object_centroid is None:
            return

        direction = _pose_xyz(view.camera_pose) - self._object_centroid
        r = float(np.linalg.norm(direction))
        if r < 1e-6:
            return
        direction /= r

        # Spherical cell: lat ∈ [0, π], lon ∈ [0, 2π)
        lat = float(np.arccos(np.clip(direction[2], -1.0, 1.0)))
        lon = float((np.arctan2(direction[1], direction[0]) + 2 * np.pi) % (2 * np.pi))
        i = min(int(lat / np.pi * self.N_LAT), self.N_LAT - 1)
        j = min(int(lon / (2 * np.pi) * self.N_LON), self.N_LON - 1)

        # Splat a Gaussian
        r2 = self.SPLAT_RADIUS_CELLS
        for di in range(-r2, r2 + 1):
            for dj in range(-r2, r2 + 1):
                ii = max(0, min(self.N_LAT - 1, i + di))
                jj = (j + dj) % self.N_LON
                self.heatmap[ii, jj] += float(np.exp(-(di * di + dj * dj) / (2.0 * r2)))

    def get_fraction_covered(self) -> float:
        return float((self.heatmap >= self.COVERED_THRESHOLD).mean())

    def get_heatmap_image(self) -> np.ndarray:
        """Return (N_LAT, N_LON, 3) uint8 colored blue → green → yellow → red."""
        peak = float(self.heatmap.max())
        h = self.heatmap / max(peak, 1e-6)
        # Simple colormap, no matplotlib dep
        r = np.clip(h * 2.0 - 0.5, 0, 1)
        g = np.clip(1.0 - np.abs(h - 0.5) * 2.0, 0, 1)
        b = np.clip(1.0 - h * 2.0, 0, 1)
        img = np.stack([r, g, b], axis=-1)
        return (img * 255).astype(np.uint8)
