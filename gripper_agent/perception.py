"""
perception.py — pixel → robot-frame mm via 4-marker homography.

Gripper tracking uses forward kinematics (see robot.get_gripper_tip_mm);
this module handles OBJECT detection only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import cv2.aruco as aruco
import numpy as np


MAT_CORNER_IDS = (0, 1, 2, 3)   # clockwise from top-left as seen by overhead cam
TARGET_ZONE_ID = 5

CALIB_FILE = Path(__file__).parent / "calibration.json"


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    name: str
    x_mm: float
    y_mm: float
    z_mm: float = 0.0
    confidence: float = 1.0
    pixel_uv: tuple[float, float] = (0.0, 0.0)


@dataclass
class HSVRange:
    low: tuple[int, int, int]
    high: tuple[int, int, int]
    name: str = "object"

    def as_arrays(self):
        return np.array(self.low, np.uint8), np.array(self.high, np.uint8)


def auto_calibrate_hsv(
    reference_bgr: np.ndarray,
    crop_xyxy: Optional[tuple[int, int, int, int]] = None,
    tol_h: int = 12, tol_s: int = 60, tol_v: int = 60,
    name: str = "object",
) -> HSVRange:
    """Sample HSV from a reference image region. Defaults to a center crop."""
    h, w = reference_bgr.shape[:2]
    if crop_xyxy is None:
        cx, cy = w // 2, h // 2
        r = min(w, h) // 10
        crop_xyxy = (cx - r, cy - r, cx + r, cy + r)
    x1, y1, x2, y2 = crop_xyxy
    patch = reference_bgr[max(0, y1):y2, max(0, x1):x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_med, s_med, v_med = np.median(hsv.reshape(-1, 3), axis=0).astype(int)
    return HSVRange(
        low=(max(0, h_med - tol_h), max(0, s_med - tol_s), max(0, v_med - tol_v)),
        high=(min(179, h_med + tol_h), min(255, s_med + tol_s), min(255, v_med + tol_v)),
        name=name,
    )


# ─────────────────────────────────────────────────────────────────────────────

class Perceiver:
    def __init__(self, mat_markers_robot_mm: dict[int, tuple[float, float]]):
        missing = set(MAT_CORNER_IDS) - set(mat_markers_robot_mm.keys())
        if missing:
            raise ValueError(f"mat_markers_robot_mm missing IDs: {sorted(missing)}")
        self.mat_markers_robot_mm = dict(mat_markers_robot_mm)

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.H: Optional[np.ndarray] = None
        self.bg_frame_bgr: Optional[np.ndarray] = None
        self._load_calibration()

    # ── Calibration ────────────────────────────────────────────────────────

    def calibrate(self, overhead_bgr: np.ndarray) -> bool:
        corners, ids, _ = self.detector.detectMarkers(overhead_bgr)
        if ids is None:
            return False
        ids_flat = ids.flatten().tolist()

        src_pts, dst_pts = [], []
        for mid in MAT_CORNER_IDS:
            if mid not in ids_flat:
                return False
            idx = ids_flat.index(mid)
            src_pts.append(corners[idx][0].mean(axis=0))
            dst_pts.append(self.mat_markers_robot_mm[mid])

        H, _ = cv2.findHomography(
            np.array(src_pts, np.float32),
            np.array(dst_pts, np.float32),
            method=0,
        )
        if H is None:
            return False
        self.H = H
        self._save_calibration()
        return True

    def _save_calibration(self):
        CALIB_FILE.write_text(json.dumps({
            "H": self.H.tolist(),
            "mat_markers_robot_mm": {str(k): list(v)
                                     for k, v in self.mat_markers_robot_mm.items()},
        }, indent=2))

    def _load_calibration(self):
        if CALIB_FILE.exists():
            try:
                self.H = np.array(json.loads(CALIB_FILE.read_text())["H"])
            except (json.JSONDecodeError, KeyError):
                pass

    # ── Homography helpers ─────────────────────────────────────────────────

    def pixel_to_robot_mm(self, u: float, v: float) -> tuple[float, float]:
        if self.H is None:
            raise RuntimeError("Calibrate first")
        pt = np.array([[[u, v]]], np.float32)
        m = cv2.perspectiveTransform(pt, self.H)
        return float(m[0, 0, 0]), float(m[0, 0, 1])

    def robot_mm_to_pixel(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        if self.H is None:
            raise RuntimeError("Calibrate first")
        pt = np.array([[[x_mm, y_mm]]], np.float32)
        m = cv2.perspectiveTransform(pt, np.linalg.inv(self.H))
        return float(m[0, 0, 0]), float(m[0, 0, 1])

    # ── Detectors ──────────────────────────────────────────────────────────

    def detect_hsv(
        self, overhead_bgr: np.ndarray, hsv_range: HSVRange, min_area_px: int = 300,
    ) -> Optional[Detection]:
        hsv = cv2.cvtColor(overhead_bgr, cv2.COLOR_BGR2HSV)
        low, high = hsv_range.as_arrays()
        mask = cv2.inRange(hsv, low, high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return self._blob_to_detection(mask, hsv_range.name, min_area_px)

    def set_background(self, empty_scene_bgr: np.ndarray):
        self.bg_frame_bgr = empty_scene_bgr.copy()

    def detect_bgsubtract(
        self, overhead_bgr: np.ndarray, name: str = "object",
        min_area_px: int = 500, threshold: int = 30,
        gripper_pixel_uv: Optional[tuple[float, float]] = None,
    ) -> Optional[Detection]:
        if self.bg_frame_bgr is None:
            return None
        diff = cv2.absdiff(overhead_bgr, self.bg_frame_bgr)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        if gripper_pixel_uv is not None:
            u, v = int(gripper_pixel_uv[0]), int(gripper_pixel_uv[1])
            cv2.circle(mask, (u, v), 60, 0, -1)
        return self._blob_to_detection(mask, name, min_area_px)

    def detect_target_by_aruco(self, overhead_bgr: np.ndarray) -> Optional[Detection]:
        if self.H is None:
            return None
        corners, ids, _ = self.detector.detectMarkers(overhead_bgr)
        if ids is None or TARGET_ZONE_ID not in ids.flatten():
            return None
        idx = int(np.where(ids.flatten() == TARGET_ZONE_ID)[0][0])
        u, v = corners[idx][0].mean(axis=0)
        x_mm, y_mm = self.pixel_to_robot_mm(float(u), float(v))
        return Detection("target", x_mm, y_mm, 0.0, 1.0, (float(u), float(v)))

    # ── Internals ──────────────────────────────────────────────────────────

    def _blob_to_detection(self, mask, name, min_area_px) -> Optional[Detection]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < min_area_px:
            return None
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        u, v = M["m10"] / M["m00"], M["m01"] / M["m00"]
        x_mm, y_mm = self.pixel_to_robot_mm(u, v)
        return Detection(
            name=name, x_mm=x_mm, y_mm=y_mm, z_mm=0.0,
            confidence=float(min(1.0, area / 5000.0)),
            pixel_uv=(float(u), float(v)),
        )