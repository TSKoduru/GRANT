"""
perception.py — Vision → robot-frame coordinates, via 4-marker homography.

Pipeline:
  1. Tape 4 ArUco markers at the corners of a rectangle on the mat.
     IDs 0, 1, 2, 3 in CLOCKWISE order starting from the top-left AS SEEN
     BY THE OVERHEAD CAMERA.
  2. Jog the arm to each marker's center and record its (x, y) in robot
     base frame. Put those numbers in configs/mat_markers.yaml (see
     measure_mat_markers.py).
  3. calibrate(frame) detects all 4 corner markers in the image and
     computes a 3x3 homography H that maps image pixels → robot-frame mm
     (plus the inverse, for drawing/masking).
  4. Every frame:
       detect_hsv(frame, ...) → Detection(...)
       detect_bgsubtract(...) → Detection(...)
     All detections go through the same pipeline:
       pixel (u, v) → cv2.perspectiveTransform → (x_mm, y_mm)

Gripper position is NOT tracked here — it comes from forward kinematics on
the arm's joint state. See robot.get_gripper_tip_mm(). The robot base is
physically fixed relative to the mat, so FK gives us the gripper's position
directly in the same frame as our mat markers.

Coordinates:
  Robot base frame: +x right, +y forward (away from robot), +z up. Mat is z=0.
  Detections reported here assume z=0 (object sits on the mat).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import cv2.aruco as aruco
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# ArUco IDs
# ─────────────────────────────────────────────────────────────────────────────
# Mat corners, CLOCKWISE as seen by the overhead camera. Order matters.
MAT_CORNER_IDS = (0, 1, 2, 3)     # top-left, top-right, bottom-right, bottom-left
TARGET_ZONE_ID = 5                # optional: if target is a marker rather than a color patch

# Gripper position is NOT tracked via a marker — it comes from forward
# kinematics on the arm's joint state. See robot.get_gripper_tip_mm().

CALIB_FILE = Path(__file__).parent / "calibration.json"


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
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
        return np.array(self.low, dtype=np.uint8), np.array(self.high, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# HSV auto-calibration — sample a color from a reference image crop
# ─────────────────────────────────────────────────────────────────────────────

def auto_calibrate_hsv(
    reference_bgr: np.ndarray,
    crop_xyxy: Optional[tuple[int, int, int, int]] = None,
    tolerance_h: int = 12,
    tolerance_s: int = 60,
    tolerance_v: int = 60,
    name: str = "object",
) -> HSVRange:
    """
    Given a reference image with the object visible, compute an HSV range.
    `crop_xyxy` = (x1, y1, x2, y2) over the object. Defaults to a center crop.
    """
    h, w = reference_bgr.shape[:2]
    if crop_xyxy is None:
        cx, cy = w // 2, h // 2
        r = min(w, h) // 10
        crop_xyxy = (cx - r, cy - r, cx + r, cy + r)

    x1, y1, x2, y2 = crop_xyxy
    patch = reference_bgr[max(0, y1):y2, max(0, x1):x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_med, s_med, v_med = np.median(hsv.reshape(-1, 3), axis=0).astype(int)

    # Hue wraps at 180 in OpenCV. Reds near 0/180 may need a two-range split;
    # keep this simple and fix in config if it bites you.
    low = (
        max(0, h_med - tolerance_h),
        max(0, s_med - tolerance_s),
        max(0, v_med - tolerance_v),
    )
    high = (
        min(179, h_med + tolerance_h),
        min(255, s_med + tolerance_s),
        min(255, v_med + tolerance_v),
    )
    return HSVRange(low=low, high=high, name=name)


# ─────────────────────────────────────────────────────────────────────────────
# Perceiver
# ─────────────────────────────────────────────────────────────────────────────

class Perceiver:
    def __init__(self, mat_markers_robot_mm: dict[int, tuple[float, float]]):
        """
        mat_markers_robot_mm: mapping from ArUco ID → (x_mm, y_mm) in ROBOT frame.
            Must include all four MAT_CORNER_IDS.

            MEASUREMENT PROCEDURE (use the arm itself):
              1. Tape the markers down on the mat. Don't move them after this.
              2. For each marker: jog the gripper tip to hover directly above
                 the marker's center. Record the gripper's (x, y) via forward
                 kinematics. That's the marker's position in robot frame.
              3. Fill in the dict and don't change it until a marker moves.
        """
        missing = set(MAT_CORNER_IDS) - set(mat_markers_robot_mm.keys())
        if missing:
            raise ValueError(f"mat_markers_robot_mm missing IDs: {sorted(missing)}")
        self.mat_markers_robot_mm = dict(mat_markers_robot_mm)

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # H: 3x3 homography, image pixels → robot-frame mm. Populated by calibrate().
        self.H: Optional[np.ndarray] = None

        # Optional empty-scene frame for background-subtraction detection
        self.bg_frame_bgr: Optional[np.ndarray] = None

        self._load_calibration()

    # ─── Calibration ───────────────────────────────────────────────────────

    def calibrate(self, overhead_bgr: np.ndarray) -> bool:
        """
        Detect the 4 mat corner markers and compute pixel → robot-frame mm
        homography. Returns True on success.
        """
        corners, ids, _ = self.detector.detectMarkers(overhead_bgr)
        if ids is None:
            return False
        ids_flat = ids.flatten().tolist()

        src_pts = []   # marker centers in image pixels
        dst_pts = []   # marker centers in robot-frame mm
        for mid in MAT_CORNER_IDS:
            if mid not in ids_flat:
                return False
            idx = ids_flat.index(mid)
            # `corners[idx]` has shape (1, 4, 2); mean over 4 corners = center
            center_uv = corners[idx][0].mean(axis=0)
            src_pts.append(center_uv)
            dst_pts.append(self.mat_markers_robot_mm[mid])

        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        H, _ = cv2.findHomography(src_pts, dst_pts, method=0)
        if H is None:
            return False

        self.H = H
        self._save_calibration()
        return True

    def _save_calibration(self):
        CALIB_FILE.write_text(json.dumps({
            "H": self.H.tolist(),
            "mat_markers_robot_mm": {
                str(k): list(v) for k, v in self.mat_markers_robot_mm.items()
            },
        }, indent=2))

    def _load_calibration(self):
        if CALIB_FILE.exists():
            try:
                data = json.loads(CALIB_FILE.read_text())
                self.H = np.array(data["H"])
            except (json.JSONDecodeError, KeyError):
                pass

    # ─── Pixel → robot-frame mm ────────────────────────────────────────────

    def pixel_to_robot_mm(self, u: float, v: float) -> tuple[float, float]:
        """Apply the calibrated homography to a single pixel."""
        if self.H is None:
            raise RuntimeError("Run calibrate() first")
        pt = np.array([[[u, v]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.H)
        x_mm, y_mm = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
        return x_mm, y_mm

    def robot_mm_to_pixel(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """Inverse of pixel_to_robot_mm. Useful for drawing FK-derived gripper
        position onto the image, and for masking the gripper region in bgsub."""
        if self.H is None:
            raise RuntimeError("Run calibrate() first")
        H_inv = np.linalg.inv(self.H)
        pt = np.array([[[x_mm, y_mm]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H_inv)
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

    # ─── Object detectors (three backends) ─────────────────────────────────

    def detect_hsv(
        self,
        overhead_bgr: np.ndarray,
        hsv_range: HSVRange,
        min_area_px: int = 300,
    ) -> Optional[Detection]:
        """Largest HSV blob in range; centroid → robot-frame mm via homography."""
        hsv = cv2.cvtColor(overhead_bgr, cv2.COLOR_BGR2HSV)
        low, high = hsv_range.as_arrays()
        mask = cv2.inRange(hsv, low, high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return self._blob_to_detection(mask, hsv_range.name, min_area_px)

    def set_background(self, empty_scene_bgr: np.ndarray):
        """Capture an empty-scene frame for background-subtraction detection."""
        self.bg_frame_bgr = empty_scene_bgr.copy()

    def detect_bgsubtract(
        self,
        overhead_bgr: np.ndarray,
        name: str = "object",
        min_area_px: int = 500,
        threshold: int = 30,
        gripper_pixel_uv: Optional[tuple[float, float]] = None,
    ) -> Optional[Detection]:
        """
        Difference against stored background; works on objects of any color.

        gripper_pixel_uv: optional (u, v) of the gripper in image pixels. If
        provided, that region is masked out so the moving gripper isn't
        detected as the object. Caller gets this by running FK → robot_mm
        → inverse homography (see robot.get_gripper_pixel_uv()).
        """
        if self.bg_frame_bgr is None:
            return None
        diff = cv2.absdiff(overhead_bgr, self.bg_frame_bgr)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        # Hide the gripper region from the diff
        if gripper_pixel_uv is not None:
            u, v = int(gripper_pixel_uv[0]), int(gripper_pixel_uv[1])
            cv2.circle(mask, (u, v), 60, 0, -1)

        return self._blob_to_detection(mask, name, min_area_px)

    def detect_sam(self, overhead_bgr: np.ndarray, click_uv: tuple[int, int]) -> Optional[Detection]:
        """
        STRETCH — MobileSAM with a point prompt. Stub.
        When you integrate:
          from mobile_sam import sam_model_registry, SamPredictor
          predictor = SamPredictor(sam_model_registry["vit_t"](checkpoint="mobile_sam.pt"))
          predictor.set_image(cv2.cvtColor(overhead_bgr, cv2.COLOR_BGR2RGB))
          masks, scores, _ = predictor.predict(point_coords=np.array([click_uv]),
                                               point_labels=np.array([1]))
          # pick best mask, compute centroid, apply self.H
        """
        raise NotImplementedError("MobileSAM integration — stretch feature")

    # ─── Target-zone detection ─────────────────────────────────────────────

    def detect_target_by_aruco(self, overhead_bgr: np.ndarray) -> Optional[Detection]:
        """If the target zone has its own ArUco marker (TARGET_ZONE_ID)."""
        if self.H is None:
            return None
        corners, ids, _ = self.detector.detectMarkers(overhead_bgr)
        if ids is None or TARGET_ZONE_ID not in ids.flatten():
            return None
        idx = int(np.where(ids.flatten() == TARGET_ZONE_ID)[0][0])
        u, v = corners[idx][0].mean(axis=0)
        x_mm, y_mm = self.pixel_to_robot_mm(float(u), float(v))
        return Detection(
            name="target",
            x_mm=x_mm, y_mm=y_mm, z_mm=0.0,
            pixel_uv=(float(u), float(v)),
        )

    # ─── Internals ─────────────────────────────────────────────────────────

    def _blob_to_detection(
        self, mask: np.ndarray, name: str, min_area_px: int,
    ) -> Optional[Detection]:
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
            name=name,
            x_mm=x_mm, y_mm=y_mm, z_mm=0.0,
            confidence=float(min(1.0, area / 5000.0)),
            pixel_uv=(float(u), float(v)),
        )