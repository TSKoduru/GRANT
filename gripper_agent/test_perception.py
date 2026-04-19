"""
test_perception.py — live sanity check for the full perception pipeline.

Shows the overhead feed with overlays:
  - green circles on detected mat markers
  - yellow cross where FK says the gripper is (projected into the image)
  - red circle on detected object
Readouts in mm for every detection.

If the yellow cross sits on the real gripper in the video, FK + calibration
are consistent. If a colored object at a known location shows the correct
(x, y) readout, the whole pipeline is good.

Keys:
  c — calibrate
  h — sample HSV from center of frame
  s — capture background frame
  b — toggle bg-subtract mode
  q — quit
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from perception import (
    HSVRange, MAT_CORNER_IDS, Perceiver, auto_calibrate_hsv,
)
from robot import Robot


MAT_YAML = Path(__file__).parent / "configs" / "mat_markers.yaml"


def _load_mat():
    if not MAT_YAML.exists():
        sys.exit(f"Missing {MAT_YAML}. Run measure_mat_markers.py first.")
    data = yaml.safe_load(MAT_YAML.read_text())
    return {int(k): tuple(v) for k, v in data["mat_markers_robot_mm"].items()}


def draw_overlay(frame, perceiver, robot, hsv_range, use_bgsub):
    out = frame.copy()
    h, w = out.shape[:2]

    corners, ids, _ = perceiver.detector.detectMarkers(frame)
    if ids is not None:
        ids_flat = ids.flatten().tolist()
        for mid in MAT_CORNER_IDS:
            if mid in ids_flat:
                idx = ids_flat.index(mid)
                c = corners[idx][0].mean(axis=0)
                cv2.circle(out, (int(c[0]), int(c[1])), 8, (0, 255, 0), 2)
                cv2.putText(out, f"mat {mid}", (int(c[0]) + 10, int(c[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    grip_uv = None
    if perceiver.H is not None:
        try:
            gx, gy, gz = robot.get_gripper_tip_mm()
            u, v = perceiver.robot_mm_to_pixel(gx, gy)
            if 0 <= u < w and 0 <= v < h:
                grip_uv = (u, v)
                cv2.drawMarker(out, (int(u), int(v)), (255, 255, 0),
                               cv2.MARKER_CROSS, 24, 2)
                cv2.putText(out, f"grip_FK ({gx:.0f},{gy:.0f},{gz:.0f})",
                            (int(u) + 14, int(v)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        except Exception as e:
            cv2.putText(out, f"FK err: {e}", (10, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    obj = None
    tag = ""
    if use_bgsub:
        obj = perceiver.detect_bgsubtract(frame, gripper_pixel_uv=grip_uv)
        tag = "[bgsub]"
    elif hsv_range is not None:
        obj = perceiver.detect_hsv(frame, hsv_range)
        tag = f"[hsv:{hsv_range.name}]"
    if obj is not None:
        u, v = int(obj.pixel_uv[0]), int(obj.pixel_uv[1])
        cv2.circle(out, (u, v), 12, (0, 0, 255), 2)
        cv2.putText(out, f"{tag} ({obj.x_mm:.0f},{obj.y_mm:.0f})",
                    (u + 14, v + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    status = [
        "CALIBRATED" if perceiver.H is not None else "NOT CALIBRATED",
        "bg=SET" if perceiver.bg_frame_bgr is not None else "bg=none",
        "mode=bgsub" if use_bgsub else ("mode=hsv" if hsv_range else "mode=none"),
    ]
    cv2.putText(out, " | ".join(status), (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(out, "q=quit  c=calibrate  s=set-bg  b=toggle-bgsub  h=sample-hsv",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return out


def main():
    markers = _load_mat()
    robot = Robot()
    perceiver = Perceiver(mat_markers_robot_mm=markers)

    hsv_range = None
    use_bgsub = False

    while True:
        obs = robot.get_observation()
        frame = obs.overhead_bgr
        vis = draw_overlay(frame, perceiver, robot, hsv_range, use_bgsub)
        cv2.imshow("test_perception", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            ok = perceiver.calibrate(frame)
            print("Calibration OK." if ok else "Calibration failed.")
        elif key == ord('s'):
            perceiver.set_background(frame)
            print("Captured background.")
        elif key == ord('b'):
            use_bgsub = not use_bgsub
            print(f"bgsub = {use_bgsub}")
        elif key == ord('h'):
            hsv_range = auto_calibrate_hsv(frame)
            print(f"HSV: low={hsv_range.low} high={hsv_range.high}")

    robot.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()