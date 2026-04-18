"""
test_perception.py — end-to-end perception sanity check.

What this does:
  1. Loads mat-marker positions from configs/mat_markers.yaml.
  2. Opens the overhead camera, calibrates from the 4 mat corner markers.
  3. Shows a live window with overlays:
       - Green circles at each detected mat marker (sanity check: all 4 found)
       - Yellow cross where FORWARD KINEMATICS says the gripper is, projected
         into the image via the inverse homography. If FK + mat calibration
         are both correct, this cross sits on the actual gripper in the video.
       - Red circle on detected object (HSV or bg-subtract), labeled with (x, y).
  4. Press 's' to capture a background frame (for bg-subtract).
  5. Press 'b' to toggle bg-subtract detection instead of HSV.
  6. Press 'c' to re-run calibration.
  7. Press 'h' to sample HSV from the center of the frame (put object there first).
  8. Press 'q' to quit.

Three sanity checks you should run:
  A. After pressing 'c', move the arm slowly. The yellow cross should track
     the gripper tip. If it drifts off, your URDF or mat markers are wrong.
  B. Put a colored object at a KNOWN location (e.g., on top of mat marker 0).
     Press 'h' with it centered, then move it. The red circle's (x, y) should
     match the object's real robot-frame coordinates within ~5mm.
  C. Numbers reported by the yellow cross (FK) and a colored object placed
     directly under the gripper should agree within a few mm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from robot import Robot
from perception import (
    Perceiver,
    HSVRange,
    auto_calibrate_hsv,
    MAT_CORNER_IDS,
)


CONFIG_PATH = Path(__file__).parent / "configs" / "mat_markers.yaml"


def load_mat_markers() -> dict[int, tuple[float, float]]:
    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found.")
        print("Run measure_mat_markers.py first.")
        sys.exit(1)
    data = yaml.safe_load(CONFIG_PATH.read_text())
    return {int(k): tuple(v) for k, v in data["mat_markers_robot_mm"].items()}


def draw_overlay(
    frame: np.ndarray,
    perceiver: Perceiver,
    robot: Robot,
    hsv_range: HSVRange | None,
    use_bgsub: bool,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # ── Draw detected mat markers in green ──
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

    # ── Gripper via FK → project back to image ──
    # If FK and the mat-marker calibration agree, this circle lands on the
    # real gripper in the frame. If they disagree, you've got a calibration bug.
    gripper_pixel_uv: tuple[float, float] | None = None
    if perceiver.H is not None:
        try:
            gx, gy, gz = robot.get_gripper_tip_mm()
            u, v = perceiver.robot_mm_to_pixel(gx, gy)
            if 0 <= u < w and 0 <= v < h:
                gripper_pixel_uv = (u, v)
                cv2.drawMarker(out, (int(u), int(v)), (255, 255, 0),
                               cv2.MARKER_CROSS, 24, 2)
                cv2.putText(out, f"grip_FK ({gx:.0f}, {gy:.0f}, {gz:.0f})mm",
                            (int(u) + 14, int(v)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        except Exception as e:
            cv2.putText(out, f"FK error: {e}", (10, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # ── Object detection ──
    if use_bgsub:
        obj = perceiver.detect_bgsubtract(
            frame, name="obj", gripper_pixel_uv=gripper_pixel_uv,
        )
        label_tag = "[bgsub]"
    elif hsv_range is not None:
        obj = perceiver.detect_hsv(frame, hsv_range)
        label_tag = f"[hsv:{hsv_range.name}]"
    else:
        obj = None
        label_tag = ""
    if obj is not None:
        u, v = int(obj.pixel_uv[0]), int(obj.pixel_uv[1])
        cv2.circle(out, (u, v), 12, (0, 0, 255), 2)
        cv2.putText(out, f"{label_tag} ({obj.x_mm:.0f}, {obj.y_mm:.0f})mm",
                    (u + 14, v + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # ── Footer ──
    status = []
    status.append("CALIBRATED" if perceiver.H is not None else "NOT CALIBRATED")
    status.append("bg=SET" if perceiver.bg_frame_bgr is not None else "bg=none")
    status.append("mode=bgsub" if use_bgsub else ("mode=hsv" if hsv_range else "mode=none"))
    footer = " | ".join(status)
    cv2.putText(out, footer, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2)
    cv2.putText(out, "q=quit  c=calibrate  s=set-bg  b=toggle-bgsub  h=sample-hsv",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return out


def main():
    markers = load_mat_markers()
    print(f"Loaded {len(markers)} mat markers:")
    for mid, xy in sorted(markers.items()):
        print(f"  id {mid}: ({xy[0]}, {xy[1]}) mm")

    robot = Robot()
    perceiver = Perceiver(mat_markers_robot_mm=markers)

    hsv_range: HSVRange | None = None
    use_bgsub = False

    print("\nPress 'c' in the window to calibrate, then do sanity checks.\n")

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
            if ok:
                print("Calibration OK. Homography updated.")
            else:
                print("Calibration FAILED — couldn't find all 4 mat markers. "
                      "Check lighting and marker visibility.")

        elif key == ord('s'):
            perceiver.set_background(frame)
            print("Captured background frame for bg-subtract.")

        elif key == ord('b'):
            use_bgsub = not use_bgsub
            print(f"bgsub mode = {use_bgsub}")

        elif key == ord('h'):
            # Sample HSV from the center of the frame.
            # To target a specific object, put it at the center before pressing 'h'.
            hsv_range = auto_calibrate_hsv(frame, name="object")
            print(f"Sampled HSV range: low={hsv_range.low}, high={hsv_range.high}")

    robot.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()