"""
measure_mat_markers.py — interactive mat-marker measurement.

For each of the 4 mat-corner markers (IDs 0, 1, 2, 3), you:
  1. Jog the arm manually so the gripper tip is centered above the marker.
  2. Press ENTER in this script.
  3. The script reads current joint positions, runs forward kinematics,
     and records (x_mm, y_mm) of the gripper tip.

Output: configs/mat_markers.yaml

Before running:
  - Update URDF_PATH below to your SO-101 URDF file.
  - Mark your 4 ArUco tags on the mat. IDs 0, 1, 2, 3 in CLOCKWISE order
    as seen by the OVERHEAD CAMERA, starting top-left.

Jogging:
  If your SO-101 supports torque-off / freedrive mode, use it — physically
  move the arm by hand. Otherwise, you'll need a separate jogging script
  (e.g., lerobot's teleop) in another terminal. This script only READS joint
  state; it doesn't command the arm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml   # pip install pyyaml

from robot import Robot
from kinematics import Kinematics


# ── EDIT THESE ─────────────────────────────────────────────────────────────
URDF_PATH = "path/to/so101.urdf"   # TODO: fill in your SO-101 URDF path
OUTPUT_YAML = Path(__file__).parent / "configs" / "mat_markers.yaml"
# ───────────────────────────────────────────────────────────────────────────

MARKER_IDS = [0, 1, 2, 3]
MARKER_LABELS = {
    0: "TOP-LEFT    (closest-left  corner of mat, from camera's view)",
    1: "TOP-RIGHT   (closest-right corner of mat, from camera's view)",
    2: "BOTTOM-RIGHT (farthest-right corner)",
    3: "BOTTOM-LEFT  (farthest-left  corner)",
}


def main():
    print("Initializing robot and kinematics...")
    robot = Robot()
    kin = Kinematics(URDF_PATH)

    # Run this once when debugging URDF loading. Uncomment to see chain structure.
    # kin.describe()

    positions: dict[int, tuple[float, float]] = {}

    print()
    print("Make sure the arm is in a mode where you can jog it freely")
    print("(freedrive, or use a separate teleop terminal).")
    print()

    for mid in MARKER_IDS:
        print(f"─── MARKER ID {mid}: {MARKER_LABELS[mid]} ───")
        print("Jog the GRIPPER TIP so it hovers directly above the center of this marker.")
        print("Tip should be as low as possible without touching the marker.")
        input("Press ENTER when positioned... ")

        joints = robot.get_joint_positions_deg()
        print(f"  Joint state: {joints}")
        try:
            x_mm, y_mm, z_mm = kin.forward(joints)
        except Exception as e:
            print(f"  FK ERROR: {e}")
            print("  Fix URDF_PATH or kinematics setup, then rerun.")
            sys.exit(1)

        print(f"  Gripper tip in robot frame: x={x_mm:.1f}mm, y={y_mm:.1f}mm, z={z_mm:.1f}mm")
        print(f"  Recording (x, y) = ({x_mm:.1f}, {y_mm:.1f}) for marker {mid}")
        positions[mid] = (round(x_mm, 1), round(y_mm, 1))
        print()

    # ── Sanity check: the 4 points should form a roughly sensible rectangle ──
    print("Sanity check:")
    for mid in MARKER_IDS:
        print(f"  marker {mid}: {positions[mid]}")

    # Compute two diagonals — they should be similar length if it's a rectangle
    import math
    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    diag1 = dist(positions[0], positions[2])   # TL → BR
    diag2 = dist(positions[1], positions[3])   # TR → BL
    print(f"  diagonal 0→2: {diag1:.1f}mm")
    print(f"  diagonal 1→3: {diag2:.1f}mm")
    if abs(diag1 - diag2) / max(diag1, diag2) > 0.15:
        print("  WARNING: diagonals differ by >15%. Your markers may not be on a rectangle,")
        print("  or one measurement is off. Homography will still work, but double-check.")

    # ── Write YAML ──
    OUTPUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_YAML.write_text(yaml.safe_dump({
        "mat_markers_robot_mm": {int(mid): list(xy) for mid, xy in positions.items()},
    }, sort_keys=True))
    print()
    print(f"Wrote {OUTPUT_YAML}")
    print("You can re-run this anytime a marker physically moves.")

    robot.close()


if __name__ == "__main__":
    main()