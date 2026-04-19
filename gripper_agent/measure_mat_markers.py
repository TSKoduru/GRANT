"""
measure_mat_markers.py — interactive mat-marker measurement.

For each of IDs 0, 1, 2, 3:
  1. Jog the arm so the gripper tip hovers directly above the marker's center.
  2. Press ENTER.
  3. We read joint state, run forward kinematics, record (x, y) mm.

You need a way to jog the arm (lerobot teleop in another terminal, or
drive.py's REPL). This script only READS joint state; it never commands.

Output: configs/mat_markers.yaml
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml

from robot import Robot


OUTPUT_YAML = Path(__file__).parent / "configs" / "mat_markers.yaml"

MARKER_IDS = [0, 1, 2, 3]
MARKER_LABELS = {
    0: "TOP-LEFT     (closest-left corner, from camera's view)",
    1: "TOP-RIGHT    (closest-right corner, from camera's view)",
    2: "BOTTOM-RIGHT (farthest-right corner)",
    3: "BOTTOM-LEFT  (farthest-left corner)",
}


def main():
    print("Connecting to arm...")
    robot = Robot()

    print()
    print("You need to jog the arm in a separate terminal or via drive.py.")
    print("This script only reads joint state.")
    print()

    positions: dict[int, tuple[float, float]] = {}

    for mid in MARKER_IDS:
        print(f"─── MARKER {mid}: {MARKER_LABELS[mid]} ───")
        print("Jog the gripper tip so it hovers directly above this marker's center.")
        input("Press ENTER when positioned... ")

        joints = robot.get_joint_positions_deg()
        try:
            x_mm, y_mm, z_mm = robot.get_gripper_tip_mm()
        except Exception as e:
            print(f"FK error: {e}")
            robot.close()
            sys.exit(1)

        print(f"  Joints: {joints}")
        print(f"  Gripper tip: ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm")
        positions[mid] = (round(x_mm, 1), round(y_mm, 1))
        print()

    # Rectangle sanity check
    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    diag1 = dist(positions[0], positions[2])
    diag2 = dist(positions[1], positions[3])
    print("Sanity check:")
    print(f"  Diagonal 0→2: {diag1:.1f}mm")
    print(f"  Diagonal 1→3: {diag2:.1f}mm")
    if abs(diag1 - diag2) / max(diag1, diag2, 1e-3) > 0.15:
        print("  WARNING: diagonals differ by >15%. Double-check measurements.")

    OUTPUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_YAML.write_text(yaml.safe_dump({
        "mat_markers_robot_mm": {int(k): list(v) for k, v in positions.items()},
    }, sort_keys=True))
    print(f"\nWrote {OUTPUT_YAML}")

    robot.close()


if __name__ == "__main__":
    main()