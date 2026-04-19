"""
kinematics.py — forward and inverse kinematics for the SO-101 via ikpy.

The SO-101 URDF we load has 7 links:
  [0] Base link                 fixed
  [1] shoulder_pan              revolute  ← joint 1
  [2] shoulder_lift             revolute  ← joint 2
  [3] elbow_flex                revolute  ← joint 3
  [4] wrist_flex                revolute  ← joint 4
  [5] wrist_roll                revolute  ← joint 5
  [6] gripper_frame_joint       fixed

The gripper servo (joint 6 in lerobot's obs) is NOT part of the kinematic
chain — it opens/closes a linkage without moving the end-effector. Every
function here works in degrees; ikpy internals use radians and meters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ikpy.chain import Chain


ACTIVE_JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# URDF-specific — validated against the SO-101 URDF from the lerobot repo.
ACTIVE_LINKS_MASK = [False, True, True, True, True, True, False]


class Kinematics:
    def __init__(self, urdf_path: str | Path):
        self.chain = Chain.from_urdf_file(
            str(urdf_path), active_links_mask=ACTIVE_LINKS_MASK
        )
        if len(self.chain.links) != len(ACTIVE_LINKS_MASK):
            raise RuntimeError(
                f"URDF has {len(self.chain.links)} links; mask expects "
                f"{len(ACTIVE_LINKS_MASK)}. Check the URDF."
            )
        self.num_links = len(self.chain.links)
        self._last_solution_rad = np.zeros(self.num_links)

    # ── Forward kinematics ─────────────────────────────────────────────────

    def forward(self, joint_degs: dict[str, float]) -> tuple[float, float, float]:
        """Gripper-frame origin in robot base frame, mm."""
        full = self._dict_to_full_vector(joint_degs)
        pose = self.chain.forward_kinematics(full)
        xyz_m = pose[:3, 3]
        return float(xyz_m[0] * 1000), float(xyz_m[1] * 1000), float(xyz_m[2] * 1000)

    def forward_matrix(self, joint_degs: dict[str, float]) -> np.ndarray:
        """Full 4x4 pose (meters) for when you need orientation too."""
        return self.chain.forward_kinematics(self._dict_to_full_vector(joint_degs))

    # ── Inverse kinematics ─────────────────────────────────────────────────

    def inverse(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        seed_joint_degs: Optional[dict[str, float]] = None,
        target_z_vector: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """
        Position IK (with optional Z-axis pointing constraint for top-down grasps).
        target_z_vector: unit vector the end-effector's +Z should align with.
        """
        target_xyz_m = np.array([x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0])
        initial = (
            self._dict_to_full_vector(seed_joint_degs)
            if seed_joint_degs is not None
            else self._last_solution_rad
        )

        kwargs = {"target_position": target_xyz_m, "initial_position": initial}
        if target_z_vector is not None:
            kwargs["target_orientation"] = target_z_vector
            kwargs["orientation_mode"] = "Z"

        solution_rad = self.chain.inverse_kinematics(**kwargs)
        self._last_solution_rad = solution_rad
        return self._full_vector_to_dict(solution_rad)

    # ── Internals ──────────────────────────────────────────────────────────

    def _dict_to_full_vector(self, joint_degs: dict[str, float]) -> np.ndarray:
        actuated_rads = np.deg2rad([joint_degs.get(n, 0.0) for n in ACTIVE_JOINT_ORDER])
        full = np.zeros(self.num_links)
        active_indices = [i for i, a in enumerate(ACTIVE_LINKS_MASK) if a]
        for link_idx, rad in zip(active_indices, actuated_rads):
            full[link_idx] = rad
        return full

    def _full_vector_to_dict(self, full_vec: np.ndarray) -> dict[str, float]:
        active_indices = [i for i, a in enumerate(ACTIVE_LINKS_MASK) if a]
        return {
            name: float(np.rad2deg(full_vec[link_idx]))
            for name, link_idx in zip(ACTIVE_JOINT_ORDER, active_indices)
        }

    def describe(self):
        print(f"Chain with {self.num_links} links:")
        for i, link in enumerate(self.chain.links):
            tag = "ACTIVE" if ACTIVE_LINKS_MASK[i] else "fixed "
            print(f"  [{i}] {tag} {link.name} (type={link.joint_type})")