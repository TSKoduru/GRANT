"""
kinematics.py — Forward and inverse kinematics for the SO-101 via ikpy.

Usage:
    kin = Kinematics(urdf_path="path/to/so101.urdf")
    xyz_mm = kin.forward(joint_degs)           # (x, y, z) in mm, robot base frame
    joint_degs = kin.inverse(x_mm, y_mm, z_mm) # {joint_name: degrees}

Notes on ikpy:
  - ikpy works in meters and radians. We convert at the boundary.
  - ikpy needs to know which links are "active" (actuated). Pass active_links_mask
    matching your URDF's link ordering, or let ikpy infer from joint types.
  - For IK, initial_position helps convergence — we seed with current joints.

If ikpy's automatic base-link detection fails, you may need to pass
base_elements=[<name of first link in your URDF>].
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from ikpy.chain import Chain
except ImportError as e:
    raise ImportError(
        "ikpy not installed. `pip install ikpy`"
    ) from e


# Order must match robot.py's JOINT_ORDER.
JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",   # gripper isn't a kinematic DOF; we ignore it in FK/IK
]

# ikpy prepends a "Base link" and may include fixed joints as non-actuated links.
# The exact mask depends on your URDF. Common pattern for a 6-DOF arm with
# ikpy is 8 total links (base + 6 actuated + tool) with mask [F, T, T, T, T, T, T, F].
# We'll build the mask dynamically from the chain below — see Kinematics.__init__.


class Kinematics:
    def __init__(
        self,
        urdf_path: str | Path,
        base_elements: Optional[list[str]] = None,
        tool_link_name: Optional[str] = None,
    ):
        urdf_path = str(urdf_path)

        # Build the chain. If auto-detection of the base fails, pass
        # base_elements=["base_link"] or whatever your URDF's root is called.
        if base_elements is not None:
            self.chain = Chain.from_urdf_file(urdf_path, base_elements=base_elements)
        else:
            self.chain = Chain.from_urdf_file(urdf_path)

        # Build active_links_mask: exclude fixed joints, include revolute/prismatic.
        # ikpy's `links` list has `.joint_type` on each link.
        self.active_mask = [
            link.joint_type != "fixed" for link in self.chain.links
        ]
        # Mask out the base link (first) — it's always fixed but some URDFs label it oddly.
        if len(self.active_mask) > 0:
            self.active_mask[0] = False
        self.chain.active_links_mask = self.active_mask

        self.num_links = len(self.chain.links)
        self.tool_link_name = tool_link_name

        # For IK seeding
        self._last_solution_rad = np.zeros(self.num_links)

    # ─── Forward kinematics ────────────────────────────────────────────────

    def forward(self, joint_degs: dict[str, float]) -> tuple[float, float, float]:
        """
        Returns (x_mm, y_mm, z_mm) of the end-effector in robot base frame.
        Input: dict from JOINT_ORDER names (except 'gripper') to degrees.
        """
        joint_rads_full = self._dict_to_full_vector(joint_degs)
        pose = self.chain.forward_kinematics(joint_rads_full)
        xyz_m = pose[:3, 3]
        return float(xyz_m[0] * 1000), float(xyz_m[1] * 1000), float(xyz_m[2] * 1000)

    # ─── Inverse kinematics ────────────────────────────────────────────────

    def inverse(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        seed_joint_degs: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        """
        Position-only IK. Returns dict of joint name → degrees.
        For orientation control, extend to use a full target_orientation.
        """
        target_xyz_m = np.array([x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0])

        if seed_joint_degs is not None:
            initial = self._dict_to_full_vector(seed_joint_degs)
        else:
            initial = self._last_solution_rad

        solution_rad = self.chain.inverse_kinematics(
            target_position=target_xyz_m,
            initial_position=initial,
        )
        self._last_solution_rad = solution_rad
        return self._full_vector_to_dict(solution_rad)

    # ─── Utilities ─────────────────────────────────────────────────────────

    def _dict_to_full_vector(self, joint_degs: dict[str, float]) -> np.ndarray:
        """
        Produce the full-length joint vector ikpy expects. Non-actuated links
        (base, fixed tool) get 0. Actuated links in order: shoulder_pan,
        shoulder_lift, elbow_flex, wrist_flex, wrist_roll.
        """
        actuated_names = [n for n in JOINT_ORDER if n != "gripper"]
        actuated_degs = [joint_degs.get(n, 0.0) for n in actuated_names]
        actuated_rads = np.deg2rad(actuated_degs)

        full = np.zeros(self.num_links)
        active_indices = [i for i, active in enumerate(self.active_mask) if active]

        if len(active_indices) != len(actuated_rads):
            raise RuntimeError(
                f"URDF has {len(active_indices)} active joints but "
                f"JOINT_ORDER has {len(actuated_rads)} actuated entries. "
                f"Fix JOINT_ORDER or active_mask to match your URDF."
            )

        for link_idx, rad in zip(active_indices, actuated_rads):
            full[link_idx] = rad
        return full

    def _full_vector_to_dict(self, full_vec: np.ndarray) -> dict[str, float]:
        actuated_names = [n for n in JOINT_ORDER if n != "gripper"]
        active_indices = [i for i, active in enumerate(self.active_mask) if active]
        out = {}
        for name, link_idx in zip(actuated_names, active_indices):
            out[name] = float(np.rad2deg(full_vec[link_idx]))
        return out

    # ─── Debug ─────────────────────────────────────────────────────────────

    def describe(self):
        """Print the chain structure. Run this once when debugging URDF loading."""
        print(f"Loaded chain with {self.num_links} links:")
        for i, link in enumerate(self.chain.links):
            active = "ACTIVE" if self.active_mask[i] else "fixed "
            print(f"  [{i}] {active} {link.name} (type={link.joint_type})")