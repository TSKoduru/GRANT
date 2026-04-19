"""
robot.py — hardware abstraction for the SO-101 + two cameras.

Low-level: no safety logic, no task logic. Always go through SafeRobot in
practice. Operates in degrees throughout.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from kinematics import Kinematics

from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig


# ─────────────────────────────────────────────────────────────────────────────
# Limits — tighten per your specific arm after a shakeout run.
# ─────────────────────────────────────────────────────────────────────────────

JOINT_LIMITS_DEG = {
    "shoulder_pan":  (-110, 110),
    "shoulder_lift": (-90,   90),
    "elbow_flex":    (-90,   90),
    "wrist_flex":    (-90,   90),
    "wrist_roll":    (-180, 180),
    "gripper":       (0,    100),
}
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
KINEMATIC_JOINTS = JOINT_ORDER[:5]

WORKSPACE_BOX_MM = {
    "x": (-250, 250),
    "y": (50,   350),
    "z": (5,    300),
}

JOINT_MAX_VEL_DPS = {k: 120.0 for k in JOINT_ORDER}


# ─────────────────────────────────────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Observation:
    overhead_bgr: np.ndarray
    wrist_bgr: np.ndarray
    overhead_jpeg_b64: str
    wrist_jpeg_b64: str
    joint_positions_deg: dict[str, float]
    gripper_width: float
    gripper_tip_mm: tuple[float, float, float]
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# Robot
# ─────────────────────────────────────────────────────────────────────────────

class Robot:
    def __init__(
        self,
        overhead_cam_idx: int = 1,
        wrist_cam_idx: int = 2,
        frame_w: int = 640,
        frame_h: int = 480,
        urdf_path: str = "models/so101.urdf",
        port: str = "/dev/ttyACM0",
        mock: bool = False,
    ):
        self.mock = mock
        self._last_joint_cmd: dict[str, float] = {}

        self.kin = Kinematics(urdf_path)

        if not mock:
            self.overhead_cam = cv2.VideoCapture(overhead_cam_idx)
            self.wrist_cam = cv2.VideoCapture(wrist_cam_idx)
            for cam in (self.overhead_cam, self.wrist_cam):
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
                cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            print(f"[Robot] Connecting SO-101 on {port}...")
            config = SO101FollowerConfig(port=port, use_degrees=True)
            self.arm = SO101Follower(config)
            self.arm.connect()
            print("[Robot] Connected.")
        else:
            self.overhead_cam = None
            self.wrist_cam = None
            self.arm = None

        self.frame_w, self.frame_h = frame_w, frame_h

        self.home_pose_deg = {
            "shoulder_pan":  0.0,
            "shoulder_lift": -20.0,
            "elbow_flex":    40.0,
            "wrist_flex":    0.0,
            "wrist_roll":    0.0,
            "gripper":       50.0,
        }
        self._last_joint_cmd = dict(self.home_pose_deg)

    # ── State ──────────────────────────────────────────────────────────────

    def get_joint_positions_deg(self) -> dict[str, float]:
        if self.mock or self.arm is None:
            return dict(self._last_joint_cmd)
        obs = self.arm.get_observation()
        q = obs["q"]
        return {name: float(q[i]) for i, name in enumerate(JOINT_ORDER)}

    def get_gripper_width(self) -> float:
        pos = self.get_joint_positions_deg().get("gripper", 0.0)
        lo, hi = JOINT_LIMITS_DEG["gripper"]
        span = hi - lo
        return max(0.0, min(1.0, (pos - lo) / span)) if span > 0 else 0.0

    def get_gripper_tip_mm(self) -> tuple[float, float, float]:
        """Gripper tip in robot base frame (mm) via forward kinematics."""
        return self.kin.forward(self.get_joint_positions_deg())

    # ── Commands ───────────────────────────────────────────────────────────

    def set_joint_positions_deg(
        self, targets: dict[str, float], speed: float = 0.3, hz: float = 30.0
    ):
        """Interpolated joint command — smooth motion at `hz`."""
        end = dict(self._last_joint_cmd)
        end.update(targets)
        end_vec = np.array([end[n] for n in JOINT_ORDER], dtype=np.float32)
        self._last_joint_cmd = end

        if self.mock or self.arm is None:
            time.sleep(0.3 / max(speed, 0.05))
            return

        start_vec = np.asarray(self.arm.get_observation()["q"], dtype=np.float32)
        max_delta = float(np.max(np.abs(end_vec - start_vec)))
        duration_s = max((max_delta / 90.0) / max(speed, 0.05), 0.05)
        steps = max(int(duration_s * hz), 1)

        for step in np.linspace(start_vec, end_vec, steps):
            self.arm.send_action(step.astype(np.float32))
            time.sleep(1.0 / hz)

    def move_ik(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        speed: float = 0.3,
        top_down: bool = True,
    ):
        """Cartesian move via IK. `top_down=True` forces a downward-facing end-effector."""
        seed = self.get_joint_positions_deg()
        target_z = np.array([0, 0, -1.0]) if top_down else None
        joint_degs = self.kin.inverse(x_mm, y_mm, z_mm, seed_joint_degs=seed,
                                       target_z_vector=target_z)
        # Preserve current gripper (IK doesn't know about it)
        joint_degs["gripper"] = seed["gripper"]
        self.set_joint_positions_deg(joint_degs, speed=speed)

    def set_gripper(self, width: float, speed: float = 0.5):
        width = max(0.0, min(1.0, width))
        lo, hi = JOINT_LIMITS_DEG["gripper"]
        self.set_joint_positions_deg({"gripper": lo + width * (hi - lo)}, speed=speed)

    def home(self):
        self.set_joint_positions_deg(self.home_pose_deg, speed=0.2)

    # ── Observation ────────────────────────────────────────────────────────

    def _read_frame(self, cam) -> np.ndarray:
        if cam is None:
            return np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        cam.grab()
        ok, frame = cam.read()
        if not ok or frame is None:
            return np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        return frame

    @staticmethod
    def _jpeg_b64(bgr: np.ndarray, quality: int = 80) -> str:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

    def get_observation(self) -> Observation:
        overhead = self._read_frame(self.overhead_cam)
        wrist = self._read_frame(self.wrist_cam)
        return Observation(
            overhead_bgr=overhead,
            wrist_bgr=wrist,
            overhead_jpeg_b64=self._jpeg_b64(overhead),
            wrist_jpeg_b64=self._jpeg_b64(wrist),
            joint_positions_deg=self.get_joint_positions_deg(),
            gripper_width=self.get_gripper_width(),
            gripper_tip_mm=self.get_gripper_tip_mm(),
        )

    def close(self):
        if self.overhead_cam is not None:
            self.overhead_cam.release()
        if self.wrist_cam is not None:
            self.wrist_cam.release()
        if self.arm is not None:
            try:
                self.arm.disconnect()
            except Exception as e:
                print(f"[Robot] disconnect warning: {e}")