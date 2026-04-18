"""
robot.py — Hardware abstraction for the lerobot SO-101 + two cameras.

Responsibilities:
  - Connect/disconnect the SO-101 arm
  - Read joint state, servo temperatures, gripper current
  - Send joint-space and Cartesian (IK) commands
  - Capture frames from overhead + wrist cameras
  - Bundle state into an `Observation` for the rest of the system

Everything hardware-facing lives here. NO safety logic in this file —
that's safety.py's job. This file is allowed to crash the robot if called
directly; always go through SafeRobot in practice.

FILL-IN POINTS are marked TODO(lerobot). You know the library better than
I do; plug in the real calls where indicated.
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
# Hardware limits — TIGHTEN THESE after calibrating your specific arm.
# These are deliberately conservative; your SO-101 can probably do more.
# ─────────────────────────────────────────────────────────────────────────────

JOINT_LIMITS_DEG = {
    "waist":    (-110, 110),   # joint 1 — base rotation
    "shoulder": (-90,   90),   # joint 2
    "elbow":    (-90,   90),   # joint 3
    "pitch":    (-90,   90),   # joint 4 — wrist pitch
    "roll":     (-180, 180),   # joint 5 — wrist roll
    "gripper":  (0,    100),   # joint 6 — NOT in kinematic chain, normalized 0..100
}
# Order matches lerobot's obs["q"] and send_action() vectors:
JOINT_ORDER = ["waist", "shoulder", "elbow", "pitch", "roll", "gripper"]
# Only these 5 are in the kinematic chain (the URDF chain, see kinematics.py):
KINEMATIC_JOINTS = ["waist", "shoulder", "elbow", "pitch", "roll"]

# Cartesian workspace box in ROBOT BASE FRAME (mm).
# +x = right, +y = forward (away from the robot), +z = up. Table is z=0.
WORKSPACE_BOX_MM = {
    "x": (-250, 250),
    "y": (50,   350),   # keep a dead zone near the base so the arm can't hit itself
    "z": (5,    300),   # 5mm floor so the gripper doesn't scrape the table
}

# Per-joint max velocity in deg/sec. Safety layer will clamp to these.
JOINT_MAX_VEL_DPS = {k: 120.0 for k in JOINT_ORDER}

# Temperature threshold in Celsius. Above this, refuse new motion commands.
SERVO_TEMP_LIMIT_C = 55.0


# ─────────────────────────────────────────────────────────────────────────────
# Observation bundle — this is what the scorer + researcher consume.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """Snapshot of everything the robot can sense, at one instant."""
    overhead_bgr: np.ndarray              # full-res BGR array (for perception.py)
    wrist_bgr: np.ndarray
    overhead_jpeg_b64: str                # compressed for LLM/logs
    wrist_jpeg_b64: str
    joint_positions_deg: dict[str, float]
    gripper_width: float                  # normalized 0..1
    gripper_current_ma: float             # proxy for grip load on SO-101
    servo_temps_c: dict[str, float]
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# Robot class
# ─────────────────────────────────────────────────────────────────────────────

class Robot:
    """Low-level hardware wrapper. Thread-unsafe by design — one caller at a time."""

    def __init__(
        self,
        overhead_cam_idx: int = 1,           # /dev/video1 — webcam, overhead view
        wrist_cam_idx: int = 2,              # /dev/video2 — gripper-mounted camera
        frame_w: int = 640,
        frame_h: int = 480,
        urdf_path: str = "models/so101.urdf",
        port: str = "/dev/ttyACM0",
        mock: bool = False,
    ):
        self.mock = mock
        self._last_joint_cmd: dict[str, float] = {}

        # ── Kinematics (FK for gripper pose, IK for Cartesian commands) ──
        self.kin = Kinematics(urdf_path)

        # ── Cameras ──
        if not mock:
            self.overhead_cam = cv2.VideoCapture(overhead_cam_idx)
            self.wrist_cam = cv2.VideoCapture(wrist_cam_idx)
            for cam in (self.overhead_cam, self.wrist_cam):
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
                cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce frame latency
        else:
            self.overhead_cam = None
            self.wrist_cam = None
        self.frame_w, self.frame_h = frame_w, frame_h

        # ── Arm ──
        if not mock:
            print(f"[Robot] Connecting to SO-101 on {port}...")
            config = SO101FollowerConfig(port=port, use_degrees=True)
            self.arm = SO101Follower(config)
            self.arm.connect()
            print("[Robot] Arm connected.")
        else:
            self.arm = None

        # Default "home" pose. Safe, unfolded, gripper mid-open.
        # Tune these after you've got the arm moving.
        self.home_pose_deg = {
            "waist":    0.0,
            "shoulder": -20.0,
            "elbow":    40.0,
            "pitch":    0.0,
            "roll":     0.0,
            "gripper":  50.0,
        }
        self._last_joint_cmd = dict(self.home_pose_deg)

    # ─── State reads ────────────────────────────────────────────────────────

    def get_joint_positions_deg(self) -> dict[str, float]:
        """Current joint angles in degrees. Order matches JOINT_ORDER."""
        if self.mock or self.arm is None:
            return dict(self._last_joint_cmd)
        obs = self.arm.get_observation()
        q = obs["q"]   # 6-element array in degrees
        return {name: float(q[i]) for i, name in enumerate(JOINT_ORDER)}

    def get_gripper_width(self) -> float:
        """Gripper opening normalized to [0, 1] based on JOINT_LIMITS_DEG['gripper']."""
        pos = self.get_joint_positions_deg().get("gripper", 0.0)
        lo, hi = JOINT_LIMITS_DEG["gripper"]
        span = hi - lo
        return max(0.0, min(1.0, (pos - lo) / span)) if span > 0 else 0.0

    def get_gripper_current_ma(self) -> float:
        """Gripper servo current — proxy for grip load. Noisy; useful for contact detection."""
        if self.mock or self.arm is None:
            return 0.0
        
        try:
            # First, see if standard observation includes 'effort'
            obs = self.arm.get_observation()
            if "effort" in obs:
                gripper_idx = JOINT_ORDER.index("gripper")
                return float(obs["effort"][gripper_idx])
            
            # Fallback: Read directly from the LeRobot motor bus if available.
            # Assuming the gripper is motor ID 6 based on a standard 6-DOF setup.
            if hasattr(self.arm, 'motor_bus'):
                raw_current = self.arm.motor_bus.read("present_current", motor_ids=[6])[0]
                # STS3215 raw units are ~6.5 mA per tick
                return float(raw_current) * 6.5
                
        except Exception as e:
            # Fail gracefully so safety.py doesn't crash on a dropped packet
            print(f"[Robot] Warning: Could not read gripper current: {e}")
            
        return 0.0

    def get_servo_temps_c(self) -> dict[str, float]:
        """Per-servo temperature. Safety layer uses this to refuse commands when hot."""
        temps = {k: 30.0 for k in JOINT_ORDER}
        if self.mock or self.arm is None:
            return temps
            
        try:
            # Access the underlying motor bus to read standard Feetech/Dynamixel registers
            if hasattr(self.arm, 'motor_bus'):
                # Motor IDs are typically 1 through 6
                motor_ids = list(range(1, 7))
                raw_temps = self.arm.motor_bus.read("present_temperature", motor_ids=motor_ids)
                
                for i, name in enumerate(JOINT_ORDER):
                    # Map the returned list back to your joint names
                    temps[name] = float(raw_temps[i])
                    
        except Exception as e:
            print(f"[Robot] Warning: Could not read servo temperatures: {e}")
            
        return temps

    # ─── Commands ───────────────────────────────────────────────────────────

    def set_joint_positions_deg(self, targets: dict[str, float], speed: float = 0.3):
        """
        Send a joint-space command. `speed` in [0,1] scales the motion profile.
        This is the LOW-LEVEL call — no validation. Go through SafeRobot normally.
        """
        merged = dict(self._last_joint_cmd)
        merged.update(targets)
        self._last_joint_cmd = merged

        if self.mock or self.arm is None:
            time.sleep(0.3 / max(speed, 0.05))
            return

        import torch
        
        # LeRobot expects a tensor of positions in the exact order configured.
        # Map our merged dictionary into a flat list based on JOINT_ORDER.
        action_list = [merged[name] for name in JOINT_ORDER]
        action_tensor = torch.tensor(action_list, dtype=torch.float32)
        
        # Send to the arm. (Note: standard lerobot send_action handles positional targets; 
        # handling 'speed' natively may require updating velocity registers via motor_bus 
        # if the SO101 driver doesn't support action-level speed profiling).
        self.arm.send_action(action_tensor)

    def move_ik(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        speed: float = 0.3,
    ):
        """
        Move end-effector to (x, y, z) in robot base frame via IK.
        Orientation (pitch, roll) is accepted for API compatibility but the
        position-only IK in kinematics.py ignores it. If you need full-pose
        IK, extend Kinematics.inverse() to take a target_orientation.
        """
        seed = self.get_joint_positions_deg()
        joint_degs = self.kin.inverse(x_mm, y_mm, z_mm, seed_joint_degs=seed)
        self.set_joint_positions_deg(joint_degs, speed=speed)

    def get_gripper_tip_mm(self) -> tuple[float, float, float]:
        """
        Gripper tip position in robot base frame (mm), from forward kinematics
        on the current joint state. This is our canonical "where is the gripper"
        query — no vision involved.
        """
        joints = self.get_joint_positions_deg()
        return self.kin.forward(joints)

    def set_gripper(self, width: float, speed: float = 0.5):
        """Open/close gripper. width=0 is closed, width=1 is fully open."""
        width = max(0.0, min(1.0, width))
        lo, hi = JOINT_LIMITS_DEG["gripper"]
        servo_deg = lo + width * (hi - lo)
        self.set_joint_positions_deg({"gripper": servo_deg}, speed=speed)

    def home(self):
        """Return to the calibrated home pose. Always safe to call."""
        self.set_joint_positions_deg(self.home_pose_deg, speed=0.2)

    # ─── Observations ───────────────────────────────────────────────────────

    def _read_frame(self, cam) -> np.ndarray:
        """Read one frame, discarding a stale buffered frame first."""
        if cam is None:
            return np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        cam.grab()   # flush buffer
        ok, frame = cam.read()
        if not ok or frame is None:
            return np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        return frame

    @staticmethod
    def _jpeg_b64(bgr: np.ndarray, quality: int = 80) -> str:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

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
            gripper_current_ma=self.get_gripper_current_ma(),
            servo_temps_c=self.get_servo_temps_c(),
        )

    # ─── Cleanup ────────────────────────────────────────────────────────────

    def close(self):
        if self.overhead_cam is not None:
            self.overhead_cam.release()
        if self.wrist_cam is not None:
            self.wrist_cam.release()
            
        # Cleanly disconnect the LeRobot arm to release the serial port
        if self.arm is not None:
            try:
                self.arm.disconnect()
                print("[Robot] Arm disconnected cleanly.")
            except Exception as e:
                print(f"[Robot] Error disconnecting arm: {e}")