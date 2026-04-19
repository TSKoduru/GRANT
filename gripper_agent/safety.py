"""
safety.py — the shielding layer. Every robot command flows through SafeRobot.

Violations raise SafetyViolation, which the trial runner catches and records.
Violations do NOT crash a trial; the LLM sees them as feedback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from robot import (
    Robot, JOINT_LIMITS_DEG, JOINT_ORDER, WORKSPACE_BOX_MM,
)


class SafetyViolation(Exception):
    """Raised when a command is rejected by the shield."""


@dataclass
class SafetyStats:
    total_calls: int = 0
    rejected: int = 0
    reasons: list[str] = field(default_factory=list)

    def record_reject(self, reason: str):
        self.rejected += 1
        self.reasons.append(reason)


class SafeRobot:
    """
    Drop-in wrapper over Robot that validates every actuation call.
    Observation methods pass through unchanged.
    """

    def __init__(self, robot: Robot, consecutive_reject_cooldown_s: float = 2.0):
        self.robot = robot
        self.stats = SafetyStats()
        self._consecutive_rejects = 0
        self._cooldown_s = consecutive_reject_cooldown_s

    # ── Pass-throughs ──────────────────────────────────────────────────────

    def get_observation(self):
        return self.robot.get_observation()

    def get_joint_positions_deg(self):
        return self.robot.get_joint_positions_deg()

    def get_gripper_tip_mm(self):
        return self.robot.get_gripper_tip_mm()

    def get_gripper_width(self):
        return self.robot.get_gripper_width()

    def home(self):
        self.robot.home()

    def close(self):
        self.robot.close()

    # ── Guarded commands ───────────────────────────────────────────────────

    def set_joint_positions_deg(self, targets: dict[str, float], speed: float = 0.3):
        self.stats.total_calls += 1
        self._check_joint_limits(targets)
        self._check_speed(speed)
        self._apply_cooldown_if_thrashing()
        try:
            self.robot.set_joint_positions_deg(targets, speed=speed)
            self._consecutive_rejects = 0
        except Exception as e:
            raise SafetyViolation(f"Hardware error during set_joint: {e}") from e

    def move_ik(self, x_mm: float, y_mm: float, z_mm: float,
                speed: float = 0.3, top_down: bool = True):
        self.stats.total_calls += 1
        self._check_workspace(x_mm, y_mm, z_mm)
        self._check_speed(speed)
        self._apply_cooldown_if_thrashing()
        try:
            self.robot.move_ik(x_mm, y_mm, z_mm, speed=speed, top_down=top_down)
            self._consecutive_rejects = 0
        except Exception as e:
            raise SafetyViolation(f"IK/hardware error during move_ik: {e}") from e

    def set_gripper(self, width: float, speed: float = 0.5):
        self.stats.total_calls += 1
        if not (0.0 <= width <= 1.0):
            self._reject(f"gripper width {width} outside [0, 1]")
        self.robot.set_gripper(width, speed=speed)
        self._consecutive_rejects = 0

    # ── Checks ─────────────────────────────────────────────────────────────

    def _check_joint_limits(self, targets: dict[str, float]):
        for name, deg in targets.items():
            if name not in JOINT_LIMITS_DEG:
                self._reject(f"unknown joint '{name}'")
            lo, hi = JOINT_LIMITS_DEG[name]
            if not (lo <= deg <= hi):
                self._reject(f"{name}={deg}° outside limits [{lo}, {hi}]")

    def _check_workspace(self, x_mm: float, y_mm: float, z_mm: float):
        for axis, val in (("x", x_mm), ("y", y_mm), ("z", z_mm)):
            lo, hi = WORKSPACE_BOX_MM[axis]
            if not (lo <= val <= hi):
                self._reject(
                    f"IK target {axis}={val:.1f}mm outside workspace [{lo}, {hi}]"
                )

    def _check_speed(self, speed: float):
        if not (0.05 <= speed <= 1.0):
            self._reject(f"speed {speed} outside [0.05, 1.0]")

    def _apply_cooldown_if_thrashing(self):
        if self._consecutive_rejects >= 3:
            print(f"[SafeRobot] Cooling down for {self._cooldown_s}s after repeated rejects")
            time.sleep(self._cooldown_s)
            self._consecutive_rejects = 0

    def _reject(self, reason: str):
        self._consecutive_rejects += 1
        self.stats.record_reject(reason)
        raise SafetyViolation(reason)