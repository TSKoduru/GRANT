"""
skills.py — high-level action macros. Policy calls these, nothing else.

Everything here takes a SafeRobot and a Perceiver and bundles a small set of
motions into a named skill. Skills log what they did so the runner can score.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from perception import Detection, HSVRange, Perceiver
from safety import SafeRobot, SafetyViolation


@dataclass
class SkillLog:
    events: list[dict] = field(default_factory=list)

    def record(self, name: str, **kwargs):
        self.events.append({"name": name, "t": time.time(), **kwargs})


class Skills:
    def __init__(
        self,
        robot: SafeRobot,
        perceiver: Perceiver,
        log: SkillLog,
        approach_height_mm: float = 80.0,
        retreat_height_mm: float = 100.0,
    ):
        self.robot = robot
        self.perceiver = perceiver
        self.log = log
        self.approach_height = approach_height_mm
        self.retreat_height = retreat_height_mm

    # ── Vision ─────────────────────────────────────────────────────────────

    def detect_object_hsv(self, hsv: HSVRange) -> Optional[Detection]:
        frame = self.robot.get_observation().overhead_bgr
        det = self.perceiver.detect_hsv(frame, hsv)
        self.log.record("detect_object_hsv",
                        found=det is not None,
                        pos=(det.x_mm, det.y_mm) if det else None)
        return det

    def detect_object_bgsub(self) -> Optional[Detection]:
        frame = self.robot.get_observation().overhead_bgr
        gx, gy, _ = self.robot.get_gripper_tip_mm()
        grip_uv = self.perceiver.robot_mm_to_pixel(gx, gy) if self.perceiver.H is not None else None
        det = self.perceiver.detect_bgsubtract(frame, gripper_pixel_uv=grip_uv)
        self.log.record("detect_object_bgsub",
                        found=det is not None,
                        pos=(det.x_mm, det.y_mm) if det else None)
        return det

    def detect_target(self) -> Optional[Detection]:
        frame = self.robot.get_observation().overhead_bgr
        det = self.perceiver.detect_target_by_aruco(frame)
        self.log.record("detect_target",
                        found=det is not None,
                        pos=(det.x_mm, det.y_mm) if det else None)
        return det

    # ── Motion primitives ──────────────────────────────────────────────────

    def move_above(self, x_mm: float, y_mm: float,
                   height_mm: Optional[float] = None, speed: float = 0.3):
        h = height_mm if height_mm is not None else self.approach_height
        self.robot.move_ik(x_mm, y_mm, h, speed=speed)
        self.log.record("move_above", x=x_mm, y=y_mm, z=h)

    def descend_to(self, z_mm: float, speed: float = 0.15):
        gx, gy, _ = self.robot.get_gripper_tip_mm()
        self.robot.move_ik(gx, gy, z_mm, speed=speed)
        self.log.record("descend_to", z=z_mm)

    def lift(self, height_mm: Optional[float] = None, speed: float = 0.3):
        h = height_mm if height_mm is not None else self.retreat_height
        gx, gy, _ = self.robot.get_gripper_tip_mm()
        self.robot.move_ik(gx, gy, h, speed=speed)
        self.log.record("lift", z=h)

    def home(self):
        self.robot.home()
        self.log.record("home")

    # ── Gripper primitives ─────────────────────────────────────────────────

    def open_gripper(self, width: float = 1.0, speed: float = 0.6):
        self.robot.set_gripper(width, speed=speed)
        self.log.record("open_gripper", width=width)

    def close_gripper(self, width: float = 0.0, speed: float = 0.4,
                      settle_s: float = 0.3):
        self.robot.set_gripper(width, speed=speed)
        time.sleep(settle_s)
        self.log.record("close_gripper", width=width, settle=settle_s)

    # ── Composite macros ───────────────────────────────────────────────────

    def pick_at(
        self,
        x_mm: float, y_mm: float,
        grasp_z_mm: float = 10.0,
        pre_grasp_width: float = 1.0,
        grasp_close_width: float = 0.15,
        settle_s: float = 0.3,
        approach_offset_x_mm: float = 0.0,
        approach_offset_y_mm: float = 0.0,
    ):
        """Full grasp sequence: approach above, open, descend, close, lift."""
        tx, ty = x_mm + approach_offset_x_mm, y_mm + approach_offset_y_mm
        self.open_gripper(pre_grasp_width)
        self.move_above(tx, ty)
        self.descend_to(grasp_z_mm)
        self.close_gripper(grasp_close_width, settle_s=settle_s)
        self.lift()

    def place_at(self, x_mm: float, y_mm: float, release_z_mm: float = 30.0):
        """Carry to (x, y), descend, release, lift."""
        self.move_above(x_mm, y_mm)
        self.descend_to(release_z_mm)
        self.open_gripper(1.0)
        self.lift()

    def push_toward(
        self, from_x_mm: float, from_y_mm: float,
        to_x_mm: float, to_y_mm: float,
        push_z_mm: float = 15.0,
        approach_overshoot_mm: float = 30.0,
        follow_through_mm: float = 20.0,
    ):
        """
        Push an object from one point toward another. Approaches from behind
        the object (overshoot in the direction OPPOSITE of travel), descends,
        then drags through the target plus follow-through.

        Used by demo 3 (the "hook" — gripper can't grasp, only push).
        """
        direction = np.array([to_x_mm - from_x_mm, to_y_mm - from_y_mm])
        norm = np.linalg.norm(direction)
        if norm < 1e-3:
            return
        unit = direction / norm

        start_x = from_x_mm - unit[0] * approach_overshoot_mm
        start_y = from_y_mm - unit[1] * approach_overshoot_mm
        end_x = to_x_mm + unit[0] * follow_through_mm
        end_y = to_y_mm + unit[1] * follow_through_mm

        self.close_gripper(0.0, settle_s=0.1)     # fully closed so the "hook" is rigid
        self.move_above(start_x, start_y)
        self.descend_to(push_z_mm)
        self.robot.move_ik(end_x, end_y, push_z_mm, speed=0.2)
        self.log.record("push_toward", start=(start_x, start_y), end=(end_x, end_y))
        self.lift()