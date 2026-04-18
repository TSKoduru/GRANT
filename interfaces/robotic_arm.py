from dataclasses import dataclass

from ..scan_types import JointAngles, Pose6D


class RoboticArm:
    """
    Single 6-DoF arm that does double duty: the end-effector carries the
    stereo depth camera, and a suction gripper is mounted adjacent to it.
    Geometry guarantees the gripper is never in the camera's frame when
    the camera is aimed at the object — so no arm segmentation is needed.

    What we give up by going single-arm: there is no independent
    seal-detection sensor. `pickup_object()` returns whether the commanded
    approach+grip motion completed cleanly, not whether the suction cup
    actually sealed. We rely on the known FK-based approach pose to land
    the cup on the surface, and on the next capture to reveal a bad grip
    (the object will have moved).
    """

    # ── Motion ──────────────────────────────────────────────────────

    def move_to_pose(
        self,
        pose: Pose6D,
        speed: float = 0.3,        # 0.0-1.0, fraction of max speed
    ) -> "ArmMoveResult":
        """Move end-effector to a world-space pose. Blocks until done."""

    def get_current_pose(self) -> Pose6D:
        """End-effector pose via forward kinematics."""

    def get_joint_angles(self) -> JointAngles:
        """Raw joint angles from servo position feedback."""

    def move_to_home(self) -> "ArmMoveResult":
        """Return to a known safe pose. Called after a scan completes."""

    # ── Scanning ────────────────────────────────────────────────────

    def get_arc_trajectory(
        self,
        axis: str,                 # "azimuth" (left→right) or "elevation" (top→bottom)
        target: Pose6D,            # point to aim the camera at (object centroid)
        n_steps: int = 12,
    ) -> list[Pose6D]:
        """
        Pre-baked camera sweep around `target` along one axis. The
        orchestrator walks this list in order, calling move_to_pose
        then capturing a frame at each step.
        """

    # ── Gripping (no seal verification) ─────────────────────────────

    def pickup_object(
        self,
        target_pose: Pose6D,              # centroid + approach vector
        approach_distance: float = 0.05,  # meters above the surface to start descent
    ) -> "GripResult":
        """
        Approach above `target_pose`, descend, activate suction, retract.
        No post-grip seal verification — `success` just means the motion
        completed without kinematic errors.
        """

    def release_object(self) -> bool:
        """De-energize suction solenoid. Returns True on commanded release."""

    def flip_object(self) -> "RotateResult":
        """
        Turn the held object ~180° about a horizontal axis and set it
        back down on the table, then release. Exact achieved rotation
        is returned via the wrist encoder; the global registration step
        absorbs small angular errors.
        """


@dataclass
class ArmMoveResult:
    success: bool
    achieved_pose: Pose6D
    error_mm: float                # distance from target to achieved position
    message: str


@dataclass
class GripResult:
    success: bool                  # did the commanded motion complete?
    grip_pose: Pose6D              # where the grip was attempted


@dataclass
class RotateResult:
    success: bool
    requested_radians: float
    achieved_radians: float
    delta_error: float
