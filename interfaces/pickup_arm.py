from dataclasses import dataclass

from ..scan_types import JointAngles, Pose6D


class PickupArm:

    def pickup_object(
        self,
        target_pose: Pose6D,              # where to reach for — centroid + approach vector
        approach_distance: float = 0.05,  # meters, how far above to start descent
    ) -> "GripResult":
        """
        Moves to approach pose, descends, activates suction, verifies seal.
        Returns whether seal was achieved and grip confidence.
        """

    def release_object(self) -> bool:
        """
        De-energizes solenoid. Returns True if pressure confirms release.
        """

    def rotate_object(
        self,
        delta_radians: float,
        speed: float = 0.2,
    ) -> "RotateResult":
        """
        Rotates wrist servo by delta_radians from current position.
        Positive = clockwise viewed from above.
        Blocks until complete. Returns actual rotation achieved via encoder.
        """

    def rotate_to_angle(
        self,
        target_radians: float,            # absolute, relative to pickup origin
    ) -> "RotateResult":
        """Rotate wrist to an absolute angle relative to pickup origin."""

    def get_wrist_angle(self) -> float:
        """Current wrist angle in radians from KY-040 encoder."""

    def is_holding_object(self) -> bool:
        """True if seal detection (LDR/pressure) confirms grip."""

    def get_current_pose(self) -> Pose6D:
        """End-effector pose of the pickup arm via forward kinematics."""

    def get_joint_angles(self) -> JointAngles:
        """Raw joint angles from servo feedback — consumed by ArmSegmenter."""


@dataclass
class GripResult:
    success: bool
    seal_confidence: float                # 0.0-1.0 from LDR reading
    grip_pose: Pose6D                     # actual pose where grip was achieved


@dataclass
class RotateResult:
    success: bool
    requested_radians: float
    achieved_radians: float
    delta_error: float
