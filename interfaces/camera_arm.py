from dataclasses import dataclass

from ..scan_types import JointAngles, Pose6D


class CameraArm:

    def move_to_pose(
        self,
        pose: Pose6D,
        speed: float = 0.3,        # 0.0-1.0, fraction of max speed
    ) -> "ArmMoveResult":
        """
        Move end-effector (stereo rig) to a world-space pose.
        Blocks until motion complete or timeout.
        Returns success/failure + actual pose achieved (may differ slightly).
        """

    def get_current_pose(self) -> Pose6D:
        """
        Returns current end-effector pose via forward kinematics
        from servo position readings.
        """

    def get_joint_angles(self) -> JointAngles:
        """
        Raw joint angles from Feetech servo position feedback.
        Used by registration to compute exact camera pose at capture time.
        """

    def get_reachable_poses(self) -> list[Pose6D]:
        """
        Discretized set of poses the arm can reach. Used by the coverage
        planner to pick the next viewpoint.
        """

    def move_to_home(self) -> "ArmMoveResult":
        """
        Return to a known safe position out of camera field of view.
        Called before each capture so arm doesn't self-occlude.
        """


@dataclass
class ArmMoveResult:
    success: bool
    achieved_pose: Pose6D
    error_mm: float                # distance from target to achieved position
    message: str
