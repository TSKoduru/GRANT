import numpy as np

from .scan_types import CameraIntrinsics, JointAngles, Pose6D, RGBDFrame


class ArmSegmenter:

    def __init__(self, arm_urdf_path: str, camera_intrinsics: CameraIntrinsics):
        """
        Loads the arm's URDF model so it can render a predicted
        silhouette given any joint configuration.
        """
        self.arm_urdf_path = arm_urdf_path
        self.camera_intrinsics = camera_intrinsics
        self.arm_hsv_range: tuple | None = None

    def get_mask(
        self,
        frame: RGBDFrame,
        joint_angles: JointAngles,        # from pickup_arm.get_joint_angles()
        camera_pose: Pose6D,              # from camera_arm.get_current_pose()
    ) -> np.ndarray:                      # H x W bool, True = arm pixel
        """
        Stage 1 — geometric prediction:
            Forward kinematics on joint_angles gives each arm link's
            pose in world space. Project each link's bounding geometry
            into image space using camera_pose + intrinsics.
            This gives a coarse but fast mask covering ~90% of arm pixels.

        Stage 2 — color refinement:
            Within a dilated version of the stage 1 mask, run HSV
            segmentation tuned to the arm's color to catch any pixels
            the geometric projection missed (joint edges, thin links).

        Stage 3 — depth consistency check:
            Any pixel where the measured depth is significantly closer
            than the object surface but wasn't caught by stages 1/2
            is also masked. Catches reflections and shadow artifacts.
        """

    def _project_arm_geometry(
        self,
        joint_angles: JointAngles,
        camera_pose: Pose6D,
    ) -> np.ndarray:                      # H x W bool, coarse geometric mask
        pass

    def _refine_with_color(
        self,
        rgb: np.ndarray,
        coarse_mask: np.ndarray,
        arm_hsv_range: tuple,             # calibrated once at startup
    ) -> np.ndarray:
        pass

    def _refine_with_depth(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        object_depth_estimate: float,     # approximate distance to object surface
    ) -> np.ndarray:
        pass

    def calibrate_arm_color(
        self,
        frame: RGBDFrame,
        joint_angles: JointAngles,
    ) -> tuple:
        """
        Call once at startup with the arm clearly visible and no object present.
        Samples HSV values from the geometric mask region to auto-tune
        the color segmentation range. Saves you hardcoding arm color.
        """
        pass
