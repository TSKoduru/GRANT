"""
Shared type stubs used across the scanning pipeline.

These are deliberately minimal placeholders. Real implementations will replace
them with concrete geometry / image types, but every module in the project
imports from here so signatures stay consistent.
"""
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Pose6D:
    """6-DoF rigid transform: translation (x, y, z) + rotation (quaternion)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0


@dataclass
class JointAngles:
    """Ordered list of joint angles in radians, base → end-effector."""
    values: list[float] = field(default_factory=list)


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics for the left camera of the stereo rig."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class RGBDFrame:
    """Aligned RGB + depth capture. arm_mask is filled in by ArmSegmenter."""
    rgb: np.ndarray                 # H x W x 3, uint8
    depth: np.ndarray               # H x W, float32 meters
    intrinsics: CameraIntrinsics
    arm_mask: Optional[np.ndarray] = None   # H x W bool, True = arm pixel


@dataclass
class PointCloud:
    """World- or camera-space points with optional colors."""
    points: np.ndarray              # N x 3 float32
    colors: Optional[np.ndarray] = None  # N x 3 uint8

    @staticmethod
    def empty() -> "PointCloud":
        return PointCloud(points=np.zeros((0, 3), dtype=np.float32))


@dataclass
class ObjectState:
    """Tracked object pose + orientation relative to pickup origin."""
    centroid: Pose6D
    bbox_min: np.ndarray            # 3, float
    bbox_max: np.ndarray            # 3, float
    current_rotation: float = 0.0   # radians, accumulated wrist rotation

    def centroid_as_pose(self) -> Pose6D:
        return self.centroid


@dataclass
class ScanResult:
    mesh: Any                       # o3d.geometry.TriangleMesh
    point_cloud: PointCloud
    coverage_achieved: float
    n_frames: int


class ScanError(RuntimeError):
    """Raised when the scan cannot proceed (e.g. grip failure)."""
