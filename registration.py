from typing import TYPE_CHECKING

from .scan_types import PointCloud, Pose6D

if TYPE_CHECKING:
    import open3d as o3d


class Registration:

    def transform_to_world(
        self,
        cloud: PointCloud,
        camera_pose: Pose6D,              # from camera_arm.get_current_pose()
    ) -> PointCloud:
        """
        Applies camera extrinsic matrix to move points from
        camera space → world space.
        This is just a matrix multiply — fast and exact if FK is accurate.
        """

    def merge(
        self,
        existing: PointCloud,
        new_cloud: PointCloud,
        run_icp: bool = True,             # True after first few frames as refinement
    ) -> PointCloud:
        """
        Combines two world-space point clouds.
        If run_icp=True, runs Open3D ICP to correct any FK drift
        before merging. ICP uses the overlap between clouds as anchor.
        Voxel-downsamples result to keep memory bounded.
        """

    def reconstruct_mesh(
        self,
        cloud: PointCloud,
    ) -> "o3d.geometry.TriangleMesh":
        """
        1. Estimate normals (Open3D KNN)
        2. Poisson surface reconstruction
        3. Trim low-density regions (removes reconstruction artifacts at edges)
        """
