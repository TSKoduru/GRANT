from .scan_types import ObjectState, Pose6D, RGBDFrame


class CoverageMap:
    """
    Sphere of viewing directions around the object centroid.
    Discretized into a grid of (elevation, azimuth) cells.
    Each cell stores how well that surface patch has been observed.
    """

    def update(
        self,
        frame: RGBDFrame,
        camera_pose: Pose6D,
        object_state: ObjectState,
    ) -> None:
        """
        For each point in the frame's point cloud, find the corresponding
        surface normal and mark that surface cell as observed.
        Weight by viewing angle (perpendicular view = high weight,
        grazing angle = low weight).
        """

    def get_coverage_score(self) -> float:
        """Fraction of cells above observation threshold. 0.0-1.0."""

    def get_next_viewpoint(
        self,
        object_state: ObjectState,
        camera_arm_reachable_poses: list[Pose6D],
    ) -> tuple[Pose6D, float]:
        """
        Finds the camera pose that would observe the most currently
        under-observed surface cells.

        Strategy: for each candidate camera pose, raycast against the
        coverage map and count how many weak cells would be hit.
        Return the pose with the highest expected information gain.

        Returns (best_camera_pose, expected_new_coverage_fraction).
        """

    def get_next_object_rotation(
        self,
        object_state: ObjectState,
    ) -> float | None:
        """
        If the best unobserved regions are currently facing away from
        all reachable camera poses, return a wrist rotation (radians)
        that would expose them.
        Returns None if current orientation is sufficient.
        """
