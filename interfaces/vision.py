from ..scan_types import ObjectState, PointCloud, RGBDFrame


class VisionSystem:

    def capture_rgbd(self) -> RGBDFrame:
        """
        Triggers stereo capture on KV260.
        Runs rectification + SGBM disparity → depth map.
        Returns RGB + depth aligned to left camera frame.
        Blocks until frame ready (~100-200ms).
        """

    def detect_object(self, frame: RGBDFrame) -> ObjectState:
        """
        Called once at scan start.
        Segments the object from background using depth discontinuities.
        Returns bounding box, centroid, initial coverage map (all zeros).
        """

    def frame_to_pointcloud(self, frame: RGBDFrame) -> PointCloud:
        """
        Backprojects depth map through camera intrinsics to 3D points.
        Colors from RGB frame.
        Returns point cloud in camera space — caller transforms to world space.
        """
