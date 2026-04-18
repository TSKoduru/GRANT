import time

from .interfaces.robotic_arm import RoboticArm
from .interfaces.vision import VisionSystem
from .registration import Registration
from .scan_types import (
    CapturedView,
    ObjectState,
    PointCloud,
    Pose6D,
    ScanError,
    ScanResult,
)


class ScanOrchestrator:
    """
    Fixed-trajectory scan with one arm that carries the depth camera (mounted
    at an angle) and a suction gripper for the mid-scan flip.

    Sweep pattern: the arm moves to a fixed position directly overhead the
    object, then rotates the wrist_roll through four angles (-90°, 0°, 90°,
    180°). Because the camera is tilted relative to the wrist, each rotation
    exposes a different side of the object to the depth sensor.

    Flow:
        1. Detect object on the table.
        2. Orientation 1: move overhead, capture at 4 wrist-roll angles.
        3. Flip: arm picks up, rotates ~180° about a horizontal axis,
           sets the object back on the table, releases.
        4. Re-detect object.
        5. Orientation 2: overhead + 4 wrist-roll angles again.
        6. Cross-orientation RANSAC+ICP align, TSDF-fuse every frame.
    """

    SETTLE_TIME_S = 0.5          # pause after wrist rotation for camera to stabilize
    MIN_ALIGNMENT_FITNESS = 0.3

    def __init__(
        self,
        arm: RoboticArm,
        vision: VisionSystem,
        registration: Registration,
        on_view_captured=None,   # optional: callable(CapturedView, phase, i, n) for live UI
    ):
        self.arm = arm
        self.vision = vision
        self.registration = registration
        self.on_view_captured = on_view_captured

    # ──────────────────────────────────────────────────────────────────

    def run_single_pass_scan(self) -> ScanResult:
        """One overhead wrist-roll pass without flipping. Use this for initial testing."""

        # ── Phase 1: Detect object ───────────────────────────────────
        init_frame = self.vision.capture_rgbd()
        object_state = self.vision.detect_object(init_frame)

        # ── Phase 2: Overhead wrist-roll scan ────────────────────────
        views = self._run_wrist_roll_scan(object_state, phase="orient-1")

        # ── Phase 3: Home ────────────────────────────────────────────
        self.arm.move_to_home()

        # ── Phase 4: TSDF fusion ─────────────────────────────────────
        mesh = self.registration.tsdf_fuse(views)
        cloud = self._fuse_views_to_cloud(views)

        return ScanResult(
            mesh=mesh,
            point_cloud=cloud,
            n_frames=len(views),
            alignment_fitness=1.0,
        )

    def run_full_scan(self) -> ScanResult:

        # ── Phase 1: Detect object ───────────────────────────────────
        init_frame = self.vision.capture_rgbd()
        object_state = self.vision.detect_object(init_frame)

        # ── Phase 2: Orientation 1 scan ──────────────────────────────
        views_orient_1 = self._run_wrist_roll_scan(object_state, phase="orient-1")

        # ── Phase 3: Flip (pickup → flip → release) ──────────────────
        self._flip_object_in_place(object_state.centroid_as_pose())

        # ── Phase 4: Re-detect after flip ────────────────────────────
        refresh_frame = self.vision.capture_rgbd()
        object_state = self.vision.detect_object(refresh_frame)

        # ── Phase 5: Orientation 2 scan ──────────────────────────────
        views_orient_2 = self._run_wrist_roll_scan(object_state, phase="orient-2")

        # ── Phase 6: Send arm home ───────────────────────────────────
        self.arm.move_to_home()

        # ── Phase 7: Cross-orientation alignment ─────────────────────
        cloud_1 = self._fuse_views_to_cloud(views_orient_1)
        cloud_2 = self._fuse_views_to_cloud(views_orient_2)

        T_2_to_1, fitness = self.registration.global_align(
            source=cloud_2, target=cloud_1
        )
        if fitness < self.MIN_ALIGNMENT_FITNESS:
            raise ScanError(
                f"Cross-orientation alignment failed "
                f"(fitness={fitness:.2f} < {self.MIN_ALIGNMENT_FITNESS}). "
                f"The flip may have left insufficient surface overlap."
            )

        views_orient_2_aligned = self.registration.apply_transform_to_views(
            views_orient_2, T_2_to_1
        )

        # ── Phase 8: TSDF fusion ─────────────────────────────────────
        all_views = views_orient_1 + views_orient_2_aligned
        mesh = self.registration.tsdf_fuse(all_views)

        # Merged point cloud returned alongside the mesh for inspection
        from .registration import _matrix_to_pose
        cloud_2_in_1 = self.registration.transform_to_world(
            cloud_2, _matrix_to_pose(T_2_to_1)
        )
        merged_cloud = self.registration.merge(cloud_1, cloud_2_in_1, run_icp=False)

        return ScanResult(
            mesh=mesh,
            point_cloud=merged_cloud,
            n_frames=len(all_views),
            alignment_fitness=fitness,
        )

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _run_wrist_roll_scan(
        self,
        object_state: ObjectState,
        phase: str = "orient-1",
    ) -> list[CapturedView]:
        """Move overhead, then capture at each wrist-roll angle."""
        target = object_state.centroid_as_pose()
        overhead = self.arm.get_overhead_pose(target)
        self.arm.move_to_pose(overhead, phase="grip")

        views: list[CapturedView] = []
        n = len(self.arm.WRIST_ROLL_SCAN_ANGLES)
        for i, angle_deg in enumerate(self.arm.WRIST_ROLL_SCAN_ANGLES):
            self.arm.rotate_wrist_roll(angle_deg)
            time.sleep(self.SETTLE_TIME_S)
            actual_pose = self.arm.get_current_pose()
            frame = self.vision.capture_rgbd()
            view = CapturedView(frame=frame, camera_pose=actual_pose)
            views.append(view)
            if self.on_view_captured is not None:
                self.on_view_captured(view, phase, i, n)
        return views

    def _flip_object_in_place(self, target_pose: Pose6D) -> None:
        grip = self.arm.pickup_object(target_pose=target_pose)
        if not grip.success:
            raise ScanError("Arm reported grip-motion failure during flip")
        flip = self.arm.flip_object()
        if not flip.success:
            raise ScanError("Arm reported flip-motion failure")
        self.arm.release_object()

    def _fuse_views_to_cloud(self, views: list[CapturedView]) -> PointCloud:
        accumulated = PointCloud.empty()
        for i, v in enumerate(views):
            cam_cloud = self.vision.frame_to_pointcloud(v.frame)
            world_cloud = self.registration.transform_to_world(cam_cloud, v.camera_pose)
            accumulated = self.registration.merge(
                accumulated, world_cloud, run_icp=(i > 0)
            )
        return accumulated
