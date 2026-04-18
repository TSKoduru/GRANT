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
    Fixed-trajectory scan with one arm that does both duties — it carries
    the depth camera for the sweeps and carries the suction gripper for
    the mid-scan flip.

    Flow:

        1. Detect object on the table.
        2. Orientation 1:
             arc 1 — camera sweeps left→right (azimuth)
             arc 2 — camera sweeps top→bottom (elevation)
        3. Flip: arm picks up the object, rotates ~180° about a horizontal
           axis, sets it back on the table, releases.
        4. Re-detect object (flip drops it at a slightly different spot).
        5. Orientation 2: arcs 1+2 again.
        6. Stitch orientations together with RANSAC+ICP on the accumulated
           point clouds, then TSDF-fuse every frame.

    The object is NOT held during capture — it sits on the table. The
    gripper is mounted so it's out of the camera's frame at all sweep
    poses, so there's no pickup-arm to segment out.
    """

    ARC_STEPS_AZIMUTH = 12
    ARC_STEPS_ELEVATION = 8
    MIN_ALIGNMENT_FITNESS = 0.3

    def __init__(
        self,
        arm: RoboticArm,
        vision: VisionSystem,
        registration: Registration,
    ):
        self.arm = arm
        self.vision = vision
        self.registration = registration

    # ──────────────────────────────────────────────────────────────────

    def run_full_scan(self) -> ScanResult:

        # ── Phase 1: Detect object ───────────────────────────────────
        init_frame = self.vision.capture_rgbd()
        object_state = self.vision.detect_object(init_frame)

        # ── Phase 2: Orientation 1 scan ──────────────────────────────
        views_orient_1 = self._run_both_arcs(object_state)

        # ── Phase 3: Flip (pickup → flip → release) ──────────────────
        self._flip_object_in_place(object_state.centroid_as_pose())

        # ── Phase 4: Re-detect after flip ────────────────────────────
        refresh_frame = self.vision.capture_rgbd()
        object_state = self.vision.detect_object(refresh_frame)

        # ── Phase 5: Orientation 2 scan ──────────────────────────────
        views_orient_2 = self._run_both_arcs(object_state)

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

    def _run_both_arcs(self, object_state: ObjectState) -> list[CapturedView]:
        views: list[CapturedView] = []
        target = object_state.centroid_as_pose()
        azimuth_poses = self.arm.get_arc_trajectory(
            axis="azimuth", target=target, n_steps=self.ARC_STEPS_AZIMUTH
        )
        elevation_poses = self.arm.get_arc_trajectory(
            axis="elevation", target=target, n_steps=self.ARC_STEPS_ELEVATION
        )
        for pose in azimuth_poses + elevation_poses:
            views.append(self._capture_at(pose))
        return views

    def _capture_at(self, camera_pose: Pose6D) -> CapturedView:
        self.arm.move_to_pose(camera_pose)
        actual_pose = self.arm.get_current_pose()
        frame = self.vision.capture_rgbd()
        return CapturedView(frame=frame, camera_pose=actual_pose)

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
