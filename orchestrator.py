from .arm_segmenter import ArmSegmenter
from .coverage import CoverageMap
from .interfaces.camera_arm import CameraArm
from .interfaces.pickup_arm import PickupArm
from .interfaces.vision import VisionSystem
from .registration import Registration
from .scan_types import PointCloud, ScanError, ScanResult


class ScanOrchestrator:

    COVERAGE_TARGET = 0.92       # stop when 92% of surface observed
    MAX_ITERATIONS = 24          # hard cap on capture attempts
    MIN_NEW_COVERAGE = 0.02      # skip a viewpoint if it adds less than 2%

    def __init__(
        self,
        camera_arm: CameraArm,
        pickup_arm: PickupArm,
        vision: VisionSystem,
        coverage_map: CoverageMap,
        registration: Registration,
        arm_segmenter: ArmSegmenter,
    ):
        self.camera_arm = camera_arm
        self.pickup_arm = pickup_arm
        self.vision = vision
        self.coverage_map = coverage_map
        self.registration = registration
        self.arm_segmenter = arm_segmenter

    def run_full_scan(self) -> ScanResult:

        # ── Phase 1: Initialization ──────────────────────────────
        init_frame = self.vision.capture_rgbd()
        object_state = self.vision.detect_object(init_frame)

        grip_result = self.pickup_arm.pickup_object(
            target_pose=object_state.centroid_as_pose(),
        )
        if not grip_result.success:
            raise ScanError("Failed to grip object")

        accumulated_cloud = PointCloud.empty()
        iteration = 0

        # ── Phase 2: Scan loop ───────────────────────────────────
        while True:

            # Termination checks
            coverage = self.coverage_map.get_coverage_score()
            if coverage >= self.COVERAGE_TARGET:
                break
            if iteration >= self.MAX_ITERATIONS:
                break

            # Decide if object needs to be rotated first
            rotation_needed = self.coverage_map.get_next_object_rotation(object_state)
            if rotation_needed is not None:
                rotate_result = self.pickup_arm.rotate_object(rotation_needed)
                object_state.current_rotation += rotate_result.achieved_radians

            # Decide where camera arm should go
            next_camera_pose, expected_gain = self.coverage_map.get_next_viewpoint(
                object_state,
                self.camera_arm.get_reachable_poses(),
            )
            if expected_gain < self.MIN_NEW_COVERAGE:
                break   # diminishing returns, we're done

            # Move camera arm. Pickup arm holds still — it's already clear of the
            # camera FOV for most angles; if not, caller should move it to a safe pose.
            self.camera_arm.move_to_pose(next_camera_pose)

            # Capture
            frame = self.vision.capture_rgbd()
            frame.arm_mask = self.arm_segmenter.get_mask(
                frame,
                joint_angles=self.pickup_arm.get_joint_angles(),
                camera_pose=self.camera_arm.get_current_pose(),
            )

            # Integrate into model
            cloud_camera_space = self.vision.frame_to_pointcloud(frame)
            cloud_world_space = self.registration.transform_to_world(
                cloud_camera_space,
                camera_pose=self.camera_arm.get_current_pose(),
            )
            accumulated_cloud = self.registration.merge(
                accumulated_cloud,
                cloud_world_space,
            )
            self.coverage_map.update(
                frame,
                self.camera_arm.get_current_pose(),
                object_state,
            )

            iteration += 1

        # ── Phase 3: Finalize ────────────────────────────────────
        self.pickup_arm.rotate_to_angle(0.0)   # return object to start orientation
        self.pickup_arm.release_object()
        self.camera_arm.move_to_home()

        mesh = self.registration.reconstruct_mesh(accumulated_cloud)

        return ScanResult(
            mesh=mesh,
            point_cloud=accumulated_cloud,
            coverage_achieved=self.coverage_map.get_coverage_score(),
            n_frames=iteration,
        )
