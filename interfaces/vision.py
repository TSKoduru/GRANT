import cv2
import numpy as np
import depthai as dai
from datetime import timedelta

try:
    from ..scan_types import ObjectState, PointCloud, Pose6D, RGBDFrame, CameraIntrinsics, ScanError
except ImportError:
    from scan_types import ObjectState, PointCloud, Pose6D, RGBDFrame, CameraIntrinsics, ScanError

class VisionSystem:
    def __init__(self):
        self.fps = 25.0
        self.pipeline = dai.Pipeline()

        # Get device reference once — used for platform check and calibration.
        # Must be called before node creation, same as depth_align.py.
        device = self.pipeline.getDefaultDevice()
        platform = device.getPlatform()

        # ── Node setup — identical topology to depth_align.py ────────
        camRgb = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        left   = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right  = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        sync   = self.pipeline.create(dai.node.Sync)

        if platform == dai.Platform.RVC4:
            align = self.pipeline.create(dai.node.ImageAlign)

        stereo.setExtendedDisparity(True)
        sync.setSyncThreshold(timedelta(seconds=1 / (2 * self.fps)))

        rgbOut   = camRgb.requestOutput(size=(1280, 960), fps=self.fps, enableUndistortion=True)
        leftOut  = left.requestOutput(size=(640, 400), fps=self.fps)
        rightOut = right.requestOutput(size=(640, 400), fps=self.fps)

        rgbOut.link(sync.inputs["rgb"])
        leftOut.link(stereo.left)
        rightOut.link(stereo.right)

        if platform == dai.Platform.RVC4:
            stereo.depth.link(align.input)
            rgbOut.link(align.inputAlignTo)
            align.outputAligned.link(sync.inputs["depth_aligned"])
        else:
            stereo.depth.link(sync.inputs["depth_aligned"])
            rgbOut.link(stereo.inputAlignTo)

        self.queue = sync.out.createOutputQueue()

        # ── Start pipeline — mirrors 'with pipeline: pipeline.start()' ─
        # depth_align.py uses the context manager for device lifecycle;
        # for a long-lived class we enter it explicitly so capture_rgbd
        # can be called repeatedly without re-starting the pipeline.
        self.pipeline.__enter__()
        self.pipeline.start()

        # ── Calibration ───────────────────────────────────────────────
        # IMPORTANT: pass resize dims — calibration stores intrinsics at
        # native sensor resolution, not at the 1280×960 RGB output size.
        try:
            calib = device.readCalibration()
            M = calib.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A, resizeWidth=1280, resizeHeight=960
            )
            self.intrinsics = CameraIntrinsics(
                fx=M[0][0], fy=M[1][1], cx=M[0][2], cy=M[1][2],
                width=1280, height=960,
            )
        except Exception:
            self.intrinsics = CameraIntrinsics(
                fx=1000.0, fy=1000.0, cx=640.0, cy=480.0, width=1280, height=960,
            )

    def close(self):
        """Stop the pipeline and release the OAK-D device."""
        self.pipeline.__exit__(None, None, None)

    def capture_rgbd(self) -> RGBDFrame:
        """
        Pulls one synchronised RGB+depth frame from the queue.
        Mirrors depth_align.py's frame-handling exactly.
        Blocks until a frame is ready (~100-200 ms).
        """
        messageGroup = self.queue.get()
        assert isinstance(messageGroup, dai.MessageGroup)

        frameRgb   = messageGroup["rgb"]
        frameDepth = messageGroup["depth_aligned"]
        assert isinstance(frameRgb,   dai.ImgFrame)
        assert isinstance(frameDepth, dai.ImgFrame)

        # RGB — depth_align.py reads getCvFrame() and converts GRAY→BGR or uses as-is
        cvFrame = frameRgb.getCvFrame()
        if len(cvFrame.shape) == 2:
            rgb_array = cv2.cvtColor(cvFrame, cv2.COLOR_GRAY2RGB)
        else:
            rgb_array = cv2.cvtColor(cvFrame, cv2.COLOR_BGR2RGB)

        # Depth — depth_align.py saves as getFrame().astype(float32) / 1000.0
        depth_array_meters = frameDepth.getFrame().astype(np.float32) / 1000.0

        return RGBDFrame(
            rgb=rgb_array,
            depth=depth_array_meters,
            intrinsics=self.intrinsics,
        )

    def detect_object(self, frame: RGBDFrame) -> ObjectState:
        """
        Segments the object from the table using depth discontinuities.
        Returns the centroid and bounding box in camera space.

        NOTE: This returns coordinates in the camera frame. If the arm is at
        home when this is called, use arm.get_current_pose() + a world→camera
        transform to convert to world coordinates before using with move_to_pose.
        For the single-pass test script, pass the object position manually instead.
        """
        depth = frame.depth
        h, w = depth.shape

        # Estimate table plane depth as the ~85th percentile of valid center pixels
        # (table is the dominant far surface; object sticks up closer)
        center = depth[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        valid_center = center[center > 0]
        if len(valid_center) < 100:
            raise ScanError("detect_object: insufficient depth data in frame center")
        table_z = float(np.percentile(valid_center, 85))

        # Object mask: valid depth at least 5 cm closer than the table plane
        obj_mask = ((depth > 0) & (depth < table_z - 0.05)).astype(np.uint8) * 255

        # Morphological open to remove salt-and-pepper noise
        kernel = np.ones((3, 3), np.uint8)
        obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obj_mask)
        if n_labels < 2:
            raise ScanError(
                f"detect_object: no object found above table plane "
                f"(table_z={table_z:.3f}m, threshold={table_z - 0.05:.3f}m)"
            )

        # Largest connected component (skip background label 0)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        ys, xs = np.where(labels == largest)
        zs = depth[ys, xs]

        # Backproject pixel centroids to 3D camera space
        intr = frame.intrinsics
        x3 = (xs - intr.cx) / intr.fx * zs
        y3 = (ys - intr.cy) / intr.fy * zs

        centroid = Pose6D(
            x=float(np.mean(x3)),
            y=float(np.mean(y3)),
            z=float(np.mean(zs)),
        )
        bbox_min = np.array([x3.min(), y3.min(), zs.min()], dtype=np.float32)
        bbox_max = np.array([x3.max(), y3.max(), zs.max()], dtype=np.float32)

        return ObjectState(centroid=centroid, bbox_min=bbox_min, bbox_max=bbox_max)

    def frame_to_pointcloud(self, frame: RGBDFrame) -> PointCloud:
        """
        Backproject depth through the pinhole intrinsics and carry RGB per point.
        Returns the cloud in camera space — the caller transforms to world.
        """
        intr = frame.intrinsics
        depth = frame.depth
        h, w = depth.shape
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        valid = depth > 0

        z = depth[valid]
        x = (xs[valid] - intr.cx) / intr.fx * z
        y = (ys[valid] - intr.cy) / intr.fy * z
        points = np.stack([x, y, z], axis=-1).astype(np.float32)

        colors = None
        if frame.rgb is not None and frame.rgb.shape[:2] == depth.shape:
            colors = frame.rgb[valid].astype(np.uint8)

        return PointCloud(points=points, colors=colors)