import cv2
import numpy as np
import depthai as dai
from datetime import timedelta

from ..scan_types import ObjectState, PointCloud, RGBDFrame, CameraIntrinsics

class VisionSystem:
    def __init__(self):
        """
        Initializes the DepthAI pipeline, nodes, and starts the device.
        """
        self.fps = 25.0
        self.pipeline = dai.Pipeline()
        self.device = self.pipeline.getDefaultDevice()
        self.platform = self.device.getPlatform()

        # Define sources and outputs
        camRgb = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        left = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        sync = self.pipeline.create(dai.node.Sync)

        if self.platform == dai.Platform.RVC4:
            align = self.pipeline.create(dai.node.ImageAlign)

        stereo.setExtendedDisparity(True)
        sync.setSyncThreshold(timedelta(seconds=1/(2*self.fps)))

        rgbOut = camRgb.requestOutput(size=(1280, 960), fps=self.fps, enableUndistortion=True)
        leftOut = left.requestOutput(size=(640, 400), fps=self.fps)
        rightOut = right.requestOutput(size=(640, 400), fps=self.fps)

        # Linking
        rgbOut.link(sync.inputs["rgb"])
        leftOut.link(stereo.left)
        rightOut.link(stereo.right)

        if self.platform == dai.Platform.RVC4:
            stereo.depth.link(align.input)
            rgbOut.link(align.inputAlignTo)
            align.outputAligned.link(sync.inputs["depth_aligned"])
        else:
            stereo.depth.link(sync.inputs["depth_aligned"])
            rgbOut.link(stereo.inputAlignTo)

        self.queue = sync.out.createOutputQueue()

        # Start the pipeline
        self.pipeline.start()

        # Attempt to read exact camera intrinsics; provide fallback if calibration fails.
        # IMPORTANT: pass resize dimensions — calibration stores intrinsics at the
        # sensor's native resolution, so without resize args the fx/fy/cx/cy will not
        # match the 1280x960 RGB output and point-cloud backprojection will be skewed.
        try:
            calibData = self.device.readCalibration()
            intrinsics_matrix = calibData.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A, resizeWidth=1280, resizeHeight=960,
            )
            self.intrinsics = CameraIntrinsics(
                fx=intrinsics_matrix[0][0], fy=intrinsics_matrix[1][1],
                cx=intrinsics_matrix[0][2], cy=intrinsics_matrix[1][2],
                width=1280, height=960,
            )
        except Exception:
            self.intrinsics = CameraIntrinsics(
                fx=1000.0, fy=1000.0, cx=640.0, cy=480.0, width=1280, height=960,
            )

    def capture_rgbd(self) -> RGBDFrame:
        """
        Triggers stereo capture on KV260.
        Runs rectification + SGBM disparity → depth map.
        Returns RGB + depth aligned to left camera frame.
        Blocks until frame ready (~100-200ms).
        """
        messageGroup = self.queue.get()
        frameRgb = messageGroup["rgb"]
        frameDepth = messageGroup["depth_aligned"]

        # 1. Process RGB
        cvFrame = frameRgb.getCvFrame()
        if len(cvFrame.shape) == 2:
            rgb_array = cv2.cvtColor(cvFrame, cv2.COLOR_GRAY2RGB)
        else:
            rgb_array = cv2.cvtColor(cvFrame, cv2.COLOR_BGR2RGB)

        # 2. Process Depth (Convert from mm uint16 to float32 meters per scan_types.py)
        depth_array_mm = frameDepth.getFrame()
        depth_array_meters = depth_array_mm.astype(np.float32) / 1000.0

        return RGBDFrame(
            rgb=rgb_array,
            depth=depth_array_meters,
            intrinsics=self.intrinsics,
        )

    def detect_object(self, frame: RGBDFrame) -> ObjectState:
        """
        Called once at scan start (and again after the flip).
        Segments the object from background using depth discontinuities.

        TODO: implement. A reasonable baseline:
          1. Crop depth to the table region (known from the arm's workspace).
          2. Threshold pixels whose depth is closer than the table plane.
          3. Connected-components; pick the largest.
          4. Centroid + axis-aligned bbox become the ObjectState.
        """
        raise NotImplementedError(
            "detect_object not implemented — see docstring for suggested approach"
        )

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