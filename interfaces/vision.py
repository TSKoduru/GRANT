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
        self.platform = self.pipeline.getDefaultDevice().getPlatform()

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

        # Attempt to read exact camera intrinsics; provide fallback if calibration fails
        try:
            calibData = self.pipeline.getDefaultDevice().readCalibration()
            # We fetch CAM_A (RGB) intrinsics because the depth map is aligned to the RGB frame
            intrinsics_matrix = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A)
            self.intrinsics = CameraIntrinsics(
                fx=intrinsics_matrix[0][0], fy=intrinsics_matrix[1][1],
                cx=intrinsics_matrix[0][2], cy=intrinsics_matrix[1][2],
                width=1280, height=960
            )
        except Exception:
            self.intrinsics = CameraIntrinsics(
                fx=1000.0, fy=1000.0, cx=640.0, cy=480.0, width=1280, height=960
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
            arm_mask=None
        )

    def detect_object(self, frame: RGBDFrame) -> ObjectState:
        """
        Called once at scan start.
        Segments the object from background using depth discontinuities.
        Returns bounding box, centroid, initial coverage map (all zeros).
        """
        pass

    def frame_to_pointcloud(self, frame: RGBDFrame) -> PointCloud:
        """
        Backprojects depth map through camera intrinsics to 3D points.
        Colors from RGB frame.
        Returns point cloud in camera space — caller transforms to world space.
        """
        pass