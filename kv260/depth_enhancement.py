"""
Per-frame depth cleanup. Runs on the KV260 (FPGA/DPU) for latency, but
the algorithm is just OpenCV primitives — this file is the reference
implementation. The KV260 firmware reuses the same parameters.

Wire it in between `VisionSystem.capture_rgbd()` and the orchestrator:

    frame = vision.capture_rgbd()
    frame = enhancer.process(frame)
"""
from __future__ import annotations

import cv2
import numpy as np

from ..scan_types import RGBDFrame


class DepthEnhancer:
    """
    Four stages:
      1. Validity mask   — drop pixels outside a plausible depth range.
      2. Bilateral       — edge-preserving smoothing of the valid region.
      3. Speckle removal — drop connected components smaller than N pixels.
      4. Small-hole fill — replace tiny holes with dilated-neighbor depth.
    """

    DEPTH_MIN = 0.10                 # meters — anything closer is a reflection artifact
    DEPTH_MAX = 1.50                 # meters — beyond the workspace, ignore
    BILATERAL_D = 5                  # neighborhood diameter in pixels
    BILATERAL_SIGMA_COLOR = 0.03     # meters — depth-similarity band
    BILATERAL_SIGMA_SPACE = 5.0
    MIN_SPECKLE_SIZE = 50            # pixels — smaller blobs are noise
    FILL_HOLES = True
    FILL_KERNEL = 3                  # pixels — max hole diameter to fill

    def process(self, frame: RGBDFrame) -> RGBDFrame:
        depth = frame.depth.astype(np.float32, copy=True)
        valid = (depth >= self.DEPTH_MIN) & (depth <= self.DEPTH_MAX)
        depth[~valid] = 0.0

        # Bilateral only makes sense on the valid region
        smoothed = cv2.bilateralFilter(
            depth,
            self.BILATERAL_D,
            self.BILATERAL_SIGMA_COLOR,
            self.BILATERAL_SIGMA_SPACE,
        )
        smoothed[~valid] = 0.0

        # Speckle removal via connected components on the validity mask
        mask_u8 = (smoothed > 0).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        for label_idx in range(1, n_labels):
            if stats[label_idx, cv2.CC_STAT_AREA] < self.MIN_SPECKLE_SIZE:
                smoothed[labels == label_idx] = 0.0

        if self.FILL_HOLES:
            hole_mask = (smoothed == 0).astype(np.uint8)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.FILL_KERNEL, self.FILL_KERNEL)
            )
            # Dilate the valid-depth image, then paste the dilated value
            # into only the small holes (not large background gaps)
            dilated = cv2.dilate(smoothed, kernel, iterations=1)
            small_holes = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, kernel) == 0
            fill_targets = (hole_mask.astype(bool)) & small_holes
            smoothed[fill_targets] = dilated[fill_targets]

        return RGBDFrame(
            rgb=frame.rgb,
            depth=smoothed,
            intrinsics=frame.intrinsics,
        )
