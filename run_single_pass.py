#!/usr/bin/env python3
"""
Single-pass overhead wrist-roll depth scan.

The arm moves directly over the object and rotates the wrist through
-90, 0, 90, 180 degrees. The depth camera is mounted at an angle so
each rotation exposes a different side of the object. The 4 frames are
TSDF-fused into a mesh.

Usage (manual object position — recommended for first test):
    python run_single_pass.py --object-x 0.2 --object-y 0.0 --object-z 0.0

Usage (auto-detect object from depth):
    python run_single_pass.py --auto-detect

After the scan, open the mesh with:
    python -c "import open3d as o3d; m=o3d.io.read_triangle_mesh('scan_output/scan.ply'); o3d.visualization.draw_geometries([m])"
"""
import argparse
import os
import time

import cv2
import numpy as np
import open3d as o3d

from interfaces.robotic_arm import RoboticArm
from interfaces.vision import VisionSystem
from registration import Registration
from scan_types import CapturedView, ObjectState, PointCloud, Pose6D, ScanError


def main():
    parser = argparse.ArgumentParser(description="Single-pass overhead wrist-roll scan")
    parser.add_argument("--object-x", type=float, default=0.20,
                        help="Object centroid X in world frame (m), default 0.20")
    parser.add_argument("--object-y", type=float, default=0.00,
                        help="Object centroid Y in world frame (m), default 0.00")
    parser.add_argument("--object-z", type=float, default=0.00,
                        help="Object centroid Z / table height in world frame (m), default 0.00")
    parser.add_argument("--auto-detect", action="store_true",
                        help="Use depth-based object detection instead of manual position")
    parser.add_argument("--out-dir", default="scan_output",
                        help="Directory to write mesh + depth images (default: scan_output)")
    parser.add_argument("--port", default="/dev/ttyACM0",
                        help="Arm serial port (default: /dev/ttyACM0)")
    parser.add_argument("--urdf", default="models/so101.urdf",
                        help="Path to arm URDF (default: models/so101.urdf)")
    parser.add_argument("--visualize", action="store_true",
                        help="Open Open3D viewer when done")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Initialize hardware ──────────────────────────────────────────
    print("[run_single_pass] Connecting to arm...")
    arm = RoboticArm(port=args.port, urdf_path=args.urdf)

    print("[run_single_pass] Connecting to depth camera...")
    vision = VisionSystem()

    reg = Registration()

    # ── Determine object position ────────────────────────────────────
    if args.auto_detect:
        print("[run_single_pass] Capturing frame for object detection...")
        init_frame = vision.capture_rgbd()
        object_state = vision.detect_object(init_frame)
        print(f"[run_single_pass] Detected object centroid (camera space): {object_state.centroid}")
        print("  WARNING: centroid is in camera space. If the arm is not at home,")
        print("  the overhead position will be wrong. Use --object-x/y/z if you know")
        print("  the approximate world position.")
    else:
        centroid = Pose6D(x=args.object_x, y=args.object_y, z=args.object_z)
        half = 0.05
        object_state = ObjectState(
            centroid=centroid,
            bbox_min=np.array([args.object_x - half, args.object_y - half, args.object_z],
                              dtype=np.float32),
            bbox_max=np.array([args.object_x + half, args.object_y + half, args.object_z + 0.1],
                              dtype=np.float32),
        )
        print(f"[run_single_pass] Object position: x={args.object_x}, y={args.object_y}, z={args.object_z}")

    # ── Move to overhead position ────────────────────────────────────
    target = object_state.centroid_as_pose()
    overhead = {
        'shoulder_pan.pos': 0,  # J1
        'shoulder_lift.pos': 0,  # J2
        'elbow_flex.pos': 0,    # J3
        'wrist_flex.pos': 90,    # J4 (Pitch)
        'wrist_roll.pos': -90,    # J5 (Roll)
        'gripper.pos': 0      # Gripper position (unchanged)
    }
    arm.robot.send_action(overhead)
    time.sleep(2)

    # ── Wrist-roll scan: 4 captures ──────────────────────────────────
    print(f"[run_single_pass] Scanning at wrist-roll angles: {arm.WRIST_ROLL_SCAN_ANGLES}")
    views: list[CapturedView] = []

    for i, angle_deg in enumerate(arm.WRIST_ROLL_SCAN_ANGLES):
        print(f"[run_single_pass]  [{i+1}/4] wrist_roll → {angle_deg:+.0f}°")
        arm.rotate_wrist_roll(angle_deg)
        time.sleep(0.5)  # extra settle on top of rotate_wrist_roll's 1.5 s

        actual_pose = arm.get_current_pose()
        frame = vision.capture_rgbd()
        views.append(CapturedView(frame=frame, camera_pose=actual_pose))

        # Save depth visualisation for inspection
        depth_vis = cv2.normalize(frame.depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_path = os.path.join(args.out_dir, f"depth_{i:02d}_roll{int(angle_deg):+04d}.png")
        cv2.imwrite(depth_path, depth_vis)
        print(f"             depth saved → {depth_path}")

    # ── Return home and release camera ──────────────────────────────
    print("[run_single_pass] Returning to home...")
    arm.move_to_home()
    vision.close()

    # ── TSDF fusion ──────────────────────────────────────────────────
    print("[run_single_pass] Fusing mesh via TSDF...")
    try:
        mesh = reg.tsdf_fuse(views)
    except Exception as exc:
        print(f"[run_single_pass] TSDF failed ({exc}), falling back to Poisson...")
        cloud = PointCloud.empty()
        for j, v in enumerate(views):
            cam_cloud = vision.frame_to_pointcloud(v.frame)
            world_cloud = reg.transform_to_world(cam_cloud, v.camera_pose)
            cloud = reg.merge(cloud, world_cloud, run_icp=(j > 0))
        mesh = reg.reconstruct_mesh(cloud)

    mesh_path = os.path.join(args.out_dir, "scan.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    n_tri = len(mesh.triangles)
    print(f"[run_single_pass] Done — {n_tri} triangles → {mesh_path}")

    if args.visualize:
        print("[run_single_pass] Opening viewer (close window to exit)...")
        o3d.visualization.draw_geometries([mesh], window_name="Single Pass Scan")


if __name__ == "__main__":
    main()
