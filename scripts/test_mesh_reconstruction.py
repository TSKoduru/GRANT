"""
End-to-end test of the fixed 4-arc pipeline using synthetic data.

No camera, no arms. We pick a ground-truth mesh, place it on a virtual
"table" (floor at z=0), and:

    Orientation 1:
        arc 1 — sweep azimuth (left→right around Z)
        arc 2 — sweep elevation (top→bottom over X)
        For each step, raycast a depth image from the camera pose and
        build an RGBDFrame. Store (frame, pose) pairs.

    Flip:
        Rotate the ground-truth mesh ~180° about X to simulate the
        pickup arm turning the object over. Apply a small random
        translation too, because the flip won't land the object back
        at exactly the same spot.

    Orientation 2:
        arc 1 and arc 2 again, identical trajectory to orientation 1.

Then:
    - Build a point cloud per orientation (transform_to_world + merge).
    - Cross-orientation align with global_align (RANSAC+ICP).
    - TSDF-fuse every frame (orientation 2 frames get their poses
      transformed by the global-align result first).
    - Report chamfer distance vs. ground truth.

Run (any of these works):
    python scripts/test_mesh_reconstruction.py                      # from GRANT/
    python GRANT/scripts/test_mesh_reconstruction.py                # from GRANT's parent
    python -m GRANT.scripts.test_mesh_reconstruction                # from GRANT's parent
"""
from __future__ import annotations

# Allow running this file directly (`python scripts/test_mesh_reconstruction.py`)
# in addition to `python -m GRANT.scripts.test_mesh_reconstruction`. Without
# this shim, the relative imports below fail with "attempted relative import
# with no known parent package".
if __name__ == "__main__" and __package__ in (None, ""):
    import pathlib
    import sys as _sys
    _here = pathlib.Path(__file__).resolve()
    # _here = .../GRANT/scripts/test_mesh_reconstruction.py
    # parents[2] = parent of GRANT, which must be on sys.path so "GRANT" imports
    _pkg_parent = _here.parents[2]
    if str(_pkg_parent) not in _sys.path:
        _sys.path.insert(0, str(_pkg_parent))
    __package__ = "GRANT.scripts"

import argparse
import math
import os

import numpy as np
import open3d as o3d

from ..registration import Registration, _matrix_to_pose
from ..scan_types import (
    CameraIntrinsics,
    CapturedView,
    PointCloud,
    Pose6D,
    RGBDFrame,
)


# ──────────────────────────────────────────────────────────────────────
# Virtual camera: intrinsics + depth raycast
# ──────────────────────────────────────────────────────────────────────

def make_intrinsics(width: int = 320, height: int = 240, fov_deg: float = 50.0) -> CameraIntrinsics:
    fx = fy = 0.5 * width / math.tan(math.radians(fov_deg) / 2)
    return CameraIntrinsics(
        fx=fx, fy=fy, cx=width / 2, cy=height / 2, width=width, height=height,
    )


def raycast_depth(
    scene: o3d.t.geometry.RaycastingScene,
    intrinsics: CameraIntrinsics,
    camera_to_world: np.ndarray,
    noise_std: float = 0.0008,
) -> np.ndarray:
    """Synthesize a depth image (meters) via raycasting."""
    K = np.array([
        [intrinsics.fx, 0, intrinsics.cx],
        [0, intrinsics.fy, intrinsics.cy],
        [0, 0, 1],
    ], dtype=np.float64)
    world_to_camera = np.linalg.inv(camera_to_world)
    rays = scene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(K),
        extrinsic_matrix=o3d.core.Tensor(world_to_camera),
        width_px=intrinsics.width,
        height_px=intrinsics.height,
    )
    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()   # distance along ray
    # t_hit is inf where ray missed — convert to 0 (TSDF ignores zero depth)
    depth = np.where(np.isfinite(t_hit), t_hit, 0.0).astype(np.float32)

    # Our rays have unit direction aligned with pixel direction, so t_hit IS
    # the distance along the ray. To get z-depth (perpendicular to image plane)
    # we'd need to scale by cos(angle from optical axis). For typical narrow
    # FOV this is close enough to ignore, but let's do it right:
    ys, xs = np.mgrid[0:intrinsics.height, 0:intrinsics.width].astype(np.float32)
    nx = (xs - intrinsics.cx) / intrinsics.fx
    ny = (ys - intrinsics.cy) / intrinsics.fy
    cos_theta = 1.0 / np.sqrt(1.0 + nx * nx + ny * ny)
    depth = depth * cos_theta

    if noise_std > 0:
        mask = depth > 0
        depth[mask] += np.random.normal(scale=noise_std, size=mask.sum()).astype(np.float32)
    return depth


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """Fake a plausible RGB image by shading the surface. TSDF color only."""
    rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
    mask = depth > 0
    if mask.any():
        shade = ((depth - depth[mask].min()) / (depth[mask].max() - depth[mask].min() + 1e-9))
        shade = (200 - 150 * shade).clip(0, 255).astype(np.uint8)
        rgb[..., 0] = shade * mask
        rgb[..., 1] = shade * mask
        rgb[..., 2] = shade * mask
    return rgb


# ──────────────────────────────────────────────────────────────────────
# Pose helpers
# ──────────────────────────────────────────────────────────────────────

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Camera-to-world 4x4. OpenCV convention: camera looks along +Z,
    +X is right, +Y is down (image coords).
    """
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)   # +X = right means right is forward × up
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)  # +Y = down in image → forward × right

    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = down
    T[:3, 2] = forward
    T[:3, 3] = eye
    return T


def matrix_to_pose6d(T: np.ndarray) -> Pose6D:
    return _matrix_to_pose(T)


# ──────────────────────────────────────────────────────────────────────
# Arc trajectories
# ──────────────────────────────────────────────────────────────────────

def cross_arc(
    target: np.ndarray,
    axis: str,                         # "x" or "y"
    half_length: float,                # ± distance from center along the axis (m)
    height: float,                     # constant Z above the object (m)
    n_steps: int,
) -> list[np.ndarray]:
    """
    One leg of the cross-parallel-to-table sweep. The camera moves in a
    straight line at constant height above the object, always aiming down
    at `target`. Two such legs (axis="x", axis="y") form the cross.
    """
    up_world = np.array([0.0, 0.0, 1.0])
    poses = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        offset = -half_length + t * 2.0 * half_length
        if axis == "x":
            eye = target + np.array([offset, 0.0, height])
        elif axis == "y":
            eye = target + np.array([0.0, offset, height])
        else:
            raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
        poses.append(look_at(eye, target, up_world))
    return poses


# ──────────────────────────────────────────────────────────────────────
# Fake scene
# ──────────────────────────────────────────────────────────────────────

def load_ground_truth(shape: str, target_extent_m: float = 0.1) -> o3d.geometry.TriangleMesh:
    if shape == "sphere":
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=60)
    elif shape == "box":
        mesh = o3d.geometry.TriangleMesh.create_box(width=0.09, height=0.06, depth=0.05)
        mesh.translate(-mesh.get_center())
    elif shape == "bunny":
        mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
    elif shape == "armadillo":
        mesh = o3d.io.read_triangle_mesh(o3d.data.ArmadilloMesh().path)
    elif shape == "knot":
        mesh = o3d.geometry.TriangleMesh.create_mobius(
            length_split=70, width_split=15,
            twists=1, raidus=0.04, flatness=1, width=0.02, scale=1.0,
        )
    else:
        raise ValueError(f"unknown shape: {shape}")

    bbox = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent())
    if extent.max() > 0:
        mesh.scale(target_extent_m / extent.max(), center=bbox.get_center())
    mesh.translate(-mesh.get_center())
    mesh.compute_vertex_normals()
    return mesh


def scene_from_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def flip_mesh(mesh: o3d.geometry.TriangleMesh, translation_std: float = 0.003) -> o3d.geometry.TriangleMesh:
    """Simulate the pickup arm flipping the object ~180° about X with small drift."""
    flipped = o3d.geometry.TriangleMesh(mesh)  # copy
    # 180° about X, plus a random ±5° wobble on each axis for realism
    wobble = np.random.uniform(-math.radians(5), math.radians(5), size=3)
    R = flipped.get_rotation_matrix_from_xyz((math.pi + wobble[0], wobble[1], wobble[2]))
    flipped.rotate(R, center=np.zeros(3))
    flipped.translate(np.random.normal(scale=translation_std, size=3))
    flipped.compute_vertex_normals()
    return flipped


# ──────────────────────────────────────────────────────────────────────
# Capture simulation
# ──────────────────────────────────────────────────────────────────────

def simulate_view(
    scene: o3d.t.geometry.RaycastingScene,
    intrinsics: CameraIntrinsics,
    camera_to_world_true: np.ndarray,
    noise_std: float,
    fk_drift_std: float,
) -> CapturedView:
    """
    Build one CapturedView. Depth comes from raycasting ground truth;
    the returned pose has FK drift applied so ICP has work to do.
    """
    depth = raycast_depth(scene, intrinsics, camera_to_world_true, noise_std=noise_std)
    rgb = depth_to_rgb(depth)
    frame = RGBDFrame(rgb=rgb, depth=depth, intrinsics=intrinsics)

    # Reported pose: true pose + translation noise
    reported = camera_to_world_true.copy()
    reported[:3, 3] += np.random.normal(scale=fk_drift_std, size=3)
    return CapturedView(frame=frame, camera_pose=matrix_to_pose6d(reported))


def run_arcs(
    scene: o3d.t.geometry.RaycastingScene,
    intrinsics: CameraIntrinsics,
    object_center: np.ndarray,
    n_x: int,
    n_y: int,
    half_length: float,
    height: float,
    noise_std: float,
    fk_drift_std: float,
) -> list[CapturedView]:
    """Run both legs of the cross parallel to the table."""
    views: list[CapturedView] = []
    for T in cross_arc(object_center, axis="x",
                       half_length=half_length, height=height, n_steps=n_x):
        views.append(simulate_view(scene, intrinsics, T, noise_std, fk_drift_std))
    for T in cross_arc(object_center, axis="y",
                       half_length=half_length, height=height, n_steps=n_y):
        views.append(simulate_view(scene, intrinsics, T, noise_std, fk_drift_std))
    return views


# ──────────────────────────────────────────────────────────────────────
# Fusion helpers (stand in for VisionSystem.frame_to_pointcloud)
# ──────────────────────────────────────────────────────────────────────

def frame_to_pointcloud(frame: RGBDFrame) -> PointCloud:
    """Backproject depth to a camera-space point cloud."""
    intr = frame.intrinsics
    depth = frame.depth
    ys, xs = np.mgrid[0:intr.height, 0:intr.width].astype(np.float32)
    valid = depth > 0
    z = depth[valid]
    x = (xs[valid] - intr.cx) / intr.fx * z
    y = (ys[valid] - intr.cy) / intr.fy * z
    pts = np.stack([x, y, z], axis=-1).astype(np.float32)
    return PointCloud(points=pts)


def views_to_cloud(views: list[CapturedView], reg: Registration) -> PointCloud:
    acc = PointCloud.empty()
    for i, v in enumerate(views):
        cam_cloud = frame_to_pointcloud(v.frame)
        world = reg.transform_to_world(cam_cloud, v.camera_pose)
        acc = reg.merge(acc, world, run_icp=(i > 0))
    return acc


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", choices=["sphere", "box", "bunny", "armadillo", "knot"],
                    default="bunny")
    ap.add_argument("--n-x", type=int, default=12, help="Cross X-axis sweep steps")
    ap.add_argument("--n-y", type=int, default=12, help="Cross Y-axis sweep steps")
    ap.add_argument("--half-length", type=float, default=0.18,
                    help="Half-length of each cross leg (m) — camera goes ± this far")
    ap.add_argument("--height", type=float, default=0.22,
                    help="Height of camera above the object (m)")
    ap.add_argument("--noise-std", type=float, default=0.0008, help="Depth sensor noise (m)")
    ap.add_argument("--fk-drift-std", type=float, default=0.002,
                    help="Per-view FK position drift (m)")
    ap.add_argument("--flip-drift-std", type=float, default=0.003,
                    help="Translation drift when the arm flips the object (m)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="test_output")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--no-tsdf", action="store_true",
                    help="Skip TSDF and fall back to Poisson on the merged cloud")
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Setup ──────────────────────────────────────────────────────
    gt = load_ground_truth(args.shape)
    intrinsics = make_intrinsics()
    reg = Registration()

    print(f"[gt] {args.shape}: {len(gt.vertices)} verts, {len(gt.triangles)} tris")
    print(f"[cam] intrinsics: {intrinsics.width}x{intrinsics.height}, "
          f"fx={intrinsics.fx:.1f}")

    # ── Orientation 1 ──────────────────────────────────────────────
    print("\n[orient-1] capturing…")
    scene_1 = scene_from_mesh(gt)
    views_1 = run_arcs(
        scene_1, intrinsics,
        object_center=np.zeros(3),
        n_x=args.n_x, n_y=args.n_y,
        half_length=args.half_length, height=args.height,
        noise_std=args.noise_std, fk_drift_std=args.fk_drift_std,
    )
    print(f"[orient-1] captured {len(views_1)} views")

    # ── Flip ───────────────────────────────────────────────────────
    print("[flip] simulating ~180° rotation about X")
    gt_flipped = flip_mesh(gt, translation_std=args.flip_drift_std)

    # ── Orientation 2 ──────────────────────────────────────────────
    print("[orient-2] capturing…")
    scene_2 = scene_from_mesh(gt_flipped)
    # Object center after flip — re-detect (here we peek at the ground truth
    # since there's no vision module in the test; real system uses detect_object)
    center_2 = np.asarray(gt_flipped.get_center())
    views_2 = run_arcs(
        scene_2, intrinsics,
        object_center=center_2,
        n_x=args.n_x, n_y=args.n_y,
        half_length=args.half_length, height=args.height,
        noise_std=args.noise_std, fk_drift_std=args.fk_drift_std,
    )
    print(f"[orient-2] captured {len(views_2)} views")

    # ── Cross-orientation alignment ────────────────────────────────
    print("\n[align] building per-orientation clouds…")
    cloud_1 = views_to_cloud(views_1, reg)
    cloud_2 = views_to_cloud(views_2, reg)
    print(f"[align] cloud_1: {cloud_1.points.shape[0]} pts, "
          f"cloud_2: {cloud_2.points.shape[0]} pts")

    print("[align] running RANSAC + ICP…")
    T_2_to_1, fitness = reg.global_align(source=cloud_2, target=cloud_1)
    print(f"[align] fitness={fitness:.3f}  (>0.3 is usually acceptable)")
    print(f"[align] translation: {T_2_to_1[:3, 3]}")

    # Sanity: decompose rotation to degrees for readability
    rot_angle = math.degrees(
        math.acos(np.clip((np.trace(T_2_to_1[:3, :3]) - 1) / 2, -1, 1))
    )
    print(f"[align] rotation angle: {rot_angle:.1f}° (expected ~180°)")

    views_2_aligned = reg.apply_transform_to_views(views_2, T_2_to_1)

    # ── Mesh ───────────────────────────────────────────────────────
    if args.no_tsdf:
        print("\n[mesh] Poisson on merged cloud…")
        cloud_2_in_1 = reg.transform_to_world(cloud_2, _matrix_to_pose(T_2_to_1))
        merged = reg.merge(cloud_1, cloud_2_in_1, run_icp=False)
        mesh = reg.reconstruct_mesh(merged)
    else:
        print("\n[mesh] TSDF fusion…")
        all_views = views_1 + views_2_aligned
        mesh = reg.tsdf_fuse(all_views)
    print(f"[mesh] {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # ── Evaluate ───────────────────────────────────────────────────
    gt_sample = gt.sample_points_uniformly(50000)
    if len(mesh.triangles) > 0:
        recon_sample = mesh.sample_points_uniformly(50000)
    else:
        recon_sample = o3d.geometry.PointCloud()
        recon_sample.points = mesh.vertices
    d1 = np.asarray(gt_sample.compute_point_cloud_distance(recon_sample))
    d2 = np.asarray(recon_sample.compute_point_cloud_distance(gt_sample))
    chamfer_mm = 0.5 * (d1.mean() + d2.mean()) * 1000.0
    print(f"[eval] chamfer distance to ground truth: {chamfer_mm:.2f} mm")

    # ── Save ───────────────────────────────────────────────────────
    mesh_path = os.path.join(args.out_dir, f"recon_{args.shape}.ply")
    gt_path = os.path.join(args.out_dir, f"gt_{args.shape}.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    o3d.io.write_triangle_mesh(gt_path, gt)
    print(f"[out] wrote {mesh_path}, {gt_path}")

    if args.visualize:
        mesh.paint_uniform_color([0.55, 0.70, 1.00])
        gt_wire = o3d.geometry.LineSet.create_from_triangle_mesh(gt)
        gt_wire.paint_uniform_color([1.0, 0.4, 0.4])
        o3d.visualization.draw_geometries([mesh, gt_wire])


if __name__ == "__main__":
    main()
