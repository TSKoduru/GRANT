"""
Robot-guided RGB scan with known camera poses → COLMAP triangulation → mesh.

The arm moves through a fixed pattern (overhead wrist-rolls + X/Y arc sweeps),
captures an RGB frame at each position, and records the FK camera pose.
Those poses are written as priors for COLMAP's point_triangulator, skipping
the error-prone SfM pose-estimation step entirely.

Usage:
    python manual_scan.py --device 2

Flags:
    --device N       Camera index (e.g. 2 for /dev/video2, default: auto)
    --port           Arm serial port (default: /dev/ttyACM0)
    --urdf           Path to arm URDF (default: models/so101.urdf)
    --out FILE       Output mesh (default: manual_scan_mesh.ply)
    --keep-ws        Keep COLMAP workspace after reconstruction
    --no-preview     Skip SIFT feature preview after capture
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot

from interfaces.robotic_arm import RoboticArm
from scan_types import Pose6D


# ── Camera ────────────────────────────────────────────────────────────────────

def open_camera(preferred: int | None) -> tuple[cv2.VideoCapture, int]:
    indices = [preferred] if preferred is not None else list(range(8))
    for i in indices:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap, i
        cap.release()
    raise RuntimeError(
        "No working camera found. "
        "Specify one with --device N (e.g. --device 2 for /dev/video2)."
    )


def _capture_frame(cap: cv2.VideoCapture, path: Path, drain_seconds: float = 0.6) -> bool:
    """
    Drain buffered frames for `drain_seconds`, then capture.
    USB webcams often buffer 5–10 frames, so a fixed grab count isn't enough —
    without this the first capture frequently shows the previous arm position.
    """
    end = time.time() + drain_seconds
    while time.time() < end:
        cap.grab()
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(str(path), frame)
    return ret


# ── Scan waypoints ────────────────────────────────────────────────────────────
# Loaded from a JSON file: list of [0, J1, J2, J3, J4, J5, 0] in radians,
# matching the format printed by write_waypoints.py.

def load_waypoints(path: Path) -> list[tuple[str, dict]]:
    """Load waypoints from JSON. Each entry is [0, J1..J5, 0] in radians."""
    raw = json.loads(path.read_text())
    out: list[tuple[str, dict]] = []
    for i, wp in enumerate(raw):
        if len(wp) != 7:
            raise ValueError(f"Waypoint {i} has {len(wp)} values, expected 7.")
        pan, lift, elbow, wflex, wroll = [float(np.degrees(v)) for v in wp[1:6]]
        out.append((
            f"wp{i+1:02d}",
            {"shoulder_pan": pan, "shoulder_lift": lift, "elbow_flex": elbow,
             "wrist_flex": wflex, "wrist_roll": wroll},
        ))
    return out

# Mat visible threshold: FK end-effector must be within this many metres of the
# arm base (x,y plane) to count as "pointing at the mat". Tune to your mat size.
MAT_RADIUS = 0.20


# ── Dry-run angle check ───────────────────────────────────────────────────────

JOINT_NAMES = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll"]
SAFE_LIMIT  = 90.0

def check_scan_angles(waypoints: list[tuple[str, dict]]) -> bool:
    """Print the planned waypoints and ask for confirmation. No arm needed."""
    header = f"{'Position':<22}" + "".join(f"{n:>12}" for n in JOINT_NAMES)
    print("\n" + header)
    print("-" * len(header))

    any_unsafe = False
    for label, wp in waypoints:
        degs = [wp["shoulder_pan"], wp["shoulder_lift"], wp["elbow_flex"],
                wp["wrist_flex"],   wp["wrist_roll"]]
        row = f"{label:<22}"
        for d in degs:
            flag = " !" if abs(d) > SAFE_LIMIT else "  "
            row += f"{d:>10.1f}°{flag}"
        if any(abs(d) > SAFE_LIMIT for d in degs):
            any_unsafe = True
        print(row)

    if any_unsafe:
        print(f"\n  ! = exceeds ±{SAFE_LIMIT:.0f}°")

    print()
    return input("Proceed with scan? [y/N] ").strip().lower() == "y"


# ── Arm scan pattern ──────────────────────────────────────────────────────────

def run_scan_pattern(
    arm: RoboticArm,
    cap: cv2.VideoCapture,
    image_dir: Path,
    waypoints: list[tuple[str, dict]],
) -> list[tuple[str, Pose6D]]:
    """Move arm through waypoints, capture one RGB frame per position."""
    image_dir.mkdir(parents=True, exist_ok=True)
    captures: list[tuple[str, Pose6D]] = []
    gripper = float(arm.robot.get_observation()["gripper.pos"])

    for label, wp in waypoints:
        action = {
            "shoulder_pan.pos":  wp["shoulder_pan"],
            "shoulder_lift.pos": wp["shoulder_lift"],
            "elbow_flex.pos":    wp["elbow_flex"],
            "wrist_flex.pos":    wp["wrist_flex"],
            "wrist_roll.pos":    wp["wrist_roll"],
            "gripper.pos":       gripper,
        }
        arm.robot.send_action(action)
        time.sleep(3)

        pose = arm.get_current_pose()
        ee_dist = float(np.sqrt(pose.x**2 + pose.y**2))
        # if ee_dist > MAT_RADIUS:
        #     print(f"  {label:<22}  → skipped (end-effector {ee_dist:.3f}m from mat centre)")
        #     continue

        name = f"frame_{len(captures):04d}.jpg"
        if _capture_frame(cap, image_dir / name):
            captures.append((name, pose))
            print(f"  {label:<22}  → {name}")
        else:
            print(f"  {label:<22}  → camera read failed, skipping")

    print(f"\n[scan] {len(captures)} frames captured.")
    return captures


# ── Feature preview ───────────────────────────────────────────────────────────

def show_feature_preview(image_dir: Path) -> None:
    """Show SIFT keypoints on each image so you can judge texture quality."""
    sift = cv2.SIFT_create()
    images = sorted(image_dir.glob("*.jpg"))
    print(f"\nFeature preview ({len(images)} images) — any key = next,  q = skip\n")

    for img_path in images:
        img = cv2.imread(str(img_path))
        kps = sift.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
        vis = cv2.drawKeypoints(img, kps, None, color=(0, 220, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        warn = "  LOW TEXTURE" if len(kps) < 300 else ""
        cv2.putText(vis, f"{img_path.name}: {len(kps)} keypoints{warn}",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
        cv2.imshow("Feature Preview", vis)
        print(f"  {img_path.name}: {len(kps)} keypoints{warn}")
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


# ── COLMAP with pose priors ───────────────────────────────────────────────────

def write_prior_model(
    model_dir: Path,
    captures: list[tuple[str, Pose6D]],
    img_w: int,
    img_h: int,
) -> None:
    """
    Write a COLMAP model (cameras/images/points3D txt) using FK poses as priors.
    FK poses are camera-to-world; COLMAP needs world-to-camera.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    focal = 1.2 * max(img_w, img_h)   # COLMAP default heuristic
    cx, cy = img_w / 2.0, img_h / 2.0

    (model_dir / "cameras.txt").write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 SIMPLE_RADIAL {img_w} {img_h} {focal:.2f} {cx:.2f} {cy:.2f} 0.0\n"
    )

    lines = [
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "# POINTS2D[] as (X, Y, POINT3D_ID)\n"
    ]
    for img_id, (name, pose) in enumerate(captures, start=1):
        # FK → camera-to-world; invert to world-to-camera for COLMAP
        R_cw = Rot.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
        t_cw = np.array([pose.x, pose.y, pose.z])
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        q = Rot.from_matrix(R_wc).as_quat()   # [qx, qy, qz, qw] scalar-last
        qw, qx, qy, qz = q[3], q[0], q[1], q[2]
        tx, ty, tz = t_wc
        lines.append(
            f"{img_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
            f"{tx:.9f} {ty:.9f} {tz:.9f} 1 {name}\n\n"
        )

    (model_dir / "images.txt").write_text("".join(lines))
    (model_dir / "points3D.txt").write_text(
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"
    )


def _run(cmd: list[str], step: str) -> None:
    print(f"\n[COLMAP] {step}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("\n".join(result.stderr.strip().splitlines()[-40:]))
        raise RuntimeError(f"COLMAP step failed: {step}")


def _matched_image_names(db_path: Path) -> set[str]:
    """
    Query the COLMAP database for images that appear in at least one verified
    match pair. point_triangulator crashes on images with no correspondences.
    """
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("SELECT image_id, name FROM images")
    id_to_name = {row[0]: row[1] for row in c.fetchall()}
    # pair_id encodes (id1, id2) as id1 * MAX_IMAGE_ID + id2
    MAX_ID = 2147483647
    c.execute("SELECT pair_id FROM two_view_geometries WHERE rows > 0")
    matched: set[str] = set()
    for (pair_id,) in c.fetchall():
        for img_id in (pair_id // MAX_ID, pair_id % MAX_ID):
            if img_id in id_to_name:
                matched.add(id_to_name[img_id])
    conn.close()
    return matched


def run_colmap(
    workspace: Path,
    image_dir: Path,
    captures: list[tuple[str, "Pose6D"]],
    img_w: int,
    img_h: int,
) -> Path:
    db = workspace / "database.db"
    prior_model = workspace / "prior_model"
    out_model = workspace / "triangulated"
    out_model.mkdir(exist_ok=True)

    _run([
        "colmap", "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "0",
    ], "Feature extraction")

    _run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db),
        "--SiftMatching.use_gpu", "0",
    ], "Exhaustive feature matching")

    # Filter to images that actually have matches — point_triangulator crashes
    # with std::out_of_range on any image that has a pose but no correspondences.
    matched = _matched_image_names(db)
    filtered = [(name, pose) for name, pose in captures if name in matched]
    dropped = len(captures) - len(filtered)
    if dropped:
        print(f"  Dropping {dropped} image(s) with no feature matches from prior model.")
    write_prior_model(prior_model, filtered, img_w, img_h)

    _run([
        "colmap", "point_triangulator",
        "--database_path", str(db),
        "--image_path", str(image_dir),
        "--input_path", str(prior_model),
        "--output_path", str(out_model),
    ], "Point triangulation (FK poses as priors)")

    # Check if triangulation produced anything; if not, fall back to mapper
    # (the FK poses don't include the camera-to-wrist offset, so they're often
    # off by a few cm — mapper estimates poses from features instead).
    points_bin = out_model / "points3D.bin"
    if not points_bin.exists() or points_bin.stat().st_size < 100:
        print("\n[COLMAP] Triangulation produced no points — falling back to SfM mapper.")
        sparse_dir = workspace / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        _run([
            "colmap", "mapper",
            "--database_path", str(db),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir),
        ], "SfM mapper (fallback)")
        # Pick largest model
        models = sorted(sparse_dir.iterdir())
        if not models:
            raise RuntimeError("Mapper produced no models. Try adding texture to the scene.")
        out_model = max(models, key=lambda p: (p / "images.bin").stat().st_size
                        if (p / "images.bin").exists() else 0)
        print(f"  Using mapper model: {out_model}")

    ply_out = workspace / "triangulated.ply"
    _run([
        "colmap", "model_converter",
        "--input_path", str(out_model),
        "--output_path", str(ply_out),
        "--output_type", "PLY",
    ], "Exporting point cloud")

    return ply_out


# ── Mesh ──────────────────────────────────────────────────────────────────────

def poisson_mesh(ply_path: Path, out_path: str) -> None:
    print(f"\nLoading point cloud...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    n_pts = len(pcd.points)
    print(f"  {n_pts} points")

    if n_pts < 100:
        raise RuntimeError(
            "Too few points for meshing — check object texture and arm coverage."
        )

    pcd = pcd.voxel_down_sample(0.005)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    print("Running Poisson reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    keep = np.asarray(densities) > np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(~keep)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()

    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"Saved: {out_path}  ({len(mesh.vertices)} verts, {len(mesh.triangles)} tris)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=None,
                        help="Camera index (e.g. 2 for /dev/video2)")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--urdf", default="models/so101.urdf")
    parser.add_argument("--out", default="manual_scan_mesh.ply")
    parser.add_argument("--waypoints", default="waypoints.json",
                        help="Path to waypoints JSON (default: waypoints.json)")
    parser.add_argument("--keep-ws", action="store_true")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip SIFT feature preview after capture")
    args = parser.parse_args()

    waypoints_path = Path(args.waypoints)
    if not waypoints_path.exists():
        print(f"Error: waypoints file not found: {waypoints_path}")
        sys.exit(1)
    waypoints = load_waypoints(waypoints_path)
    print(f"Loaded {len(waypoints)} waypoints from {waypoints_path}")

    if shutil.which("colmap") is None:
        print("Error: 'colmap' not found in PATH.")
        sys.exit(1)

    timestamp = int(time.time())
    scan_dir = Path.cwd() / "scans" / f"scan_{timestamp}"
    image_dir = scan_dir / "images"
    scan_dir.mkdir(parents=True)
    workspace = Path.cwd() / f"scan_workspace_{timestamp}"
    workspace.mkdir()
    print(f"Captures will be saved to: {scan_dir}")
    print(f"COLMAP workspace:         {workspace}\n")

    arm: RoboticArm | None = None
    cap: cv2.VideoCapture | None = None

    try:
        print("Connecting to arm...")
        arm = RoboticArm(port=args.port, urdf_path=args.urdf)

        cap, cam_idx = open_camera(args.device)
        print(f"Using camera /dev/video{cam_idx}")

        # Measure actual frame size
        _, test_frame = cap.read()
        img_h, img_w = test_frame.shape[:2]

        print("\nComputing planned joint angles (no movement yet)...")
        if not check_scan_angles(waypoints):
            print("Aborted.")
            return

        captures = run_scan_pattern(arm, cap, image_dir, waypoints)

        # Persist poses next to the images so the scan can be replayed through
        # debug_colmap.py without touching the arm.
        poses_data = [
            {"name": name, "x": p.x, "y": p.y, "z": p.z,
             "qx": p.qx, "qy": p.qy, "qz": p.qz, "qw": p.qw,
             "img_w": img_w, "img_h": img_h}
            for name, p in captures
        ]
        (scan_dir / "poses.json").write_text(json.dumps(poses_data, indent=2))
        print(f"Saved {len(captures)} poses to {scan_dir / 'poses.json'}")

        if len(captures) < 5:
            raise RuntimeError(f"Too few captures ({len(captures)}), aborting.")

        if not args.no_preview:
            show_feature_preview(image_dir)

        ply_path = run_colmap(workspace, image_dir, captures, img_w, img_h)
        poisson_mesh(ply_path, args.out)

    except RuntimeError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        if cap is not None:
            cap.release()
        if arm is not None:
            arm.move_to_home()
        if args.keep_ws:
            print(f"Workspace kept at: {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
