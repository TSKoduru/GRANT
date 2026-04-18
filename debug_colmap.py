"""
Re-run COLMAP + mesh on an existing scan directory, without touching the arm.

A scan directory has the structure produced by manual_scan.py:
    scans/scan_<timestamp>/
        images/
            frame_0000.jpg
            ...
        poses.json          # FK poses written during capture

Usage:
    python debug_colmap.py scans/scan_1745000000 [--out mesh.ply] [--keep-ws]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from scan_types import Pose6D
from manual_scan import run_colmap, poisson_mesh, show_feature_preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scan_dir", help="Path to scans/scan_<timestamp>/")
    parser.add_argument("--out", default="manual_scan_mesh.ply")
    parser.add_argument("--keep-ws", action="store_true",
                        help="Keep COLMAP workspace after reconstruction")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip SIFT feature preview before COLMAP")
    args = parser.parse_args()

    if shutil.which("colmap") is None:
        print("Error: 'colmap' not found in PATH.")
        sys.exit(1)

    scan_dir = Path(args.scan_dir)
    image_dir = scan_dir / "images"
    poses_path = scan_dir / "poses.json"

    if not image_dir.is_dir():
        print(f"Error: {image_dir} does not exist."); sys.exit(1)
    if not poses_path.exists():
        print(f"Error: {poses_path} does not exist."); sys.exit(1)

    poses_data = json.loads(poses_path.read_text())
    if not poses_data:
        print("Error: poses.json is empty."); sys.exit(1)

    captures: list[tuple[str, Pose6D]] = [
        (d["name"], Pose6D(x=d["x"], y=d["y"], z=d["z"],
                           qx=d["qx"], qy=d["qy"], qz=d["qz"], qw=d["qw"]))
        for d in poses_data
    ]
    img_w = poses_data[0]["img_w"]
    img_h = poses_data[0]["img_h"]
    print(f"Loaded {len(captures)} poses; image size {img_w}×{img_h}")

    workspace = Path.cwd() / f"scan_workspace_{int(time.time())}"
    workspace.mkdir()
    print(f"COLMAP workspace: {workspace}\n")

    try:
        if not args.no_preview:
            show_feature_preview(image_dir)

        ply_path = run_colmap(workspace, image_dir, captures, img_w, img_h)
        poisson_mesh(ply_path, args.out)

    except RuntimeError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        if args.keep_ws:
            print(f"Workspace kept at: {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
