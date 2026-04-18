# GRANT

A single-arm 3D scanning pipeline. One 6-DoF arm carries both a stereo depth camera and a suction gripper, so the same arm that sweeps the camera around an object also flips the object over between scans. The gripper is mounted so it's out of frame whenever the camera is aimed at the object — no arm-segmentation step is needed.

The reconstruction path (`registration.py`, `orchestrator.py`) is fully implemented. Hardware-facing modules are interface stubs.

---

## Algorithm overview

```
                       ┌────────────────────────┐
                       │   ScanOrchestrator     │
                       │   (orchestrator.py)    │
                       └───────────┬────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
     ┌──────────────┐       ┌─────────────┐        ┌──────────────┐
     │ RoboticArm   │       │   Vision    │        │Registration  │
     │ (camera +    │       │   (KV260    │        │ (align +     │
     │  gripper)    │       │    stereo)  │        │   fuse)      │
     └──────────────┘       └──────┬──────┘        └──────┬───────┘
                                   │                      │
                                   ▼                      ▼
                              RGBDFrame +           Mesh (TSDF)
                              CapturedView
```

Types live in [scan_types.py](scan_types.py): `Pose6D`, `JointAngles`, `RGBDFrame`, `PointCloud`, `ObjectState`, `CameraIntrinsics`, `CapturedView`, `ScanResult`, `ScanError`.

> ⚠️ The module is `scan_types`, **not** `types` — `types` shadows Python's stdlib and breaks the interpreter.

---

## File map

| File | Role | Status |
| --- | --- | --- |
| [orchestrator.py](orchestrator.py) | `ScanOrchestrator.run_full_scan()`. Drives the fixed 4-arc scan + flip + alignment + TSDF. | **Implemented** |
| [registration.py](registration.py) | Per-view transforms, ICP merge, global RANSAC+ICP align, TSDF fusion, Poisson fallback. | **Implemented** |
| [scripts/test_mesh_reconstruction.py](scripts/test_mesh_reconstruction.py) | Synthetic end-to-end test (raycast depth + simulated flip). No hardware required. | **Implemented** |
| [scan_types.py](scan_types.py) | Shared dataclasses used at every seam. | **Implemented** |
| [interfaces/robotic_arm.py](interfaces/robotic_arm.py) | The single arm. Camera sweep (`get_arc_trajectory`, `move_to_pose`) and gripping (`pickup_object`, `flip_object`, `release_object`) on one interface. | Stub |
| [interfaces/vision.py](interfaces/vision.py) | KV260 stereo capture + depth backprojection. | Stub |
| [coverage.py](coverage.py) | Not used in the fixed algorithm. Retained as a design reference. | Legacy |

### Single-arm tradeoffs

What we gain: simpler hardware, no second-arm coordination.

What we lose: no independent seal-detection sensor during pickup. The arm still commands the suction cup, but `GripResult.success` reports only whether the approach+grip motion executed — it can't confirm that the cup actually sealed. We rely on the FK-based approach pose to land accurately, and on the next capture to reveal a failed grip (the object will have moved unexpectedly).

---

## The scan flow ([orchestrator.py](orchestrator.py))

**Phase 1 — Detect object**: `vision.capture_rgbd()` + `vision.detect_object(frame)` → `ObjectState` (centroid + bbox). The object sits on the table — it is **not** held during capture.

**Phase 2 — Orientation 1**: `_run_both_arcs(object_state)`
1. `arm.get_arc_trajectory("azimuth", …)` → left-to-right sweep.
2. `arm.get_arc_trajectory("elevation", …)` → top-to-bottom sweep.
3. For each pose: `arm.move_to_pose`, then `vision.capture_rgbd`, stored as `CapturedView(frame, pose)`.

**Phase 3 — Flip**: `_flip_object_in_place(centroid)`
1. `arm.pickup_object(target_pose=centroid)` — FK-approach + suction.
2. `arm.flip_object()` — rotates ~180° about a horizontal axis and sets the object back on the table.
3. `arm.release_object()`.

**Phase 4 — Re-detect**: the flip won't land the object at exactly the same spot, so `detect_object` runs again to pick up the new centroid.

**Phase 5 — Orientation 2**: identical to Phase 2, new centroid.

**Phase 6 — Home**: `arm.move_to_home()`.

**Phase 7 — Cross-orientation alignment**
1. Fuse each orientation's views into a single world-space cloud (`transform_to_world` + ICP-refined `merge`).
2. `registration.global_align(source=cloud_2, target=cloud_1)` → `(T, fitness)`. RANSAC on FPFH features finds a coarse alignment across the ~180° flip, then ICP refines on the full clouds.
3. Abort with `ScanError` if `fitness < MIN_ALIGNMENT_FITNESS` (0.3) — too little surface overlap.
4. `apply_transform_to_views(views_orient_2, T)` rewrites every orientation-2 pose into orientation-1's frame.

**Phase 8 — TSDF fusion**: `registration.tsdf_fuse(all_views)` integrates every frame's depth into a `ScalableTSDFVolume` and extracts the mesh. Returns `ScanResult`.

### Why RANSAC + ICP + TSDF

- **RANSAC** handles the ~180° flip. ICP started from identity would lock onto a local minimum.
- **ICP** refines RANSAC's cm-level output to sub-mm on the full-resolution clouds.
- **TSDF** weights each view by viewing angle and distance automatically and produces a watertight mesh without Poisson's normal-orientation headaches.

---

## Tuning knobs

### `ScanOrchestrator`
| Constant | Meaning | Default |
| --- | --- | --- |
| `ARC_STEPS_AZIMUTH` | Frames per left→right sweep. | `12` |
| `ARC_STEPS_ELEVATION` | Frames per top→bottom sweep. | `8` |
| `MIN_ALIGNMENT_FITNESS` | Abort threshold for cross-orientation alignment. | `0.3` |

### `Registration`
| Constant | Effect |
| --- | --- |
| `VOXEL_SIZE` | Downsample grid after each merge. 2mm; tighter = more detail + memory. |
| `ICP_MAX_DIST` | ICP correspondence threshold. Bigger than FK drift, smaller than object features. |
| `GLOBAL_VOXEL_SIZE` | Downsample for FPFH feature extraction. 5mm. |
| `RANSAC_MAX_CORR_DIST` | RANSAC inlier threshold (`1.5 × GLOBAL_VOXEL_SIZE`). |
| `TSDF_VOXEL` | Output mesh resolution. 2mm. |
| `TSDF_TRUNC` | Signed-distance truncation band (`4 × TSDF_VOXEL`). |
| `TSDF_DEPTH_TRUNC` | Ignore depth past this (meters). |
| `POISSON_DEPTH` | Octree depth for Poisson fallback. 8 fast / 10 slow. |

---

## Running the reconstruction pipeline

Minimal real-hardware harness (raises at the first stub call — that's expected):

```python
from GRANT.orchestrator import ScanOrchestrator
from GRANT.registration import Registration
from GRANT.interfaces.robotic_arm import RoboticArm
from GRANT.interfaces.vision import VisionSystem

orch = ScanOrchestrator(
    arm=RoboticArm(),
    vision=VisionSystem(),
    registration=Registration(),
)
result = orch.run_full_scan()
print(f"{result.n_frames} frames, alignment fitness {result.alignment_fitness:.2f}")
```

---

## Running the synthetic test

`scripts/test_mesh_reconstruction.py` runs the full pipeline on raycast synthetic data. **Three invocations all work** — pick whichever fits where you're standing in the shell:

```bash
# From inside GRANT/
python scripts/test_mesh_reconstruction.py

# From GRANT's parent directory
python GRANT/scripts/test_mesh_reconstruction.py
python -m GRANT.scripts.test_mesh_reconstruction
```

Example runs:

```bash
python scripts/test_mesh_reconstruction.py --shape bunny
python scripts/test_mesh_reconstruction.py --shape box --n-az 16 --n-el 10
python scripts/test_mesh_reconstruction.py --shape armadillo --visualize
python scripts/test_mesh_reconstruction.py --no-tsdf       # use Poisson instead of TSDF
```

Flags: `--noise-std` (m), `--fk-drift-std` (m), `--flip-drift-std` (m), `--radius` (m), `--n-az`, `--n-el`, `--seed`.

### Common import errors

| Error | Fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'open3d'` | `pip install open3d` (and `numpy` if you don't have it). |
| `ModuleNotFoundError: No module named 'GRANT'` | You ran `python -m GRANT.…` from the wrong directory — you need to be in GRANT's *parent* directory. Or just use `python scripts/test_mesh_reconstruction.py` from inside GRANT/. |
| `ImportError: attempted relative import with no known parent package` | Shouldn't happen anymore — the script has a `sys.path` shim at the top. If it still does, check that `GRANT/__init__.py` and `GRANT/scripts/__init__.py` both exist (they should be empty files). |
| `ImportError: cannot import name 'MappingProxyType' from partially initialized module 'types'` | You have a file named `types.py` somewhere — it's shadowing Python's stdlib. Ours is called `scan_types.py` for exactly this reason. |

### Benchmarks

On a 10cm object with 0.8mm sensor noise and 2mm FK drift:

| Shape | TSDF | Poisson (`--no-tsdf`) |
| --- | --- | --- |
| Bunny | ~3.2mm chamfer | ~1.7mm |
| Box | ~3.2mm | — |
| Sphere | ~3.5mm | — |

TSDF bottoms out at roughly the FK-drift scale — expected. For tight FK, TSDF wins on larger / more-featured objects; for noisy FK, the ICP-refined Poisson path is more forgiving on small ones.
