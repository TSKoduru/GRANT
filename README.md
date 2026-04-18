# GRANT

A two-armed 3D scanning pipeline. One arm **holds** the object, the other arm **looks** at it. The orchestrator moves the viewing arm around, captures RGB-D frames, fuses them into a point cloud, and runs Poisson reconstruction to produce a mesh. It keeps going until coverage crosses a threshold or information gain flattens out.

Everything in this repo is currently **interface stubs**. No method has a body — the shapes are frozen so the subsystems can be implemented in parallel against the same contract.

---

## How the pieces fit together

```
                       ┌────────────────────────┐
                       │   ScanOrchestrator     │
                       │   (orchestrator.py)    │
                       └───────────┬────────────┘
                                   │ drives
     ┌──────────────┬──────────────┼──────────────┬──────────────┐
     │              │              │              │              │
     ▼              ▼              ▼              ▼              ▼
┌─────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────┐
│CameraArm│   │PickupArm │   │ Vision   │   │ Coverage  │   │Registration  │
│(looks)  │   │ (holds)  │   │ (KV260)  │   │   Map     │   │ (world-space │
│         │   │          │   │          │   │           │   │   fusion)    │
└─────────┘   └──────────┘   └────┬─────┘   └─────┬─────┘   └──────┬───────┘
                                  │               │                │
                                  ▼               ▼                ▼
                              RGBDFrame   (picks next pose)   PointCloud
                                  │                                │
                                  └──────────┬─────────────────────┘
                                             ▼
                                      ┌──────────────┐
                                      │ArmSegmenter  │
                                      │(masks pickup │
                                      │ arm pixels)  │
                                      └──────────────┘
```

Data types used at every seam live in [scan_types.py](scan_types.py): `Pose6D`, `JointAngles`, `RGBDFrame`, `PointCloud`, `ObjectState`, `CameraIntrinsics`, `ScanResult`, `ScanError`.

> ⚠️ The module is `scan_types`, **not** `types` — `types` shadows Python's stdlib and breaks the interpreter.

---

## File map

| File | Role |
| --- | --- |
| [orchestrator.py](orchestrator.py) | `ScanOrchestrator.run_full_scan()`. Entry point. Owns the phase loop. |
| [interfaces/camera_arm.py](interfaces/camera_arm.py) | Viewing arm: moves the stereo rig, reports joint/pose state. |
| [interfaces/pickup_arm.py](interfaces/pickup_arm.py) | Holding arm: suction pickup, wrist rotate, release. |
| [interfaces/vision.py](interfaces/vision.py) | KV260 stereo capture → `RGBDFrame`; depth backprojection → `PointCloud`. |
| [coverage.py](coverage.py) | Elevation/azimuth sphere tracking which patches of surface have been seen. Picks next viewpoint + next object rotation. |
| [registration.py](registration.py) | Camera → world transform, ICP merge, Poisson mesh reconstruction. |
| [arm_segmenter.py](arm_segmenter.py) | Removes pickup-arm pixels from each frame (geometric → color → depth refinement). |
| [scan_types.py](scan_types.py) | Shared dataclass stubs used by every module. |

---

## The scan loop ([orchestrator.py:32](orchestrator.py#L32))

**Phase 1 — Initialization**
1. `vision.capture_rgbd()` → first frame
2. `vision.detect_object(frame)` → `ObjectState` (centroid + bbox)
3. `pickup_arm.pickup_object(centroid_as_pose)` → grip. If it fails, raise `ScanError`.

**Phase 2 — Capture loop**, until either `COVERAGE_TARGET` is met, `MAX_ITERATIONS` is hit, or expected gain drops below `MIN_NEW_COVERAGE`:
1. `coverage_map.get_next_object_rotation()` — rotate the object if the best unseen regions face away.
2. `coverage_map.get_next_viewpoint(object_state, camera_arm.get_reachable_poses())` — pick the pose with highest expected information gain.
3. `camera_arm.move_to_pose(next_camera_pose)`
4. `vision.capture_rgbd()` → frame
5. `arm_segmenter.get_mask(frame, pickup_arm.get_joint_angles(), camera_arm.get_current_pose())` — write `frame.arm_mask` so the holding arm's pixels don't pollute the cloud.
6. `vision.frame_to_pointcloud(frame)` → camera-space cloud
7. `registration.transform_to_world(cloud, camera_pose)` → world-space cloud
8. `registration.merge(accumulated, new_cloud)` — ICP-refined fusion
9. `coverage_map.update(frame, camera_pose, object_state)`

**Phase 3 — Finalize**
1. `pickup_arm.rotate_to_angle(0.0)`, `release_object()`
2. `camera_arm.move_to_home()`
3. `registration.reconstruct_mesh(accumulated_cloud)` → Poisson mesh
4. Return `ScanResult(mesh, point_cloud, coverage_achieved, n_frames)`

---

## Tuning knobs (class constants on `ScanOrchestrator`)

| Constant | Meaning | Default |
| --- | --- | --- |
| `COVERAGE_TARGET` | Stop once this fraction of the surface sphere is observed. | `0.92` |
| `MAX_ITERATIONS` | Hard cap on capture attempts per scan. | `24` |
| `MIN_NEW_COVERAGE` | Break early if the best next viewpoint adds less than this. | `0.02` |

---

## Implementing a module

1. Pick a file from the table above.
2. Keep the existing signatures — the orchestrator (and other modules) rely on them.
3. Import types from `.scan_types`, not `.types`.
4. Parameter order and names are part of the contract, since [orchestrator.py](orchestrator.py) calls several of them as keyword arguments (e.g. `target_pose=`, `camera_pose=`, `joint_angles=`).

### Recommended implementation order

1. **`scan_types.py`** — already fleshed out; tighten fields as you need them.
2. **`interfaces/vision.py`** — nothing else is testable without frames.
3. **`interfaces/camera_arm.py`** and **`interfaces/pickup_arm.py`** — hardware drivers.
4. **`registration.py`** — pure function of `PointCloud` + `Pose6D`, easy to unit-test with synthetic clouds.
5. **`arm_segmenter.py`** — needs URDF + calibrated HSV range. `calibrate_arm_color()` at startup, once, with no object present.
6. **`coverage.py`** — last, because its quality depends on the others working.
7. **`orchestrator.py`** — already wired up; mostly untouched once the above exist.

---

## Running it

The package is importable from its parent directory:

```bash
cd /path/to/projects
python -c "from GRANT.orchestrator import ScanOrchestrator"
```

A minimal harness looks like:

```python
from GRANT.orchestrator import ScanOrchestrator
from GRANT.coverage import CoverageMap
from GRANT.registration import Registration
from GRANT.arm_segmenter import ArmSegmenter
from GRANT.interfaces.camera_arm import CameraArm
from GRANT.interfaces.pickup_arm import PickupArm
from GRANT.interfaces.vision import VisionSystem

orch = ScanOrchestrator(
    camera_arm=CameraArm(),
    pickup_arm=PickupArm(),
    vision=VisionSystem(),
    coverage_map=CoverageMap(),
    registration=Registration(),
    arm_segmenter=ArmSegmenter(arm_urdf_path="...", camera_intrinsics=...),
)
result = orch.run_full_scan()
```

This will raise at the first stub call — the interfaces compile but have no bodies yet. That's the expected state while subsystems are being built out.
