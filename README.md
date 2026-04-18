# GRANT

A single-arm 3D scanning system. One 6-DoF arm carries a stereo depth camera and a suction gripper. For each scan the arm sweeps a **cross pattern parallel to the table** (one pass along X, one along Y), flips the object with the gripper, then sweeps the cross a second time. The two orientations are stitched with RANSAC+ICP and fused into a mesh via TSDF.

The reconstruction pipeline (`registration.py`, `orchestrator.py`) is implemented. Hardware-facing modules have real scaffolding where possible and stubs where not.

---

## Compute topology

```
 ┌─────────────────────────────┐   depth frames,         ┌─────────────────────────────┐
 │           KV260             │   enhanced              │            AI PC            │
 │                             │────────────────────────▶│                             │
 │ • Stereo capture + align    │                         │ • Arm control               │
 │ • Depth enhancement per-    │   coverage updates      │ • Point cloud stitching     │
 │   frame (bilateral, specks, │◀────────────────────────│ • RANSAC+ICP align          │
 │   hole fill)                │                         │ • TSDF / Poisson mesh       │
 │ • Live coverage heatmap     │                         │ • FastAPI web server        │
 └─────────────────────────────┘                         └─────────┬───────────────────┘
                                                                   │ mesh + scan status
                                                                   │
                                                                   ▼
                        ┌──────────────────────────────────────────────────────┐
                        │                    ESP32-S3                          │
                        │                                                      │
                        │ • WiFi AP the user connects to                       │
                        │ • Reverse-proxy /scan/* to AI PC                     │
                        │ • Onshape completion webhook (holds API credentials) │
                        └──────────────────────────┬───────────────────────────┘
                                                   │ HTTPS
                                                   ▼
                        ┌──────────────────────────────────────────────────────┐
                        │                  Web dashboard                       │
                        │                                                      │
                        │ • Start-scan button  • Live coverage heatmap         │
                        │ • Progress + phase   • Mesh preview + Onshape link   │
                        └──────────────────────────────────────────────────────┘
```

Data types are shared across boards via [scan_types.py](scan_types.py).

> ⚠️ The module is `scan_types`, not `types` — `types` shadows Python's stdlib and breaks the interpreter.

---

## File map

| Path | Role | Status |
| --- | --- | --- |
| [orchestrator.py](orchestrator.py) | `ScanOrchestrator.run_full_scan()`. Drives the cross-sweep × 2 + flip + align + TSDF. | **Implemented** |
| [registration.py](registration.py) | Per-view transforms, ICP merge, global RANSAC+ICP, TSDF fusion, Poisson fallback. | **Implemented** |
| [scan_types.py](scan_types.py) | Shared dataclasses: `Pose6D`, `RGBDFrame`, `PointCloud`, `CapturedView`, etc. | **Implemented** |
| [interfaces/robotic_arm.py](interfaces/robotic_arm.py) | Single arm: `move_to_pose`, `get_arc_trajectory(axis="x"/"y")`, `pickup_object`, `flip_object`. | Stub |
| [interfaces/vision.py](interfaces/vision.py) | DepthAI v3 pipeline — stereo capture + depth alignment to RGB. `frame_to_pointcloud` implemented, `detect_object` TODO. | Partial |
| [kv260/depth_enhancement.py](kv260/depth_enhancement.py) | Bilateral + speckle + hole-fill depth cleanup. Runs on the KV260. | **Implemented** |
| [kv260/coverage_heatmap.py](kv260/coverage_heatmap.py) | Viewing-sphere coverage tracker, emits a colored PNG. | **Implemented** |
| [webserver/server.py](webserver/server.py) | FastAPI app: `/scan/start`, `/scan/status`, `/scan/heatmap`, `/scan/mesh`, serves dashboard. | **Implemented** |
| [webserver/scan_session.py](webserver/scan_session.py) | Threaded wrapper around `ScanOrchestrator` + heatmap + snapshot state. | **Implemented** |
| [webserver/onshape.py](webserver/onshape.py) | Completion webhook — forwards to ESP32 or (stub) direct Onshape API. | Partial |
| [dashboard/index.html](dashboard/index.html) | Self-contained page: start button, status, heatmap, mesh link. | **Implemented** |
| [esp32/README.md](esp32/README.md) + [esp32/firmware_sketch.ino](esp32/firmware_sketch.ino) | ESP32-S3 bridge: WiFi AP, HTTP proxy, Onshape uploader. | Stub |
| [scripts/test_mesh_reconstruction.py](scripts/test_mesh_reconstruction.py) | Synthetic end-to-end test (raycast depth + cross sweep + flip). No hardware required. | **Implemented** |

---

## The scan flow

**Phase 1 — Detect**: `vision.capture_rgbd()` + `vision.detect_object(frame)` → `ObjectState`. Object sits on the table, not held.

**Phase 2 — Orientation 1**: `_run_both_arcs(object_state, phase="orient-1")`. Two legs of a cross parallel to the table:
- `arm.get_arc_trajectory("x", target, n)` — camera moves along table X, aiming down at the object.
- `arm.get_arc_trajectory("y", target, n)` — same along Y.

At each step: `move_to_pose` → `time.sleep(SETTLE_TIME_S)` → `vision.capture_rgbd` → store as `CapturedView(frame, pose)`. The sleep gives the camera time to de-smear after the move.

**Phase 3 — Flip**: `pickup_object` (no seal verification — we trust FK) → `flip_object` (~180° about a horizontal axis, set back down) → `release_object`.

**Phase 4 — Re-detect**: `detect_object` again to pick up the new centroid.

**Phase 5 — Orientation 2**: same cross sweep, new centroid.

**Phase 6 — Home**: `arm.move_to_home()`.

**Phase 7 — Cross-orientation alignment**: `registration.global_align(cloud_2, cloud_1)` → RANSAC on FPFH features (ICP from identity can't converge across the flip) then ICP refinement. Below fitness 0.3 it raises `ScanError`.

**Phase 8 — TSDF fusion**: every frame (orientation-2 poses transformed by the alignment) goes into a `ScalableTSDFVolume`. Extract and return the mesh.

### Tuning knobs

| Class | Constant | Effect |
| --- | --- | --- |
| `ScanOrchestrator` | `ARC_STEPS_X` / `ARC_STEPS_Y` | Frames per leg of the cross. |
| `ScanOrchestrator` | `SETTLE_TIME_S` | Post-move dwell (default 0.15s). Increase if depth looks smeared. |
| `ScanOrchestrator` | `MIN_ALIGNMENT_FITNESS` | Abort threshold (0.3). |
| `Registration` | `VOXEL_SIZE`, `ICP_MAX_DIST` | Local fusion. |
| `Registration` | `GLOBAL_VOXEL_SIZE`, `RANSAC_*` | Cross-orientation alignment. |
| `Registration` | `TSDF_VOXEL`, `TSDF_TRUNC` | Output mesh resolution. |
| `DepthEnhancer` | `DEPTH_MIN/MAX`, `MIN_SPECKLE_SIZE` | Per-frame depth cleanup. |
| `CoverageHeatmap` | `N_LAT`, `N_LON`, `SPLAT_RADIUS_CELLS` | Heatmap granularity. |

---

## Running the web server + dashboard

Install runtime deps:

```bash
pip install open3d numpy opencv-python fastapi uvicorn pillow requests
```

Start the server from the repo's parent directory:

```bash
uvicorn GRANT.webserver.server:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/ — the dashboard loads, click **Start Scan**. The scan runs in a background thread; the dashboard polls `/scan/status` every 750ms and refreshes `/scan/heatmap` as views come in.

When running on the real hardware, users connect to the ESP32's WiFi AP, open the same dashboard via the ESP32's reverse proxy, and get the same UI.

---

## Running the synthetic test (no hardware)

The test script raycasts a ground-truth mesh to produce fake RGBD frames for the cross sweep and simulates the flip.

```bash
# From inside GRANT/
python scripts/test_mesh_reconstruction.py
python scripts/test_mesh_reconstruction.py --shape bunny --n-x 14 --n-y 14
python scripts/test_mesh_reconstruction.py --shape box --visualize
python scripts/test_mesh_reconstruction.py --no-tsdf      # Poisson instead of TSDF

# From GRANT's parent
python -m GRANT.scripts.test_mesh_reconstruction
```

Flags: `--noise-std` (m), `--fk-drift-std` (m), `--flip-drift-std` (m), `--half-length` (m, cross leg half-length), `--height` (m, camera above object), `--n-x`, `--n-y`, `--seed`, `--visualize`, `--no-tsdf`.

**Recent run:** bunny, cross sweep 14×14, 0.8mm noise + 2mm FK drift → fitness 0.90, 177° rotation recovered, **2.94mm chamfer**.

### Common import errors

| Error | Fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'open3d'` (or `fastapi`, `cv2`, `PIL`) | `pip install open3d fastapi uvicorn opencv-python pillow requests` |
| `ModuleNotFoundError: No module named 'GRANT'` | You ran `python -m GRANT.…` from the wrong directory — be in GRANT's *parent*, or use `python scripts/...` from inside GRANT. |
| `ImportError: attempted relative import with no known parent package` | The test script has a `sys.path` shim. If it still fires, check that `GRANT/__init__.py` and `GRANT/scripts/__init__.py` exist (empty files are fine). |
| `ImportError: cannot import name 'MappingProxyType' from partially initialized module 'types'` | You have a `types.py` shadowing the stdlib. Ours is `scan_types.py` on purpose. |
| `dai.Platform.RVC4` AttributeError | Your DepthAI is older than v3. `pip install -U depthai`. |

### Cross sweep caveat (important)

A cross parallel to the table only observes the object from above, at oblique angles from two perpendicular directions. The flip is essential — without it the bottom of the object never gets captured. RANSAC overlap between orientations comes from the object's *sides*, which both orientations see at grazing angles. If the object is nearly-symmetric under a 180° flip about X, alignment fitness will still be high but the recovered rotation may lock onto a symmetry-equivalent transform; this is fine for reconstruction but surprising when debugging.

# ROADMAP

1. Install / set up

On the AI PC:

pip install open3d numpy opencv-python fastapi uvicorn pillow requests depthai

Connect the OAK-D over USB (or PoE if that's your model), the arm's control interface (USB serial, Ethernet, or whatever the driver needs), and verify both separately before combining.

On the KV260: Petalinux + the Vitis AI runtime you'll use for depth enhancement. The Python code in kv260/depth_enhancement.py is the reference — you'll port those three stages (bilateral / speckle / hole-fill) to the KV260's accelerator.

On the ESP32-S3: Arduino IDE + ESP32 board package v3.x, libraries listed in esp32/README.md. Flash esp32/firmware_sketch.ino, then seed SSID, PSK, AI_PC_URL, and Onshape creds via serial.

2. Code you still need to write (blocker order)

🔴 Blocks the scan entirely

interfaces/robotic_arm.py	
    Every method is a stub. Implement move_to_pose, get_current_pose, get_joint_angles, move_to_home, get_arc_trajectory("x" | "y", target, n), pickup_object, release_object, flip_object. get_arc_trajectory is pure math — produce a list of Pose6D along a line ±half_length from target at fixed height, camera pointing down at the target. The rest are hardware drivers for whatever arm you're using.

interfaces/vision.py	
    detect_object currently raises NotImplementedError. Suggested baseline is in the docstring — depth-threshold the table plane, connected-components, pick the largest blob, return its centroid + bbox as an ObjectState. Without this, the orchestrator aborts in Phase 1.

🟡 Wiring gap (silent but broken)
The depth enhancer isn't called yet. Two options, pick one:

Apply it inside VisionSystem.capture_rgbd() before returning.

Or let the orchestrator apply it: add self.enhancer = DepthEnhancer() to ScanOrchestrator.__init__, then in _capture_at do frame = self.enhancer.process(frame) before building the CapturedView.

I'd put it in VisionSystem — the rest of the pipeline shouldn't need to know depth arrives pre-cleaned.

🟢 Polish, not blocking
webserver/onshape.py _post_direct_stub — implement if you want to skip the ESP32 bridge during dev. Requires HMAC request signing.

esp32/firmware_sketch.ino — two TODOs: the reverse-proxy is synchronous (blocks the event loop); the /onshape handler is a 501. Hackathon-acceptable as-is but not production.

3. Tuning (values I had to guess — verify on hardware)

SETTLE_TIME_S	orchestrator.py:51	0.15s	
    Capture a frame at rest vs. immediately after a move — if the "moving" frame looks smeared or ghosted, increase.

ARC_STEPS_X / _Y	orchestrator.py:49-50	12 each	
    More = better reconstruction + slower scan. 12 gives ~8.3° spacing across a ±18cm sweep at 22cm height — reasonable start.

Cross geometry (half_length, height)	Passed into get_arc_trajectory	0.18m / 0.22m in the test	
    Set by your arm's reach + the camera's FOV and min focus distance. Check a single capture shows the whole object.

ICP_MAX_DIST	registration.py:19	1cm	
    Measure your arm's repeatability — this should exceed FK drift and stay smaller than the smallest feature you care about.

TSDF_VOXEL / TSDF_TRUNC	registration.py:43-44	2mm / 8mm	
    If output mesh looks blocky, shrink TSDF_VOXEL. If it has holes, grow TSDF_TRUNC.

TSDF_DEPTH_TRUNC	registration.py:45	0.5m	
    Distance beyond which depth readings are discarded. Set to a hair more than height + half_length.

DEPTH_MIN / DEPTH_MAX	kv260/depth_enhancement.py:20-21	0.10m / 1.50m	
    Match the camera's working distance.

MIN_ALIGNMENT_FITNESS	orchestrator.py:52	0.30	
    If real scans fail alignment despite looking correct, lower it. Below 0.2 you probably don't have enough side-surface overlap across the flip.

4. Suggested testing sequence

Synthetic smoke test (no hardware): python scripts/test_mesh_reconstruction.py --shape bunny — should finish in ~10s, chamfer < 5mm. Confirms registration + TSDF still work.

Camera-only: instantiate VisionSystem() in a REPL, call capture_rgbd(), save the depth map as a grayscale PNG. Verify the object shows clearly.

Camera + enhancer: same, but run the frame through DepthEnhancer().process() — compare before/after PNGs for speckle removal.

Arm-only: manually drive the arm to a few poses, confirm get_current_pose() matches what you commanded within a few mm.

Arc trajectory: call arm.get_arc_trajectory("x", object_pose, 5) and step through the returned poses manually — check the camera is pointing at the object at each.

Full orchestrator, mock vision: stub VisionSystem.detect_object with a hardcoded centroid, run ScanOrchestrator.run_full_scan(). Confirms the arm sequence end-to-end.

Full scan with real detect_object: once detect_object works, repeat — chamfer vs. a known-size test object (e.g. a 3D-printed cube).

Web dashboard: uvicorn GRANT.webserver.server:app --port 8000, open http://localhost:8000/, Start Scan. Watch phase/progress/heatmap update.

ESP32 bridge + Onshape: last — once everything else works.

Priority for today: the arm driver (step 2 blocks step 3, which blocks step 4, etc.). detect_object can come after — stub it with a hardcoded centroid for now and work on the arm.