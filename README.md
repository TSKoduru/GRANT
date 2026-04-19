<div align="center">

<img src="image%20copy.png" alt="G.R.A.N.T. logo" width="280"/>

# G.R.A.N.T.

### **G**ripper-agnostic **R**einforcement via **A**utoresearch with **N**umeric **T**uning

*A software layer that makes any gripper adaptive — the robot watches itself fail, edits its own control parameters, and converges on a working policy. The same codebase, the same skills, the same reward: radically different grippers, zero human tuning.*


</div>

---

## Background

Grippers are typically engineered for a specific task. Add a foam pad, swap a finger, wrap the jaws in rubber — the carefully-calibrated control code breaks. **G.R.A.N.T. treats the gripper as an unknown variable** and learns around it using a tight closed loop:

> **propose parameter change → run trial → measure outcome → keep or revert**

It is the autoresearch paradigm (LLM edits its own code on a measured objective) applied to physical manipulation for the first time at this scope.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         RESEARCHER LOOP                              │
│                                                                      │
│   LLM (Claude cloud / local VLM on AMD Ryzen PC) reads logs          │    
│   + images, proposes PARAMS edits, writes policy.py                  │
│            │                                                         │
│            ▼                                                         │
│   Trial Runner: import policy → execute → score                      │
│            │                                                         │
│            ▼                                                         │
│   Score improved? → commit. Else → revert.                           │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        POLICY (editable)                             │
│   - Tunable PARAMS dict (offsets, thresholds, strategy)              │
│   - Fixed execute() body calling skill macros                        │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    SKILLS (vetted actions)                           │
│      move to x/y/z, move up/down, etc.                               │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    SAFETY — shielding layer                          │
│   Every hardware command validated: joint limits, workspace box,     │
│   speed caps. Violations logged + fed back to the LLM as signal.     │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│           HARDWARE — AMD-powered LeRobot SO-101 + Perception         │
│   SO-101 on LeRobot • overhead + wrist cams • homography + FK        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Hardware Integration

### AMD AI PC + LeRobot SO-101 (AMD Track)

The AI PC is the brain of the autoresearch loop, the FPGA is the accelerator on the perception path, and the LeRobot SO-101 is the body they drive. All three are one integrated stack.

**LeRobot SO-101 arm.** G.R.A.N.T. is built on top of the open-source [LeRobot](https://github.com/huggingface/lerobot) stack. We drive a **5-DOF SO-101 follower arm + parallel gripper** at **30 Hz interpolated motion control** using `lerobot.robots.so101follower`. Joint state, servo temperatures, and gripper current stream back into the observation bundle every tick. A URDF of the arm (`gripper_agent/models/so101.urdf`) feeds **ikpy** for forward and inverse kinematics — giving us the gripper-tip position in the same coordinate frame as our object detection, with no fiducial marker needed on the end-effector.

- **Joint-space + Cartesian control** through a single `SafeRobot` wrapper
- **Two-camera perception** (overhead fixed + wrist-mounted) via LeRobot's camera abstraction
- **Homography-based workspace calibration** — 4 ArUco markers on a mat, jogged to with the arm itself, give a pixel → millimeter map in 5 minutes to ~2 mm accuracy

**AI PC — the researcher host.** The  AI PC runs the researcher loop, the trial runner, and the LLM backend. 

### Ford Novel Gripper Challenge (Ford Track)

Ford's challenge: *develop an invention that demonstrates an improvement to a robot's ability to grasp objects.* Most entries will improve the gripper itself — a new jaw geometry, a new compliant material, a new sensor. **G.R.A.N.T. improves grasping without touching the hardware.** One software layer makes *any* gripper better at grasping, because it measures what the current gripper actually does and tunes itself around it. The improvement compounds: any future physical-gripper innovation just becomes one more starting point G.R.A.N.T. can converge from.

Our demos show the same code recovering grasp performance across several very different gripper modifications. **Same code, same skills, same LLM, same scoring function** — only the physical gripper changes.

---

## The File Set

### Core reasoning loop

| File | Role |
|---|---|
| **`researcher.py`** | Drives the autoresearch loop. Reads last trial outcome, calls LLM for new PARAMS, writes them into `policy.py`, runs next trial, keeps-or-reverts based on score. |
| **`policy.py`** | The single file the LLM edits. A tunable PARAMS dict at the top, a fixed `execute()` body that reads PARAMS and calls skills. Ships with both grasp and push strategies pre-seeded. |
| **`trial_runner.py`** | Runs one trial end-to-end: snapshot start state, import policy, execute, snapshot end state, compute score, dump everything to `trials/trial_N/`. |
| **`scorer.py`** | Deterministic scoring. Score = base + progress toward target + success bonus − safety violations − duration. No LLM in the loop. |
| **`memory.py`** | Persists best-params-ever and full trial history per demo. Drives the convergence graph that appears in the demo video. |

### Robot + world

| File | Role |
|---|---|
| **`robot.py`** | Hardware abstraction: SO-101 via LeRobot (port, joint read/write, interpolated motion at 30 Hz), both cameras, and FK-based gripper-tip queries. |
| **`kinematics.py`** | ikpy wrapper over the SO-101 URDF. Forward kinematics for tracking, inverse kinematics for Cartesian commands. |
| **`perception.py`** | Vision pipeline. 4-marker ArUco homography calibrates pixel → mm once; HSV segmentation and background subtraction find objects every frame. |
| **`safety.py`** | The shielding layer every command flows through. Validates joint limits, workspace box, speed caps. Rejections don't crash trials — they're logged and fed back to the LLM as learning signal. |
| **`skills.py`** | High-level action macros. Abstracts *how* the arm moves so the policy only decides *what* to do. |

### Entry points + setup

| File | Role |
|---|---|
| **`main.py`** | CLI entry point. `python3 main.py --demo offset --trials 30 --llm claude` runs a full training session. |
| **`drive.py`** | Interactive REPL for manual arm control. Useful for jogging, testing, and debugging without a teleop leader. |
| **`measure_mat_markers.py`** | One-time setup. Walks you through jogging the gripper to each mat corner marker and records its position in robot coordinates via FK. |
| **`test_perception.py`** | Live sanity check. Shows the camera feed with overlays: green circles on detected markers, yellow cross where FK places the gripper, red circle on detected object. Proves the whole pipeline works before you start training. |

### Configs + prompts

| File | Role |
|---|---|
| **`prompts/system.md`** | LLM's role description, scoring formula, tuning heuristics, output format rules. |
| **`prompts/feedback_template.md`** | Per-trial feedback format sent to the LLM (score, positions, current params, best params, images). |
| **`configs/mat_markers.yaml`** | Generated by `measure_mat_markers.py`. The 4 mat corner positions in robot frame. |

---

## Impact

Most adaptive-manipulation work is either (a) deep RL in simulation with massive compute, or (b) hand-tuned controllers that fail the moment you change the hardware. G.R.A.N.T. is neither.

- **Converges in minutes on real hardware** — not millions of sim steps
- **Zero retraining** when the gripper changes — just new trials
- **Deterministic, auditable scoring** — no black-box reward model

The underlying technique (autoresearch: LLM-driven code evolution against a measured objective) is bleeding-edge for ML training. We believe this is one of its first serious applications to physical manipulation.

<div align="center">

---

*Built at StarkHacks*
</div>
