"""
trial_runner.py — runs a single trial of policy.py end-to-end.

Responsibilities:
  - Home + ensure calibration
  - Snapshot start frames
  - Import policy (fresh each trial) and run execute()
  - Snapshot end frames
  - Build SafetyStats-aware score
  - Dump logs/frames to trials/trial_<n>/
"""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from perception import Perceiver, HSVRange
from robot import Robot
from safety import SafeRobot, SafetyViolation
from scorer import Score, compute_score
from skills import Skills, SkillLog


TRIALS_DIR = Path(__file__).parent / "trials"
TRIALS_DIR.mkdir(exist_ok=True)


def _save_frame(path: Path, bgr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)


def _detect_object_post(perceiver: Perceiver, frame, params: dict,
                        gripper_pixel_uv) -> Optional[tuple[float, float]]:
    """Re-run the detector on the end-frame to locate the object."""
    if params["detector"] == "hsv":
        hsv = HSVRange(tuple(params["hsv_low"]), tuple(params["hsv_high"]))
        det = perceiver.detect_hsv(frame, hsv)
    else:
        det = perceiver.detect_bgsubtract(frame, gripper_pixel_uv=gripper_pixel_uv)
    return (det.x_mm, det.y_mm) if det else None


def run_trial(
    trial_idx: int,
    robot: Robot,
    perceiver: Perceiver,
    demo_name: str,
) -> tuple[Score, dict]:
    """
    Returns (Score, trial_info). Also writes everything to trials/<demo>/trial_<n>/.
    """
    trial_dir = TRIALS_DIR / demo_name / f"trial_{trial_idx:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = trial_dir / "frames"

    safe = SafeRobot(robot)
    log = SkillLog()

    # Fresh import each trial — picks up the LLM's latest edits.
    import policy  # noqa: F401
    importlib.reload(policy)

    skills = Skills(safe, perceiver, log)

    # ── Observation: start ──
    start_obs = robot.get_observation()
    _save_frame(frames_dir / "start_overhead.jpg", start_obs.overhead_bgr)
    _save_frame(frames_dir / "start_wrist.jpg", start_obs.wrist_bgr)

    # ── Execute ──
    aborted_reason: Optional[str] = None
    t0 = time.time()
    try:
        result = policy.execute(skills)
    except SafetyViolation as e:
        aborted_reason = f"safety_violation:{e}"
        result = {"object_start": None, "target": None, "aborted": aborted_reason}
    except Exception as e:
        aborted_reason = f"policy_exception:{type(e).__name__}:{e}"
        result = {"object_start": None, "target": None, "aborted": aborted_reason}
    duration_s = time.time() - t0

    # ── Observation: end ──
    end_obs = robot.get_observation()
    _save_frame(frames_dir / "end_overhead.jpg", end_obs.overhead_bgr)
    _save_frame(frames_dir / "end_wrist.jpg", end_obs.wrist_bgr)

    # ── Find object in end-frame ──
    gx, gy, _ = robot.get_gripper_tip_mm()
    grip_uv = (perceiver.robot_mm_to_pixel(gx, gy)
               if perceiver.H is not None else None)
    object_end = _detect_object_post(perceiver, end_obs.overhead_bgr,
                                     policy.PARAMS, grip_uv)

    # ── Score ──
    score = compute_score(
        object_start=result.get("object_start"),
        object_end=object_end,
        target=result.get("target"),
        trial_duration_s=duration_s,
        safety_violations=safe.stats.rejected,
        aborted_reason=result.get("aborted") or aborted_reason,
    )

    # ── Persist everything ──
    with (trial_dir / "log.jsonl").open("w") as f:
        for event in log.events:
            f.write(json.dumps(event) + "\n")

    (trial_dir / "score.json").write_text(json.dumps({
        "total": score.total,
        "success": score.success,
        "breakdown": score.breakdown,
    }, indent=2))

    (trial_dir / "params.json").write_text(json.dumps(policy.PARAMS, indent=2))

    trial_info = {
        "trial_idx": trial_idx,
        "demo": demo_name,
        "duration_s": duration_s,
        "safety_violations": safe.stats.rejected,
        "safety_reasons": list(safe.stats.reasons),
        "object_start": result.get("object_start"),
        "object_end": object_end,
        "target": result.get("target"),
        "aborted_reason": result.get("aborted") or aborted_reason,
        "start_frame": str(frames_dir / "start_overhead.jpg"),
        "end_frame": str(frames_dir / "end_overhead.jpg"),
    }
    (trial_dir / "info.json").write_text(json.dumps(trial_info, indent=2))

    return score, trial_info