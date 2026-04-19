"""
policy.py — THE FILE THE LLM EDITS.

The LLM adjusts PARAMS between trials. The execute() body stays stable.
The researcher loop git-commits accepted changes and git-restores rejects.

All strategies (grasp / push) are pre-seeded. The LLM picks one by setting
PARAMS["strategy"] and tunes the parameters for that strategy.
"""

from __future__ import annotations

from perception import HSVRange
from skills import Skills


# ─────────────────────────────────────────────────────────────────────────────
# PARAMS — the LLM edits these.
# ─────────────────────────────────────────────────────────────────────────────

PARAMS = {
    # which high-level strategy to run
    "strategy": "grasp",                    # "grasp" | "push"

    # object detection
    "detector": "hsv",                      # "hsv" | "bgsub"
    "hsv_low":  [0, 120, 70],
    "hsv_high": [10, 255, 255],

    # target (static position in robot-frame mm)
    "target_x_mm": 200.0,
    "target_y_mm": 250.0,

    # grasp-strategy params
    "approach_offset_x_mm": 0.0,
    "approach_offset_y_mm": 0.0,
    "grasp_z_mm": 10.0,
    "pre_grasp_width": 1.0,
    "grasp_close_width": 0.15,
    "grasp_settle_s": 0.35,
    "release_z_mm": 30.0,

    # push-strategy params (demo 3)
    "push_z_mm": 15.0,
    "push_approach_overshoot_mm": 30.0,
    "push_follow_through_mm": 20.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Execute — the runner calls this. Don't restructure unless absolutely needed.
# ─────────────────────────────────────────────────────────────────────────────

def execute(skills: Skills) -> dict:
    """
    Run one trial. Returns a dict of trial-scoped data for the scorer
    (where the object started, where the target was, etc.).
    """
    skills.home()

    if PARAMS["detector"] == "hsv":
        hsv = HSVRange(
            low=tuple(PARAMS["hsv_low"]),
            high=tuple(PARAMS["hsv_high"]),
            name="target_object",
        )
        obj = skills.detect_object_hsv(hsv)
    else:
        obj = skills.detect_object_bgsub()

    if obj is None:
        return {"object_start": None, "target": None, "aborted": "object_not_found"}

    object_start = (obj.x_mm, obj.y_mm)
    target = (PARAMS["target_x_mm"], PARAMS["target_y_mm"])

    if PARAMS["strategy"] == "grasp":
        skills.pick_at(
            obj.x_mm, obj.y_mm,
            grasp_z_mm=PARAMS["grasp_z_mm"],
            pre_grasp_width=PARAMS["pre_grasp_width"],
            grasp_close_width=PARAMS["grasp_close_width"],
            settle_s=PARAMS["grasp_settle_s"],
            approach_offset_x_mm=PARAMS["approach_offset_x_mm"],
            approach_offset_y_mm=PARAMS["approach_offset_y_mm"],
        )
        skills.place_at(*target, release_z_mm=PARAMS["release_z_mm"])

    elif PARAMS["strategy"] == "push":
        skills.push_toward(
            obj.x_mm, obj.y_mm, target[0], target[1],
            push_z_mm=PARAMS["push_z_mm"],
            approach_overshoot_mm=PARAMS["push_approach_overshoot_mm"],
            follow_through_mm=PARAMS["push_follow_through_mm"],
        )

    else:
        return {"object_start": object_start, "target": target,
                "aborted": f"unknown_strategy_{PARAMS['strategy']}"}

    skills.home()
    return {"object_start": object_start, "target": target, "aborted": None}