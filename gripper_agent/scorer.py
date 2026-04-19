"""
scorer.py — deterministic, sensor-based trial scoring.

No VLM in the loop. Everything here is measured distances + log inspection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Score:
    total: float
    success: bool
    breakdown: dict = field(default_factory=dict)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_score(
    object_start: Optional[tuple[float, float]],
    object_end: Optional[tuple[float, float]],
    target: Optional[tuple[float, float]],
    trial_duration_s: float,
    safety_violations: int,
    aborted_reason: Optional[str],
    success_threshold_mm: float = 40.0,
) -> Score:
    """
    Score formula:
      base 50
      + min(progress_toward_target * 0.5, 40)
      + 20 if success
      - 5 * safety_violations
      - 0.2 * trial_duration_s
      - 20 if aborted
    Clamped to [0, 100].
    """
    breakdown = {
        "base": 50,
        "aborted_reason": aborted_reason,
        "safety_violations": safety_violations,
        "duration_s": round(trial_duration_s, 2),
    }

    if aborted_reason is not None or object_start is None or object_end is None or target is None:
        breakdown["reason"] = "missing_data_or_aborted"
        penalty = 20 if aborted_reason else 0
        total = max(0, 50 - penalty - 5 * safety_violations - 0.2 * trial_duration_s)
        return Score(total=total, success=False, breakdown=breakdown)

    start_to_target = _dist(object_start, target)
    end_to_target = _dist(object_end, target)
    progress = start_to_target - end_to_target
    success = end_to_target < success_threshold_mm

    breakdown.update({
        "start_to_target_mm": round(start_to_target, 1),
        "end_to_target_mm": round(end_to_target, 1),
        "progress_mm": round(progress, 1),
        "success_threshold_mm": success_threshold_mm,
    })

    score = 50.0
    score += min(progress * 0.5, 40.0)
    if success:
        score += 20.0
        breakdown["success_bonus"] = 20.0
    score -= 5.0 * safety_violations
    score -= 0.2 * trial_duration_s
    score = max(0.0, min(100.0, score))

    return Score(total=score, success=success, breakdown=breakdown)