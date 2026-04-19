"""
memory.py — persistent skill memory + trial history.

Two files per demo:
  memory/best_<demo>.json     — best params ever seen, keyed by demo name
  memory/history_<demo>.jsonl — one line per trial (score, params, aborted)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from scorer import Score


MEMORY_DIR = Path(__file__).parent / "memory"
MEMORY_DIR.mkdir(exist_ok=True)


def _best_path(demo: str) -> Path:
    return MEMORY_DIR / f"best_{demo}.json"


def _history_path(demo: str) -> Path:
    return MEMORY_DIR / f"history_{demo}.jsonl"


def load_best(demo: str) -> Optional[dict]:
    p = _best_path(demo)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def save_best(demo: str, params: dict, score: Score, trial_idx: int):
    _best_path(demo).write_text(json.dumps({
        "trial_idx": trial_idx,
        "score": score.total,
        "success": score.success,
        "params": params,
    }, indent=2))


def append_history(demo: str, trial_idx: int, params: dict, score: Score,
                   accepted: bool):
    entry = {
        "trial_idx": trial_idx,
        "score": score.total,
        "success": score.success,
        "accepted": accepted,
        "breakdown": score.breakdown,
        "params": params,
    }
    with _history_path(demo).open("a") as f:
        f.write(json.dumps(entry) + "\n")


def load_history(demo: str) -> list[dict]:
    p = _history_path(demo)
    if not p.exists():
        return []
    out = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def clear_history(demo: str):
    for p in (_best_path(demo), _history_path(demo)):
        if p.exists():
            p.unlink()