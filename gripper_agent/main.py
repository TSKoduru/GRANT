"""
main.py — CLI entry point.

Examples:
  # Train the offset demo for 30 trials with Claude
  python3 main.py --demo offset --trials 30 --llm claude

  # Train with local VLM instead
  python3 main.py --demo softie --trials 30 --llm ollama

  # Run a single pre-set trial without LLM edits (for debugging)
  python3 main.py --demo offset --trials 1 --no-llm

  # Reset a demo's memory
  python3 main.py --demo offset --reset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from memory import clear_history
from perception import Perceiver
from researcher import ClaudeBackend, OllamaBackend, Researcher, write_params
from robot import Robot
from trial_runner import run_trial


CONFIGS_DIR = Path(__file__).parent / "configs"
MAT_MARKERS_PATH = CONFIGS_DIR / "mat_markers.yaml"


def _load_mat_markers() -> dict[int, tuple[float, float]]:
    if not MAT_MARKERS_PATH.exists():
        sys.exit(f"Missing {MAT_MARKERS_PATH}. Run measure_mat_markers.py first.")
    data = yaml.safe_load(MAT_MARKERS_PATH.read_text())
    return {int(k): tuple(v) for k, v in data["mat_markers_robot_mm"].items()}


def _load_demo_config(demo: str) -> dict:
    p = CONFIGS_DIR / f"demo_{demo}.yaml"
    if not p.exists():
        sys.exit(f"Missing demo config {p}.")
    return yaml.safe_load(p.read_text())


def _build_llm(kind: str):
    if kind == "claude":
        return ClaudeBackend()
    if kind == "ollama":
        return OllamaBackend()
    sys.exit(f"Unknown LLM backend '{kind}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True,
                    choices=["offset", "softie", "hook"],
                    help="Which demo config to run")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--llm", choices=["claude", "ollama"], default="ollama")
    ap.add_argument("--no-llm", action="store_true",
                    help="Run trials with initial params, no LLM edits")
    ap.add_argument("--reset", action="store_true",
                    help="Clear memory for this demo and exit")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--urdf", default="models/so101.urdf")
    args = ap.parse_args()

    if args.reset:
        clear_history(args.demo)
        print(f"Cleared memory for demo '{args.demo}'.")
        return

    demo_cfg = _load_demo_config(args.demo)
    initial_params = demo_cfg["params"]

    mat_markers = _load_mat_markers()
    perceiver = Perceiver(mat_markers_robot_mm=mat_markers)
    robot = Robot(port=args.port, urdf_path=args.urdf)

    try:
        # Ensure calibration before starting
        if perceiver.H is None:
            print("[main] No calibration on disk. Calibrating from current frame...")
            obs = robot.get_observation()
            if not perceiver.calibrate(obs.overhead_bgr):
                sys.exit("Calibration failed — check that all 4 mat markers are visible.")

        if args.no_llm:
            print(f"[main] Running {args.trials} trial(s) WITHOUT LLM edits.")
            write_params(initial_params)
            for i in range(1, args.trials + 1):
                score, info = run_trial(i, robot, perceiver, args.demo)
                print(f"[main] Trial {i}: score={score.total:.2f} "
                      f"success={score.success}")
        else:
            llm = _build_llm(args.llm)
            researcher = Researcher(llm, robot, perceiver, args.demo)
            result = researcher.run(args.trials, initial_params=initial_params)
            print(f"\n[main] DONE. Best score: {result.best_score:.2f}")
            print(f"[main] Best params:")
            import json as _j
            print(_j.dumps(result.best_params, indent=2))

    finally:
        robot.close()


if __name__ == "__main__":
    main()