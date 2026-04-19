"""
researcher.py — the autoresearch loop.

Drives repeated trials:
  1. Read current policy + last trial's outcome
  2. Ask the LLM for new PARAMS
  3. Write new PARAMS into policy.py
  4. Run trial via trial_runner
  5. If score improved, keep; else revert
  6. Record history + best-so-far in memory.py
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from perception import Perceiver
from robot import Robot
from scorer import Score
from trial_runner import run_trial
from memory import append_history, load_best, save_best


POLICY_PATH = Path(__file__).parent / "policy.py"
PROMPT_DIR = Path(__file__).parent / "prompts"


# ─────────────────────────────────────────────────────────────────────────────
# LLM backend protocol — swap Claude / Ollama transparently
# ─────────────────────────────────────────────────────────────────────────────

class LLMBackend(Protocol):
    def propose_params(
        self,
        system_prompt: str,
        feedback: str,
        start_image_jpeg_b64: str,
        end_image_jpeg_b64: str,
    ) -> dict: ...


class ClaudeBackend:
    """Claude API backend. Requires anthropic package + ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def propose_params(self, system_prompt, feedback, start_image_jpeg_b64, end_image_jpeg_b64):
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg",
                        "data": start_image_jpeg_b64,
                    }},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg",
                        "data": end_image_jpeg_b64,
                    }},
                    {"type": "text", "text": feedback},
                ],
            }],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return _extract_json(text)


class OllamaBackend:
    """Local VLM via Ollama. Needs 'ollama' python package + running daemon."""

    def __init__(self, model: str = "qwen2.5vl:7b"):
        import ollama
        self.client = ollama
        self.model = model

    def propose_params(self, system_prompt, feedback, start_image_jpeg_b64, end_image_jpeg_b64):
        resp = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": feedback,
                 "images": [start_image_jpeg_b64, end_image_jpeg_b64]},
            ],
            options={"temperature": 0.2},
        )
        return _extract_json(resp["message"]["content"])


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of the LLM's response."""
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    # Find first {...} block (non-greedy across lines)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in LLM response:\n{text[:500]}")
    return json.loads(match.group(0))


# ─────────────────────────────────────────────────────────────────────────────
# PARAMS read/write via AST surgery (preserves formatting, safer than regex)
# ─────────────────────────────────────────────────────────────────────────────

def read_current_params() -> dict:
    src = POLICY_PATH.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "PARAMS"):
            return ast.literal_eval(node.value)
    raise RuntimeError("PARAMS dict not found in policy.py")


def write_params(new_params: dict):
    src = POLICY_PATH.read_text()
    pretty = json.dumps(new_params, indent=4)
    pretty = pretty.replace(": true", ": True").replace(": false", ": False").replace(": null", ": None")
    new_block = f"PARAMS = {pretty}\n"
    # Replace from PARAMS = { through its closing brace
    pattern = re.compile(r"PARAMS\s*=\s*\{.*?\n\}\n", re.DOTALL)
    if not pattern.search(src):
        raise RuntimeError("Couldn't locate PARAMS block to replace in policy.py")
    new_src = pattern.sub(new_block, src, count=1)
    POLICY_PATH.write_text(new_src)


# ─────────────────────────────────────────────────────────────────────────────
# Researcher loop
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResearchResult:
    best_score: float
    best_params: dict
    trials_run: int


class Researcher:
    def __init__(
        self,
        llm: LLMBackend,
        robot: Robot,
        perceiver: Perceiver,
        demo_name: str,
    ):
        self.llm = llm
        self.robot = robot
        self.perceiver = perceiver
        self.demo_name = demo_name

        self.system_prompt = (PROMPT_DIR / "system.md").read_text()
        self.feedback_template = (PROMPT_DIR / "feedback_template.md").read_text()

    def run(self, num_trials: int, initial_params: Optional[dict] = None) -> ResearchResult:
        if initial_params is not None:
            write_params(initial_params)

        best = load_best(self.demo_name)
        best_score = best["score"] if best else float("-inf")
        best_params = best["params"] if best else read_current_params()

        last_trial_info = None
        last_score: Optional[Score] = None

        for trial_idx in range(1, num_trials + 1):
            if trial_idx > 1 and last_trial_info is not None and last_score is not None:
                self._propose_and_write(last_trial_info, last_score,
                                         best_params, best_score)

            current_params = read_current_params()

            print(f"[Researcher] --- Trial {trial_idx} / {num_trials} ---")
            score, trial_info = run_trial(
                trial_idx, self.robot, self.perceiver, self.demo_name,
            )

            accepted = score.total > best_score
            if accepted:
                print(f"[Researcher] Accepted. score={score.total:.2f} "
                      f"(prev best {best_score:.2f})")
                best_score = score.total
                best_params = current_params
                save_best(self.demo_name, current_params, score, trial_idx)
            else:
                print(f"[Researcher] Rejected. score={score.total:.2f} "
                      f"(best {best_score:.2f}). Reverting.")
                write_params(best_params)

            append_history(self.demo_name, trial_idx, current_params, score, accepted)

            last_trial_info = trial_info
            last_score = score

        return ResearchResult(
            best_score=best_score,
            best_params=best_params,
            trials_run=num_trials,
        )

    # ── LLM proposal ───────────────────────────────────────────────────────

    def _propose_and_write(
        self,
        trial_info: dict,
        score: Score,
        best_params: dict,
        best_score: float,
    ):
        current_params = read_current_params()
        feedback = self.feedback_template.format(
            trial_idx=trial_info["trial_idx"],
            score=f"{score.total:.2f}",
            success=score.success,
            breakdown=json.dumps(score.breakdown, indent=2),
            best_score=f"{best_score:.2f}",
            object_start=trial_info.get("object_start"),
            object_end=trial_info.get("object_end"),
            target=trial_info.get("target"),
            safety_reasons=trial_info.get("safety_reasons", []),
            aborted_reason=trial_info.get("aborted_reason"),
            current_params=json.dumps(current_params, indent=2),
            best_params=json.dumps(best_params, indent=2),
        )

        start_b64 = _read_image_b64(trial_info["start_frame"])
        end_b64 = _read_image_b64(trial_info["end_frame"])

        try:
            proposed = self.llm.propose_params(
                self.system_prompt, feedback, start_b64, end_b64,
            )
        except Exception as e:
            print(f"[Researcher] LLM proposal failed: {e}. Keeping current PARAMS.")
            return

        # Merge proposed keys into current (LLM may propose only diffs).
        merged = dict(current_params)
        merged.update(proposed)
        try:
            write_params(merged)
            print(f"[Researcher] Wrote new PARAMS. Changed keys: "
                  f"{sorted(set(proposed.keys()))}")
        except Exception as e:
            print(f"[Researcher] Write failed: {e}. Keeping current PARAMS.")


def _read_image_b64(path: str) -> str:
    import base64
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")