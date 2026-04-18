# planner.py
import json
import ollama
import re

MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = """You are a robot arm planner. Your job is to generate a PICKUP plan.
The delivery step is handled separately — your plan should END with the object lifted.

THE WORKSPACE:
A 3x3 grid labeled A1-C3 is laid out on the table in front of the arm.
- Row A is farthest from the arm (back of the workspace)
- Row C is closest to the arm (front of the workspace)
- Column 1 is leftmost, column 3 is rightmost
- So A1 is back-left, A3 is back-right, C1 is front-left, C3 is front-right, B2 is center

THE GRIPPER:
<<GRIPPER>>

AVAILABLE PRIMITIVES:
- {"action": "home"} — return to safe resting position
- {"action": "move_to_grid", "cell": "A1"} — move above a grid cell (valid: A1-C3)
- {"action": "open_gripper"}
- {"action": "close_gripper"}
- {"action": "descend"} — lower arm to contact position at current cell
- {"action": "lift"} — raise arm back to hover position at current cell
- {"action": "approach_from", "direction": "above" | "side"}

RULES:
1. Start with 'home' and 'open_gripper'.
2. The correct pickup sequence is: home → open_gripper → move_to_grid → descend → close_gripper → lift.
3. DO NOT include a move to drop_zone — the plan ends after the lift.
4. Read the scene carefully and target the cell mentioned.
5. Think about the gripper's capabilities — a hook cannot pinch, a pincer cannot scoop.

EXAMPLE:
Scene: A blue cube is at grid cell B2.
Goal: Pick up the blue cube.
Gripper: A standard two-finger pincer gripper.

Correct plan:
{"plan": [
  {"action": "home"},
  {"action": "open_gripper"},
  {"action": "move_to_grid", "cell": "B2"},
  {"action": "approach_from", "direction": "above"},
  {"action": "descend"},
  {"action": "close_gripper"},
  {"action": "lift"}
], "target_cell": "B2"}

OUTPUT FORMAT:
Output ONLY a JSON object of the form:
{"plan": [ ... ], "target_cell": "X#"}
No explanation, no markdown fences, just the JSON object.
"""

USER_PROMPT = """Scene description: <<SCENE>>
Goal: <<GOAL>>

Output the JSON plan."""


RETRY_SYSTEM_PROMPT = """You are a robot arm planner. A previous grasp attempt FAILED.

THE WORKSPACE:
A 3x3 grid labeled A1-C3.
- A is back, C is front. 1 is left, 3 is right.

THE GRIPPER:
<<GRIPPER>>

THE SITUATION:
The shirt is spread across multiple cells. Previous attempts at these cells FAILED:
<<FAILED_CELLS>>

Pick a DIFFERENT cell than any failed one. Choose a cell ADJACENT to a previously-failed cell, since the shirt is wide and neighboring cells likely still have fabric.

Valid cells: A1, A2, A3, B1, B2, B3, C1, C2, C3

AVAILABLE PRIMITIVES:
Same as before: home, move_to_grid, open_gripper, close_gripper, descend, lift, approach_from.

Pickup sequence: home → open_gripper → move_to_grid → descend → close_gripper → lift.

OUTPUT FORMAT:
{"plan": [ ... ], "target_cell": "X#", "reasoning": "one sentence about why this cell"}
"""


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def plan_pickup(goal, scene_description, gripper_description):
    """Generate the initial pickup plan. Returns dict with 'plan' and 'target_cell'."""
    system = SYSTEM_PROMPT.replace("<<GRIPPER>>", gripper_description)
    user = USER_PROMPT.replace("<<SCENE>>", scene_description).replace("<<GOAL>>", goal)

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format="json",
        options={"temperature": 0.2},
    )
    parsed = extract_json(response["message"]["content"])
    if "plan" not in parsed or "target_cell" not in parsed:
        raise ValueError(f"Malformed plan response: {parsed}")
    return parsed


def retry_pickup(gripper_description, failed_cells):
    """Generate a retry plan targeting a different cell."""
    system = RETRY_SYSTEM_PROMPT.replace("<<GRIPPER>>", gripper_description)
    system = system.replace("<<FAILED_CELLS>>", ", ".join(failed_cells))

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "Generate a retry pickup plan."},
        ],
        format="json",
        options={"temperature": 0.5},  # higher variety for retries
    )
    parsed = extract_json(response["message"]["content"])
    if "plan" not in parsed or "target_cell" not in parsed:
        raise ValueError(f"Malformed retry response: {parsed}")
    return parsed


if __name__ == "__main__":
    PINCER = "A standard two-finger pincer gripper that closes to grasp objects."
    result = plan_pickup(
        goal="Pick up the shirt.",
        scene_description="The shirt is spread across A3 and surrounding cells.",
        gripper_description=PINCER,
    )
    print(json.dumps(result, indent=2))