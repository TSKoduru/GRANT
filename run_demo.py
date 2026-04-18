# run_demo.py
import json
import time
from primitives import Arm, execute_primitive
from planner import plan_pickup, retry_pickup
from verifier import capture_baseline, verify_grasp_vs_baseline

PINCER = "A standard two-finger pincer gripper that closes to grasp objects by squeezing them."
HOOK = "A rigid hook attachment (the pincer fingers have been removed). It cannot grasp or squeeze. To pick something up, slide the hook UNDER the object and lift."

MAX_RETRIES = 3

DELIVERY_STEPS = [
    {"action": "move_to_grid", "cell": "drop_zone"},
    {"action": "open_gripper"},
    {"action": "home"},
]


def execute_plan(arm, steps):
    """Run a list of primitives sequentially."""
    for step in steps:
        print(f"    → {step}")
        execute_primitive(arm, step)


def inspect_and_verify(arm, baseline_img):
    """Move to inspect pose and check if gripper is holding the object."""
    print("  Moving to inspect pose...")
    execute_primitive(arm, {"action": "inspect"})
    time.sleep(0.5)  # let camera settle
    print("  Checking grasp...")
    result = verify_grasp_vs_baseline(baseline_img, save_frame_path="last_grasp_check.jpg")
    grasped = result.get("grasped", False)
    print(f"  Verdict: grasped={grasped} (confidence={result.get('confidence')})")
    print(f"  Reason: {result.get('reason')}")
    return grasped


def run_adaptive_demo(goal, scene_description, gripper_description):
    print(f"\n{'=' * 60}")
    print(f"GOAL: {goal}")
    print(f"SCENE: {scene_description}")
    print(f"GRIPPER: {gripper_description[:80]}")
    print(f"{'=' * 60}\n")

    arm = Arm()
    try:
        # --- Setup: baseline capture ---
        print("SETUP: Capturing baseline (empty gripper at inspect pose)...")
        execute_primitive(arm, {"action": "home"})
        execute_primitive(arm, {"action": "open_gripper"})
        execute_primitive(arm, {"action": "inspect"})
        time.sleep(0.5)
        baseline_img = capture_baseline(save_path="baseline_empty.jpg")
        print("  Baseline captured.\n")

        # --- Initial pickup attempt ---
        print("Generating initial pickup plan...")
        plan_result = plan_pickup(goal, scene_description, gripper_description)
        steps = plan_result["plan"]
        current_cell = plan_result["target_cell"]

        print(f"Plan (target cell: {current_cell}):")
        for i, s in enumerate(steps):
            print(f"  {i+1}. {s}")
        input("\nPress Enter to execute, or Ctrl+C to abort...")

        failed_cells = []
        grasped = False

        for attempt in range(1, MAX_RETRIES + 2):  # 1 initial + MAX_RETRIES retries
            print(f"\n--- ATTEMPT {attempt} (targeting {current_cell}) ---")
            execute_plan(arm, steps)

            grasped = inspect_and_verify(arm, baseline_img)

            if grasped:
                print(f"\n✓ GRASP SUCCEEDED on attempt {attempt}!")
                break

            # Failed — record and retry if we have attempts left
            print(f"\n✗ Grasp failed on attempt {attempt}.")
            failed_cells.append(current_cell)

            if attempt > MAX_RETRIES:
                print(f"Out of retries. Giving up.")
                break

            # Release and return home to prepare for retry
            print("Releasing and returning to home for retry...")
            execute_primitive(arm, {"action": "open_gripper"})
            execute_primitive(arm, {"action": "home"})

            # Generate new plan
            print(f"Asking planner for a new target (avoiding: {failed_cells})...")
            retry_result = retry_pickup(gripper_description, failed_cells)
            steps = retry_result["plan"]
            current_cell = retry_result["target_cell"]
            reasoning = retry_result.get("reasoning", "(no reasoning provided)")
            print(f"  New target: {current_cell}")
            print(f"  Reasoning: {reasoning}")

        # --- Delivery or graceful cleanup ---
        if grasped:
            print("\n--- DELIVERY ---")
            execute_plan(arm, DELIVERY_STEPS)
            print("\n✓ Demo complete!")
        else:
            print("\n--- CLEANUP ---")
            execute_plan(arm, [
                {"action": "open_gripper"},
                {"action": "home"},
            ])
            print("\n✗ Demo failed to grasp the shirt after all retries.")
            print(f"  Failed cells: {failed_cells}")

    finally:
        arm.shutdown()


if __name__ == "__main__":
    run_adaptive_demo(
        goal="Pick up the shirt and move it to the drop zone.",
        scene_description="The shirt is spread across the workspace, centered near A3 but extending into neighboring cells (A2, B3).",
        gripper_description=PINCER,
    )