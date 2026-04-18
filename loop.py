import ollama
import cv2
import base64
import json

# === Camera handles ===
overhead_cam = cv2.VideoCapture(0)  # adjust index
wrist_cam = cv2.VideoCapture(1)

def snap(cam):
    ret, frame = cam.read()
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode()

# === VLM calls ===
PLANNER_PROMPT = """You are controlling a LeRobot SO101 6DOF robot arm with a pincer gripper.
The workspace has a mat on which the robot is placed. Use the corners of the mat as visual reference to guide you.
Goal: Your goal is to use the provided action primitives to first guide the arm towards the shirt in this case, then
grasp it, and move it outside the mat and then let go. Finally, return to base. Feel free to do this step-by-step; You'll
have an opportunity to retry motions until it works. Note that the mesh is split up into grid coordinates, with A1-A3 being
the cells furthest from the robots, A1 with the most negative X, then A2, then A3. B is the next row, then C. YOu also have
access to home and inspect positions, as well as the drop zone.

Current scene from overhead camera is attached.

Output ONLY a JSON list of primitives. Available primitives:
- {"action": "move_to_grid", "cell": "B2"}
- {"action": "approach_from", "direction": "above" | "side"}
- {"action": "close_gripper"}
- {"action": "open_gripper"}
- {"action": "lift"}
- {"action": "home"}

Think about the gripper's capabilities before planning.
"""

VERIFIER_PROMPT = """You just executed: {last_action}
Attached is the wrist camera view.
Did this action succeed? Respond with JSON:
{"success": true/false, "reason": "brief explanation", "retry_suggestion": "..." or null}
"""

def plan(goal, gripper_description):
    img = snap(overhead_cam)
    response = ollama.chat(
        model="llama3.2-vision:11b",
        messages=[{
            "role": "user",
            "content": PLANNER_PROMPT.format(goal=goal, gripper_description=gripper_description),
            "images": [img]
        }]
    )
    return json.loads(response["message"]["content"])

def verify(last_action):
    img = snap(wrist_cam)
    response = ollama.chat(
        model="llama3.2-vision:11b",
        messages=[{
            "role": "user",
            "content": VERIFIER_PROMPT.format(last_action=json.dumps(last_action)),
            "images": [img]
        }]
    )
    return json.loads(response["message"]["content"])

# === Main loop ===
def run(goal, gripper_description, max_retries=3):
    plan_steps = plan(goal, gripper_description)
    print("Plan:", plan_steps)
    
    for step in plan_steps:
        retries = 0
        while retries < max_retries:
            execute_primitive(step)  # <-- your LeRobot wrapper
            result = verify(step)
            print(f"Step {step} -> {result}")
            if result["success"]:
                break
            retries += 1
        else:
            print(f"Failed step {step}, aborting")
            return False
    return True