# primitives.py
import json
import time
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/ttyACM0"
ROBOT_ID = "hackathon_arm"

# Gripper values in degrees (since use_degrees=True)
# Tune these after you've recorded the open/closed positions on your arm
GRIPPER_OPEN = 40.0
GRIPPER_CLOSED = -10.0

SETTLE_TIME = 2.5  # seconds after each move


class Arm:
    def __init__(self, port=PORT, pose_file="grid_poses.json"):
        cfg = SO101FollowerConfig(
            port=port,
            id=ROBOT_ID,
            use_degrees=True,
            # max_relative_target caps how far a single command can move.
            # Great safety net for hackathons. Tune upward if moves feel slow.
            max_relative_target=None,
        )
        self.robot = SO101Follower(cfg)
        self.robot.connect()
        try:
            for motor_name in self.robot.bus.ids:
                self.robot.bus.write("Moving_Speed", 150, motor_name)
        except Exception as e:
            print(f"Couldn't set speed: {e}")    
        with open(pose_file) as f:
            self.poses = json.load(f)

        self.last_cell = None

    def _current_pose(self):
        obs = self.robot.get_observation()
        return {k: v for k, v in obs.items() if k.endswith(".pos")}

    def _goto_pose(self, target_pose, preserve_gripper=False):
        """Send the full target pose to the robot."""
        if preserve_gripper:
            current = self._current_pose()
            target_pose = dict(target_pose)  # copy
            target_pose["gripper.pos"] = current["gripper.pos"]
        self.robot.send_action(target_pose)
        time.sleep(SETTLE_TIME)

    def _set_gripper(self, value):
        current = self._current_pose()
        current["gripper.pos"] = value
        self.robot.send_action(current)
        time.sleep(0.8)

    # --- Primitives exposed to the LLM ---

    def move_to_grid(self, cell):
        if cell not in self.poses:
            raise ValueError(f"Unknown cell: {cell}. Known: {list(self.poses.keys())}")
        self._goto_pose(self.poses[cell], preserve_gripper=True)
        self.last_cell = cell

    def approach_from(self, direction):
        """Adjust wrist orientation for different approach angles."""
        current = self._current_pose()
        if direction == "above":
            # Top-down — standard orientation, no change if grid poses are top-down
            pass
        elif direction == "side":
            # Rotate wrist_flex to come in horizontally.
            # Sign and magnitude depend on your arm's zero orientation — tune this.
            current["wrist_flex.pos"] = current["wrist_flex.pos"] - 45.0
            self.robot.send_action(current)
            time.sleep(SETTLE_TIME)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def open_gripper(self):
        self._set_gripper(GRIPPER_OPEN)

    def close_gripper(self):
        self._set_gripper(GRIPPER_CLOSED)

    def descend(self):
        """Move from current hover position to contact position at the same cell."""
        if self.last_cell is None:
            raise ValueError("descend called without a current cell — call move_to_grid first")
        contact_key = f"{self.last_cell}_contact"
        if contact_key not in self.poses:
            raise ValueError(f"No contact pose recorded for {self.last_cell}. Re-run calibrate_grid.py.")
        current = self._current_pose()
        target = dict(self.poses[contact_key])
        target["gripper.pos"] = current["gripper.pos"]
        self.robot.send_action(target)
        time.sleep(SETTLE_TIME)

    def lift(self):
        """Return to the hover pose of the current cell (reverse of descend)."""
        if self.last_cell is None:
            raise ValueError("lift called without a current cell")
        current = self._current_pose()
        target = dict(self.poses[self.last_cell])
        target["gripper.pos"] = current["gripper.pos"]
        self.robot.send_action(target)
        time.sleep(SETTLE_TIME)
        

    def home(self):
        self._goto_pose(self.poses["home"])
        self.last_cell = "home"

    def inspect(self):
        """Move to a pose where the wrist cam can see the gripper."""
        self._goto_pose(self.poses["inspect"], preserve_gripper=True)

    def shutdown(self):
        try:
            self.home()
            time.sleep(1)
        finally:
            self.robot.disconnect()



# --- JSON action -> method dispatcher ---
def execute_primitive(arm, step):
    action = step["action"]
    if action == "move_to_grid":
        arm.move_to_grid(step["cell"])
    elif action == "approach_from":
        arm.approach_from(step["direction"])
    elif action == "open_gripper":
        arm.open_gripper()
    elif action == "close_gripper":
        arm.close_gripper()
    elif action == "home":
        arm.home()
    elif action == "inspect":
        arm.inspect()
    elif action == "descend":
        arm.descend()   # no more amount parameter
    elif action == "lift":
        arm.lift()      # no more amount parameter
    else:
        raise ValueError(f"Unknown action: {action}")