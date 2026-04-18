# recalibrate_one.py
import json
import os
import sys
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/ttyACM0"
ROBOT_ID = "hackathon_arm"
POSE_FILE = "grid_poses.json"


def main():
    if len(sys.argv) != 2:
        print("Usage: python recalibrate_one.py <pose_name>")
        print("Examples:")
        print("  python recalibrate_one.py A3")
        print("  python recalibrate_one.py A3_contact")
        print("  python recalibrate_one.py drop_zone")
        sys.exit(1)

    pose_name = sys.argv[1]

    # Load existing poses (or start fresh if file doesn't exist)
    if os.path.exists(POSE_FILE):
        with open(POSE_FILE) as f:
            poses = json.load(f)
    else:
        print(f"No existing {POSE_FILE} found — creating new one.")
        poses = {}

    if pose_name in poses:
        print(f"'{pose_name}' already exists. Current value:")
        print(f"  {poses[pose_name]}")
        confirm = input("Overwrite? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    # Connect to arm
    cfg = SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        use_degrees=True,
    )
    robot = SO101Follower(cfg)
    robot.connect()

    # Disable torque so you can move by hand
    try:
        robot.bus.disable_torque()
        print("\nTorque disabled. Move the arm freely.")
    except AttributeError:
        print("\nWarning: couldn't disable torque programmatically.")
        print("If the arm is stiff, power-cycle it before continuing.")

    try:
        input(f"\nMove arm to '{pose_name}', then press Enter to record...")
        obs = robot.get_observation()
        pose = {k: v for k, v in obs.items() if k.endswith(".pos")}
        poses[pose_name] = pose

        with open(POSE_FILE, "w") as f:
            json.dump(poses, f, indent=2)

        print(f"\nRecorded '{pose_name}':")
        print(f"  {pose}")
        print(f"Saved to {POSE_FILE}")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()