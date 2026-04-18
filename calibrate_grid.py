# calibrate_grid.py
import json
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/ttyACM0"  # change to your port
ROBOT_ID = "hackathon_arm"

CELLS = [
    "A1", "A1_contact",
    "A2", "A2_contact",
    "A3", "A3_contact",
    "B1", "B1_contact",
    "B2", "B2_contact",
    "B3", "B3_contact",
    "C1", "C1_contact",
    "C2", "C2_contact",
    "C3", "C3_contact",
    "home",
    "inspect",
    "drop_zone",
    "drop_zone_contact",  # in case you want to descend before releasing too
]

def main():
    cfg = SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        use_degrees=True,   # more human-readable when we debug
    )
    robot = SO101Follower(cfg)
    robot.connect()

    # Disable torque so you can move the arm by hand.
    # The underlying bus is accessible — if this attribute is named differently,
    # we'll fall back to disconnect/reconnect trickery.
    try:
        robot.bus.disable_torque()
        print("Torque disabled. Move the arm freely.\n")
    except AttributeError:
        print("Note: couldn't access bus directly to disable torque.")
        print("If the arm resists movement, we'll need another approach.\n")

    poses = {}
    for cell in CELLS:
        input(f"Move arm to '{cell}', then press Enter to record...")
        obs = robot.get_observation()
        # Keep only the .pos fields (observation may include other things)
        pose = {k: v for k, v in obs.items() if k.endswith(".pos")}
        poses[cell] = pose
        print(f"  Recorded {cell}: {pose}\n")

    with open("grid_poses.json", "w") as f:
        json.dump(poses, f, indent=2)
    print("Saved grid_poses.json")

    robot.disconnect()

if __name__ == "__main__":
    main()