from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from scan_types import Pose6D
import time
from interfaces.robotic_arm import RoboticArm
import numpy as np
import math 

def init_arm():
    return RoboticArm(port="/dev/ttyACM0", urdf_path="models/so101.urdf")

# Function to set joint angles for all joints
def set_joint_angles(arm, shoulder_pan_angle, shoulder_lift_angle, elbow_flex_angle, wrist_flex_angle, wrist_roll_angle, gripper_angle):
    # Create an instance of the RoboticArm
    # Get the current joint angles
    current_angles = arm.get_joint_angles().values
    print(f"Current Joint Angles (radians): {current_angles}")

    # Set the specified joint angles
    target_angles = [
        0.0,  # Fixed base (not used in control)
        shoulder_pan_angle,  # Shoulder Pan (J1)
        shoulder_lift_angle,  # Shoulder Lift (J2)
        elbow_flex_angle,  # Elbow Flex (J3)
        wrist_flex_angle,  # Wrist Flex (Pitch) (J4)
        wrist_roll_angle,  # Wrist Roll (J5)
        gripper_angle  # Gripper (not directly controlled via angles, but you can set this)
    ]
    
    # Construct the action dictionary
    action_dict = {
        'shoulder_pan.pos': target_angles[1],  # J1
        'shoulder_lift.pos': target_angles[2],  # J2
        'elbow_flex.pos': target_angles[3],    # J3
        'wrist_flex.pos': target_angles[4],    # J4 (Pitch)
        'wrist_roll.pos': target_angles[5],    # J5 (Roll)
        'gripper.pos': target_angles[6]        # Gripper position (unchanged)
    }

    # Send the action to the robot
    arm.robot.send_action(action_dict)

    # Wait a little to ensure the robot moves to the new position
    time.sleep(2)

    # Print the new joint angles
    new_angles = arm.get_joint_angles().values
    print(f"New Joint Angles (radians): {new_angles}")

    # Optionally, print the current pose after moving to the specified joint angles
    pose = arm.get_current_pose()
    print(f"New Pose after setting specified joint angles: {pose}")

    for angle in [-90, 0, 90, 180]:
        time.sleep(2)
        action_dict['wrist_roll.pos'] = angle
        arm.robot.send_action(action_dict)
        pose = arm.get_current_pose()
        print(f"New Pose after setting specified joint angles: {pose}")


if __name__ == "__main__":
    # Example: Set all joint angles
    arm = init_arm()
    new_angles = arm.get_joint_angles().values
    print(f"New Joint Angles (radians): {new_angles}")
    # shoulder_pan_angle = 0 # radians (example)
    # shoulder_lift_angle = 0 # radians (example)
    # elbow_flex_angle = 0 # radians (example)
    # wrist_flex_angle = 90# radians (example)
    # wrist_roll_angle = -180  # radians (example)
    # gripper_angle = 0 # radians (example)

    # # Call the function with the specified angles
    # set_joint_angles(arm, shoulder_pan_angle, shoulder_lift_angle, elbow_flex_angle, wrist_flex_angle, wrist_roll_angle, gripper_angle)