import time
import math
import numpy as np
import ikpy.chain
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from scipy.optimize import minimize
from scan_types import JointAngles, Pose6D

# LeRobot imports
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

@dataclass
class ArmMoveResult:
    success: bool
    achieved_pose: Pose6D
    error_mm: float
    message: str

@dataclass
class GripResult:
    success: bool
    grip_pose: Pose6D

@dataclass
class RotateResult:
    success: bool
    requested_radians: float
    achieved_radians: float
    delta_error: float

class RoboticArm:
    """
    Implementation for the SO-101 5-DoF robotic arm equipped with an Oak-D.
    Utilizes ikpy for Kinematics and LeRobot for the hardware abstraction.
    """

    OVERHEAD_HEIGHT = 0.22              # meters above table for overhead scan position
    WRIST_ROLL_SCAN_ANGLES = [-90.0, 0.0, 90.0, 180.0]  # degrees — one full rotation in 4 steps

    def __init__(self, port: str = "/dev/ttyACM0", urdf_path: str = "models/so101.urdf"):
        # Scan tuning parameters
        self.sweep_height = 0.15       # meters above target
        self.sweep_half_length = 0.1  # meters
        
        # 1. Initialize Kinematics Solver (Radians)
        # Note: Active mask is [Base(Fixed), J1, J2, J3, J4, J5, EndEffector(Fixed)]
        self.chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path, 
            active_links_mask=[False, True, True, True, True, True, False]
        )
        
        # 2. Setup LeRobot Hardware (Degrees)
        print(f"[RoboticArm] Connecting LeRobot to {port}...")
        config = SO101FollowerConfig(port=port, use_degrees=True)
        self.robot = SO101Follower(config)
        self.robot.connect()
        print("[RoboticArm] Hardware connection successful.")

        # Speed limits
        try:
            # Extract motor names dynamically from the observation keys
            obs_keys = [k for k in self.robot.get_observation().keys() if k.endswith(".pos")]
            self.motor_names = [k.replace(".pos", "") for k in obs_keys]

            # Tune these to your preference. 
            # Note: Feetech scale is typically 0-4095 for speed, and 0-254 for acceleration.
            # Start conservative. If it's too slow, increase these values.
            default_velocity = 800
            default_accel = 15 
            
            # Loop through and write to each motor individually
            for motor_name in self.motor_names:
                self.robot.bus.write("Goal_Velocity", motor_name, default_velocity)
                self.robot.bus.write("Acceleration", motor_name, default_accel)
                
            print("[RoboticArm] Hardware velocity profiles applied successfully.")
        except Exception as e:
            print(f"[RoboticArm] Note: Could not set default hardware speed limits: {e}")

    # ── Hardware & Kinematics Layer ─────────────────────────────────

    def get_joint_angles(self) -> JointAngles:
        """Reads raw state from LeRobot, converts degrees to radians, pads for IK."""
        obs = self.robot.get_observation()
        
        # Extract the 5 active arm joints in sequential order
        raw_active_degrees = [
            float(obs['shoulder_pan.pos']),  # Waist / J1
            float(obs['shoulder_lift.pos']), # Shoulder / J2
            float(obs['elbow_flex.pos']),    # Elbow / J3
            float(obs['wrist_flex.pos']),    # Pitch / J4
            float(obs['wrist_roll.pos'])     # Roll / J5
        ]
        
        # Convert to radians
        active_rads = np.deg2rad(raw_active_degrees).tolist()
        
        # Pad for ikpy: [Fixed Base, J1..J5, Fixed EE]
        padded_rads = [0.0] + active_rads + [0.0]
        
        return JointAngles(values=padded_rads)

    def get_current_pose(self) -> Pose6D:
        """End-effector pose via Forward Kinematics."""
        angles = self.get_joint_angles().values
        fk_matrix = self.chain.forward_kinematics(angles)
        
        x, y, z = fk_matrix[:3, 3]
        rot_matrix = fk_matrix[:3, :3]
        qx, qy, qz, qw = R.from_matrix(rot_matrix).as_quat()
        
        return Pose6D(x=float(x), y=float(y), z=float(z), qx=float(qx), qy=float(qy), qz=float(qz), qw=float(qw))

    def _solve_orientation_only(self, target_pointing_vec: np.ndarray, 
                                target_pos: np.ndarray,
                                initial_angles: list) -> np.ndarray:
        active_indices = [i for i, m in enumerate(self.chain.active_links_mask) if m]
        x0 = np.array([initial_angles[i] for i in active_indices])

        bounds = []
        for i in active_indices:
            link = self.chain.links[i]
            lo = link.bounds[0] if link.bounds else -np.pi
            hi = link.bounds[1] if link.bounds else  np.pi
            bounds.append((lo, hi))

        def cost(x):
            full = list(initial_angles)
            for idx, val in zip(active_indices, x):
                full[idx] = val
            fk = self.chain.forward_kinematics(full)

            # Orientation error — camera pointing
            achieved_pointing = -fk[:3, :3][:, 1]
            dot = np.clip(np.dot(achieved_pointing, target_pointing_vec), -1.0, 1.0)
            orient_err = np.arccos(dot)

            # Position error — penalize Z drift heavily so arm stays at safe height
            achieved_pos = fk[:3, 3]
            pos_err = np.linalg.norm(achieved_pos - target_pos)

            # Orientation is primary, position is secondary — tune the 2.0 weight
            return orient_err + 2.0 * pos_err

        result = minimize(cost, x0, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 1000, "ftol": 1e-9})

        full = list(initial_angles)
        for idx, val in zip(active_indices, result.x):
            full[idx] = val
        return np.array(full)

    def move_to_pose(self, pose: Pose6D, speed: float = 0.3, phase: str = "grip") -> ArmMoveResult:
        target_pos = [pose.x, pose.y, pose.z]
        rot_matrix = R.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
        target_pointing_vec = rot_matrix[:, 2]
        current_angles = self.get_joint_angles().values

        if phase == "scan":
            current_ee_pos = self.chain.forward_kinematics(current_angles)[:3, 3]
            target_pointing_vec = np.array(target_pos) - current_ee_pos
            target_pointing_vec /= np.linalg.norm(target_pointing_vec)

            target_angles = self._solve_orientation_only(
                target_pointing_vec,
                np.array(target_pos),   # pass the full XYZ target so Z is anchored
                current_angles
            )

        else:
            target_angles = self.chain.inverse_kinematics(
                target_position=target_pos,
                target_orientation=None,
                orientation_mode=None,
                initial_position=current_angles,
            )
            fk_matrix = self.chain.forward_kinematics(target_angles)
            achieved_pos = fk_matrix[:3, 3]
            error_val = float(np.linalg.norm(np.array(target_pos) - achieved_pos) * 1000.0)
            success = error_val < 15.0
            message = f"IK solved. Error: {error_val:.2f}mm"

        target_active_degs = np.rad2deg(target_angles[1:6]).tolist()
        current_gripper_deg = float(self.robot.get_observation()['gripper.pos'])

        action_dict = {
            'shoulder_pan.pos':  target_active_degs[0],
            'shoulder_lift.pos': target_active_degs[1],
            'elbow_flex.pos':    target_active_degs[2],
            'wrist_flex.pos':    target_active_degs[3],
            'wrist_roll.pos':    target_active_degs[4],
            'gripper.pos':       current_gripper_deg
        }

        self.robot.send_action(action_dict)
        time.sleep(1.5)

        return ArmMoveResult(
            success=success,
            achieved_pose=self.get_current_pose(),
            error_mm=error_val,
            message=message
        )

    def move_to_home(self) -> ArmMoveResult:
        print("[RoboticArm] Returning to home position...")
        target_angles_rad = [0.0, 0.0, -1.04, 1.04, 0.0, 1.57, 0.0]
        target_active_degs = np.rad2deg(target_angles_rad[1:6]).tolist()
        current_gripper_deg = float(self.robot.get_observation()['gripper.pos'])

        action_dict = {
            'shoulder_pan.pos':  target_active_degs[0],
            'shoulder_lift.pos': target_active_degs[1],
            'elbow_flex.pos':    target_active_degs[2],
            'wrist_flex.pos':    target_active_degs[3],
            'wrist_roll.pos':    target_active_degs[4],
            'gripper.pos':       current_gripper_deg
        }

        self.robot.send_action(action_dict)
        time.sleep(2.5)  # give hardware time to physically arrive

        return ArmMoveResult(success=True, achieved_pose=self.get_current_pose(), error_mm=0.0, message="Homed")
        
    # ── Scanning & Orchestration ────────────────────────────────────

    def get_overhead_pose(self, target: Pose6D) -> Pose6D:
        """Returns a position directly above target at OVERHEAD_HEIGHT for move_to_pose(phase='grip')."""
        return Pose6D(x=target.x, y=target.y, z=self.OVERHEAD_HEIGHT,
                      qx=0.0, qy=0.0, qz=0.0, qw=1.0)

    def rotate_wrist_roll(self, angle_deg: float) -> None:
        """Hold all other joints fixed, rotate only wrist_roll to angle_deg (degrees)."""
        obs = self.robot.get_observation()
        action_dict = {
            'shoulder_pan.pos':  float(obs['shoulder_pan.pos']),
            'shoulder_lift.pos': float(obs['shoulder_lift.pos']),
            'elbow_flex.pos':    float(obs['elbow_flex.pos']),
            'wrist_flex.pos':    float(obs['wrist_flex.pos']),
            'wrist_roll.pos':    angle_deg,
            'gripper.pos':       float(obs['gripper.pos']),
        }
        self.robot.send_action(action_dict)
        time.sleep(1.5)

    def get_arc_trajectory(self, axis: str, target: Pose6D, n_steps: int = 12) -> list[Pose6D]:
        poses = []
        target_pos = np.array([target.x, target.y, target.z])
        steps = np.linspace(-self.sweep_half_length, self.sweep_half_length, n_steps)

        # Arm stays at this height throughout the sweep — tune this value
        SAFE_Z = 0.22  # meters — above the target, below the arm's upper limit

        for step in steps:
            if axis == "x":
                cam_pos = np.array([target_pos[0] + step, target_pos[1], SAFE_Z])
            else:
                cam_pos = np.array([target_pos[0], target_pos[1] + step, SAFE_Z])

            # Vector from camera position down to target
            point_vec = target_pos - cam_pos
            norm = np.linalg.norm(point_vec)
            if norm < 1e-6:
                continue
            point_vec /= norm

            qx, qy, qz, qw = R.from_rotvec(point_vec).as_quat()
            poses.append(Pose6D(
                x=float(cam_pos[0]), y=float(cam_pos[1]), z=float(cam_pos[2]),
                qx=float(qx), qy=float(qy), qz=float(qz), qw=float(qw)
            ))
        return poses

    def pickup_object(self, target_pose: Pose6D, approach_distance: float = 0.05) -> GripResult:
        """Approach above target_pose, descend, activate suction, retract."""
        approach_pos = Pose6D(
            x=target_pose.x, y=target_pose.y, z=target_pose.z + approach_distance,
            qx=target_pose.qx, qy=target_pose.qy, qz=target_pose.qz, qw=target_pose.qw
        )
        
        res = self.move_to_pose(approach_pos, speed=0.3, phase="grip")
        if not res.success: return GripResult(success=False, grip_pose=approach_pos)
            
        res = self.move_to_pose(target_pose, speed=0.1, phase="grip")
        if not res.success: return GripResult(success=False, grip_pose=target_pose)
            
        # NOTE: Your old script didn't define how the suction hardware is triggered.
        # If it's on a relay connected to the AI PC, you'd trigger it here 
        # (e.g., via pyserial to an Arduino, or requests.post to the ESP32).
        print("[RoboticArm] Activating Suction.")
        time.sleep(0.5) 
        
        self.move_to_pose(approach_pos, speed=0.2, phase="grip")
        return GripResult(success=True, grip_pose=target_pose)

    def release_object(self) -> bool:
        """De-energize suction solenoid."""
        print("[RoboticArm] Releasing Suction.")
        return True

    def flip_object(self) -> RotateResult:
        """Turn the held object ~180° about a horizontal axis."""
        current_pose = self.get_current_pose()
        
        r = R.from_quat([current_pose.qx, current_pose.qy, current_pose.qz, current_pose.qw])
        euler = r.as_euler('xyz', degrees=True)
        
        requested_rotation = 180.0
        euler[0] += requested_rotation 
        
        new_q = R.from_euler('xyz', euler, degrees=True).as_quat()
        
        flipped_pose = Pose6D(
            x=current_pose.x, y=current_pose.y, z=current_pose.z + 0.05,
            qx=float(new_q[0]), qy=float(new_q[1]), qz=float(new_q[2]), qw=float(new_q[3])
        )
        
        res = self.move_to_pose(flipped_pose, speed=0.2, phase="scan")
        return RotateResult(
            success=res.success, requested_radians=float(np.radians(requested_rotation)),
            achieved_radians=float(np.radians(requested_rotation)), delta_error=0.0
        )
