"""
drive.py — interactive REPL for driving the SO-101 follower.

Useful for jogging during mat-marker measurement, testing motions,
and general tinkering. Bypasses the safety layer, so use small inputs.

Run: python3 drive.py
     python3 drive.py --port /dev/ttyACM1

Type 'help' at the prompt for commands.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import traceback

from robot import Robot, JOINT_ORDER, JOINT_LIMITS_DEG


HELP = """
Commands:
  state                            show joint state + FK gripper tip
  home                             go to home pose
  set <joint> <deg> [speed]        absolute joint angle
  jog <joint> <delta_deg> [speed]  relative joint delta
  move <x> <y> <z> [speed]         IK move to (x, y, z) mm
  grip <0-1> [speed]               gripper width (0=closed, 1=open)
  raw <j1> <j2> <j3> <j4> <j5> <j6> [speed]
                                   command all 6 joints at once (degrees)
  limits                           print joint limits
  help                             this help
  quit | q | exit                  disconnect and exit

Joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
Speed default 0.3, range [0.05, 1.0].
"""


def print_state(robot: Robot):
    joints = robot.get_joint_positions_deg()
    print("Joint state (deg):")
    for name in JOINT_ORDER:
        val = joints.get(name, float("nan"))
        lo, hi = JOINT_LIMITS_DEG[name]
        print(f"  {name:15s} {val:+7.2f}    [{lo}, {hi}]")
    try:
        x, y, z = robot.get_gripper_tip_mm()
        print(f"Gripper tip (FK, mm):  x={x:+7.1f}  y={y:+7.1f}  z={z:+7.1f}")
    except Exception as e:
        print(f"FK error: {e}")


def cmd_set(robot, args):
    joint = args[0]
    if joint not in JOINT_ORDER:
        print(f"Unknown joint. Valid: {JOINT_ORDER}")
        return
    deg = float(args[1])
    speed = float(args[2]) if len(args) > 2 else 0.3
    lo, hi = JOINT_LIMITS_DEG[joint]
    if not (lo <= deg <= hi):
        print(f"Refusing: {deg}° outside [{lo}, {hi}]")
        return
    robot.set_joint_positions_deg({joint: deg}, speed=speed)


def cmd_jog(robot, args):
    joint = args[0]
    if joint not in JOINT_ORDER:
        print(f"Unknown joint.")
        return
    delta = float(args[1])
    speed = float(args[2]) if len(args) > 2 else 0.3
    current = robot.get_joint_positions_deg()[joint]
    target = current + delta
    lo, hi = JOINT_LIMITS_DEG[joint]
    if not (lo <= target <= hi):
        print(f"Refusing: {target:.1f}° outside [{lo}, {hi}]")
        return
    print(f"  {joint}: {current:.1f}° → {target:.1f}°")
    robot.set_joint_positions_deg({joint: target}, speed=speed)


def cmd_move(robot, args):
    x, y, z = float(args[0]), float(args[1]), float(args[2])
    speed = float(args[3]) if len(args) > 3 else 0.3
    print(f"  IK → ({x}, {y}, {z}) mm")
    try:
        robot.move_ik(x, y, z, speed=speed)
    except Exception as e:
        print(f"IK failed: {e}")


def cmd_grip(robot, args):
    w = float(args[0])
    speed = float(args[1]) if len(args) > 1 else 0.5
    robot.set_gripper(w, speed=speed)


def cmd_raw(robot, args):
    if len(args) < 6:
        print("Need 6 joint values.")
        return
    targets = {name: float(args[i]) for i, name in enumerate(JOINT_ORDER)}
    speed = float(args[6]) if len(args) > 6 else 0.3
    for name, val in targets.items():
        lo, hi = JOINT_LIMITS_DEG[name]
        if not (lo <= val <= hi):
            print(f"Refusing: {name}={val}° outside [{lo}, {hi}]")
            return
    robot.set_joint_positions_deg(targets, speed=speed)


def cmd_limits(_robot, _args):
    print("Joint limits (deg):")
    for name, (lo, hi) in JOINT_LIMITS_DEG.items():
        print(f"  {name:15s}  [{lo}, {hi}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    args = ap.parse_args()

    try:
        robot = Robot(port=args.port)
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    print("Connected. Type 'help' for commands.\n")
    print_state(robot)

    handlers = {
        "state":  lambda r, a: print_state(r),
        "home":   lambda r, a: (r.home(), print("Homed.")),
        "set":    cmd_set,
        "jog":    cmd_jog,
        "move":   cmd_move,
        "grip":   cmd_grip,
        "raw":    cmd_raw,
        "limits": cmd_limits,
    }

    try:
        while True:
            try:
                line = input("so101> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in ("quit", "q", "exit"):
                break
            if line == "help":
                print(HELP)
                continue
            parts = shlex.split(line)
            cmd, cmd_args = parts[0], parts[1:]
            handler = handlers.get(cmd)
            if handler is None:
                print(f"Unknown: {cmd}. Type 'help'.")
                continue
            try:
                handler(robot, cmd_args)
            except Exception:
                traceback.print_exc()
    finally:
        print("Disconnecting...")
        robot.close()


if __name__ == "__main__":
    main()