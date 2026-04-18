"""
scan_bus.py — what motors are physically on the bus?

Runs a broadcast ping on the Feetech bus and prints every servo ID that
responds, regardless of what lerobot expected to find. Use this to tell
whether motors are electrically disconnected vs. misconfigured.

Run: python3 scan_bus.py
     python3 scan_bus.py --port /dev/ttyACM1   # override port
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    args = ap.parse_args()

    try:
        from lerobot.motors.feetech.feetech import FeetechMotorsBus
    except ImportError as e:
        print(f"Couldn't import FeetechMotorsBus: {e}")
        print("Your lerobot version may use a different import path. Try:")
        print("  from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus")
        sys.exit(1)

    bus = FeetechMotorsBus(port=args.port, motors={})
    try:
        bus.connect()
    except Exception as e:
        print(f"Connection to {args.port} failed: {e}")
        print("\nPossible fixes:")
        print(f"  - Wrong port. Try: ls /dev/ttyACM* /dev/ttyUSB*")
        print(f"  - Permission denied: sudo chmod 666 {args.port}")
        print(f"  - Arm not powered: check the barrel jack")
        sys.exit(1)

    print(f"Connected to {args.port}. Broadcasting ping to IDs 1-253...\n")
    try:
        found = bus.broadcast_ping()
    except Exception as e:
        print(f"broadcast_ping failed: {e}")
        bus.disconnect()
        sys.exit(1)

    if not found:
        print("NO MOTORS RESPONDED.")
        print("  - Check power (LEDs on the servos)")
        print("  - Check the USB cable")
        print("  - Check that you're on the right port")
        bus.disconnect()
        sys.exit(1)

    print(f"Found {len(found)} motor(s):")
    for motor_id in sorted(found.keys()):
        print(f"  ID {motor_id}: model {found[motor_id]}")

    expected = {1, 2, 3, 4, 5, 6}
    actual = set(found.keys())
    missing = expected - actual
    extra = actual - expected

    print()
    if missing:
        print(f"MISSING (expected but not found): {sorted(missing)}")
    if extra:
        print(f"EXTRA (found but not expected): {sorted(extra)}")
    if not missing and not extra:
        print("All 6 expected motors are present. ✓")

    bus.disconnect()


if __name__ == "__main__":
    main()