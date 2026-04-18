"""
Record arm waypoints with a live camera preview.

Move the arm manually to a position. The preview window shows what the camera
sees, plus a live SIFT keypoint count (low count = COLMAP will struggle here).
Press Space or Enter to record the current joint angles. Press 'q' to finish.

Usage:
    python write_waypoints.py [--device 2]
"""
import argparse
import time

import cv2

from interfaces.robotic_arm import RoboticArm


def open_camera(preferred: int | None, exposure: float | None = None) -> tuple[cv2.VideoCapture, int]:
    indices = [preferred] if preferred is not None else list(range(8))
    for i in indices:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                if exposure is not None:
                    # 0.25 = manual mode on most UVC webcams (0.75 = auto)
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                return cap, i
        cap.release()
    raise RuntimeError("No working camera found. Pass --device N to choose one.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=None,
                        help="Camera index (e.g. 2 for /dev/video2)")
    parser.add_argument("--exposure", type=float, default=None,
                        help="Manual exposure (UVC: try -6 darker → -2 brighter; "
                             "or absolute µs like 100 on some cams)")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--urdf", default="models/so101.urdf")
    args = parser.parse_args()

    print("Connecting to arm...")
    arm = RoboticArm(port=args.port, urdf_path=args.urdf)

    cap, idx = open_camera(args.device, exposure=args.exposure)
    print(f"Using /dev/video{idx}")

    sift = cv2.SIFT_create()
    angles: list[list[float]] = []
    last_kp_check = 0.0
    kp_count = 0

    print("\nLive preview open.")
    print("  Space/Enter — record current arm angles")
    print("  q          — finish and print waypoints\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed.")
                break

            # Recompute keypoint count once a second so the preview stays smooth
            now = time.time()
            if now - last_kp_check > 1.0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kp_count = len(sift.detect(gray, None))
                last_kp_check = now

            # Overlay
            display = frame.copy()
            kp_color = (0, 220, 0) if kp_count >= 300 else (0, 165, 255)
            cv2.putText(display, f"Recorded: {len(angles)}",
                        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
            cv2.putText(display, f"SIFT keypoints: {kp_count}"
                                 + (" (LOW)" if kp_count < 300 else ""),
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, kp_color, 2)
            cv2.putText(display, "Space/Enter = record   q = done",
                        (10, display.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Waypoint Recorder", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord(" "), 13):  # Space or Enter
                wp = arm.get_joint_angles().values
                angles.append(wp)
                print(f"  Recorded waypoint #{len(angles)} (kp={kp_count})")
                # Brief flash so the user knows it captured
                flash = display.copy()
                flash[:] = (0, 220, 0)
                cv2.addWeighted(display, 0.5, flash, 0.5, 0, flash)
                cv2.imshow("Waypoint Recorder", flash)
                cv2.waitKey(80)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nFinal angles ({len(angles)} waypoints):")
    print(angles)


if __name__ == "__main__":
    main()
