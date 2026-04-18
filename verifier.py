# verifier.py
import base64
import cv2
import json
import ollama
import re
import time
import numpy as np

VISION_MODEL = "llama3.2-vision:11b"
WRIST_CAM_INDEX = 2  # UPDATE to whatever worked for you

_wrist_cam = None


def _get_cam():
    global _wrist_cam
    if _wrist_cam is None:
        _wrist_cam = cv2.VideoCapture(WRIST_CAM_INDEX)
        _wrist_cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        _wrist_cam.set(cv2.CAP_PROP_EXPOSURE, -6)
        for _ in range(5):
            _wrist_cam.read()
    return _wrist_cam


def _capture_frame():
    """Grab a fresh frame, flushing stale buffer content."""
    cam = _get_cam()
    for _ in range(3):
        cam.read()
    ret, frame = cam.read()
    if not ret:
        raise RuntimeError("Failed to capture wrist cam frame")
    # Crop to center — reduces background noise for the VLM
    h, w = frame.shape[:2]
    crop = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    return crop


def snap_wrist(save_path=None):
    """Grab a frame and return as base64-encoded JPEG."""
    frame = _capture_frame()
    if save_path:
        cv2.imwrite(save_path, frame)
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode()


def snap_wrist_raw(save_path=None):
    """Grab a frame and return as a numpy array (for later stitching)."""
    frame = _capture_frame()
    if save_path:
        cv2.imwrite(save_path, frame)
    return frame


def capture_baseline(save_path="baseline_empty.jpg"):
    """Capture a reference 'empty gripper (closed)' frame as a numpy array."""
    frame = snap_wrist_raw(save_path=save_path)
    print(f"Baseline captured to {save_path}")
    return frame


def _stitch_side_by_side(left_img, right_img, label_left="BEFORE", label_right="AFTER"):
    """Stitch two images side by side with labels, return base64 JPEG."""
    # Match heights
    h = min(left_img.shape[0], right_img.shape[0])
    left = cv2.resize(left_img, (int(left_img.shape[1] * h / left_img.shape[0]), h))
    right = cv2.resize(right_img, (int(right_img.shape[1] * h / right_img.shape[0]), h))

    # Add a thin separator
    separator = np.zeros((h, 8, 3), dtype=np.uint8)
    separator[:, :] = (255, 255, 255)

    combined = np.hstack([left, separator, right])

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, label_left, (10, 30), font, 0.9, (0, 255, 0), 2)
    cv2.putText(combined, label_right, (left.shape[1] + 18, 30), font, 0.9, (0, 255, 0), 2)

    _, buf = cv2.imencode(".jpg", combined)
    return base64.b64encode(buf).decode(), combined


COMPARE_PROMPT = """This is ONE image containing TWO side-by-side photos from a robot arm's wrist camera.

LEFT photo (labeled "BEFORE"): gripper is CLOSED and EMPTY. This is the baseline — no object.
RIGHT photo (labeled "AFTER"): gripper has attempted a grasp. Unknown whether it succeeded.

Compare ONLY the region BETWEEN the gripper fingers in both photos.

Question: Is there MORE material visible between the fingers in the RIGHT photo than in the LEFT photo?
- If yes → the gripper is holding something (grasp succeeded)
- If the two look the same between the fingers → gripper is empty (grasp failed)

Ignore background differences. Focus on what's between the fingertips.

Respond with ONLY this JSON:
{"grasped": true|false, "confidence": "high"|"medium"|"low", "reason": "describe specifically what's different (or not) between the fingers in the two photos"}
"""


def _extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def verify_grasp_vs_baseline(baseline_frame, save_frame_path="last_verify.jpg",
                              save_comparison_path="last_comparison.jpg"):
    """Compare current wrist view to the baseline empty-closed-gripper frame."""
    current_frame = snap_wrist_raw(save_path=save_frame_path)

    combined_b64, combined_img = _stitch_side_by_side(baseline_frame, current_frame)
    cv2.imwrite(save_comparison_path, combined_img)

    t0 = time.time()
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": COMPARE_PROMPT,
            "images": [combined_b64],  # single stitched image
        }],
        format="json",
        options={"temperature": 0.1},
    )
    elapsed = time.time() - t0
    print(f"  [verify took {elapsed:.1f}s]")

    raw = response["message"]["content"]
    return _extract_json(raw)


if __name__ == "__main__":
    print("Test harness — ensure gripper is at inspect pose and CLOSED and EMPTY.")
    input("Press Enter to capture baseline...")
    baseline = capture_baseline()

    while True:
        print("\nTest: adjust gripper/shirt, then verify.")
        choice = input("Press Enter to verify (or 'q' to quit): ")
        if choice.strip().lower() == "q":
            break
        result = verify_grasp_vs_baseline(baseline)
        print(json.dumps(result, indent=2))
        print("(See last_comparison.jpg for what the VLM saw.)")