import cv2

def display_camera_feeds():
    # Try to open camera feeds (usually the camera indexes are 0, 1, 2, etc.)
    index = 0
    camera_feeds = []

    # Keep trying to open cameras until they are unavailable
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        camera_feeds.append((index, cap))
        index += 1

    if len(camera_feeds) == 0:
        print("No cameras detected!")
        return

    print(f"Found {len(camera_feeds)} camera(s). Displaying feeds...")

    # Display each camera feed in a separate window
    while True:
        for camera_index, cap in camera_feeds:
            ret, frame = cap.read()
            if ret:
                # Display feed with the camera index as the window name
                cv2.imshow(f"Camera {camera_index}", frame)

        # Exit loop when the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release all captures and close windows
    for _, cap in camera_feeds:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_camera_feeds()