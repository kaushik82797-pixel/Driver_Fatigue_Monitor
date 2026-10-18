"""
Driver Fatigue & Distraction Safety Monitor
Standalone Prototype 1: Camera Feed and Face Detection (OpenCV Haar Cascade)

This prototype verifies webcam access, live frame capture, and basic
face detection using OpenCV's built-in Haar Cascade classifier.

Controls:
    Press 'q' in the video window to safely exit and release the webcam.
"""

import sys
import os
import argparse
import urllib.request
import cv2


def get_face_cascade():
    """
    Locates or retrieves the Haar cascade frontal face XML model.
    """
    xml_filename = "haarcascade_frontalface_default.xml"

    # 1. Check cv2 data directory
    cascade_path = os.path.join(cv2.data.haarcascades, xml_filename)
    if os.path.exists(cascade_path):
        return cv2.CascadeClassifier(cascade_path)

    # 2. Check local directory
    local_path = os.path.join(os.path.dirname(__file__), xml_filename)
    if os.path.exists(local_path):
        return cv2.CascadeClassifier(local_path)

    # 3. Auto-download from official OpenCV repository if missing
    try:
        print("[*] Downloading haarcascade_frontalface_default.xml from OpenCV repository...")
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{xml_filename}"
        urllib.request.urlretrieve(url, cascade_path)
        if os.path.exists(cascade_path):
            return cv2.CascadeClassifier(cascade_path)
    except Exception as e:
        print(f"[!] Warning: Could not auto-download cascade: {e}")

    # Fallback to direct name (OpenCV internal lookup)
    return cv2.CascadeClassifier(xml_filename)


def run_camera_test(camera_index: int = 0, max_frames: int = None, headless: bool = False):
    """
    Opens the specified camera, runs face detection loop, and displays overlay.

    Args:
        camera_index: Index of the camera to open (default: 0).
        max_frames: Optional frame limit (useful for non-interactive automated testing).
        headless: If True, skips cv2.imshow (for headless environments).
    """
    print("=" * 65)
    print(" Driver Fatigue & Distraction Safety Monitor - Camera Test ")
    print("=" * 65)
    print(f"Initializing camera (index {camera_index})...")

    # Load Haar cascade
    face_cascade = get_face_cascade()
    if face_cascade.empty():
        print("[ERROR] Failed to load Haar cascade classifier.")
        print("Please verify that haarcascade_frontalface_default.xml is accessible.")
        return 1

    # Attempt to open camera using DirectShow backend on Windows first for fast startup
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)

    # Fallback to default backend if DirectShow fails
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    # Validate camera connection
    if not cap.isOpened():
        print("\n" + "!" * 65)
        print(f"[ERROR] Could not open camera at index {camera_index}.")
        print("!" * 65)
        print("Troubleshooting Steps:")
        print("  1. Ensure your webcam is plugged in and recognized by your OS.")
        print("  2. Verify that no other application (Zoom, Teams, Browser) is using the camera.")
        print("  3. On Windows, check Camera Privacy Settings:")
        print("     Settings -> Privacy & Security -> Camera -> Allow apps to access camera.")
        print("  4. If you have multiple cameras, try running with: --camera 1")
        print("!" * 65 + "\n")
        return 1

    # Fetch camera resolution properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[OK] Camera initialized successfully ({width}x{height} @ ~{fps:.1f} FPS)")
    print("[*] Starting video loop. Press 'q' in the window to quit.\n")

    frame_count = 0
    window_name = "Driver Safety Monitor - Camera Test Prototype"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Failed to grab frame from camera. Exiting loop.")
                break

            frame_count += 1

            # Convert frame to grayscale for Haar cascade face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces: scaleFactor=1.1, minNeighbors=5, minimum face size 60x60
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Determine detection status and color
            num_faces = len(faces)
            if num_faces > 0:
                status_text = f"Face Detected ({num_faces})"
                status_color = (0, 220, 0)   # Green in BGR
                # Draw bounding box for each detected face
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), status_color, 2)
                    cv2.putText(
                        frame,
                        "Driver Face",
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        status_color,
                        2
                    )
            else:
                status_text = "No Face Detected"
                status_color = (0, 0, 220)   # Red in BGR

            # Draw HUD status overlay on top-left of the frame
            cv2.rectangle(frame, (10, 10), (320, 75), (20, 20, 20), -1)
            cv2.rectangle(frame, (10, 10), (320, 75), (60, 60, 60), 1)

            cv2.putText(
                frame,
                f"Status: {status_text}",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2
            )
            cv2.putText(
                frame,
                "Prototype 1: Face Detection Only",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1
            )

            # Footer instructions
            cv2.putText(
                frame,
                "Press 'q' to exit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1
            )

            # Display window unless running in headless test mode
            if not headless:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[*] 'q' pressed by user. Stopping camera feed.")
                    break

            # Frame limit for automated validation
            if max_frames and frame_count >= max_frames:
                print(f"[*] Reached test limit of {max_frames} frames. Stopping.")
                break

    except KeyboardInterrupt:
        print("\n[*] Interrupted by keyboard. Stopping.")
    finally:
        # Release hardware and destroy windows
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        print(f"[OK] Camera released and windows closed. (Total frames processed: {frame_count})")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Driver Safety Monitor Camera Test Prototype")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--frames", type=int, default=None, help="Stop after N frames (optional)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display window")
    args = parser.parse_args()

    exit_code = run_camera_test(
        camera_index=args.camera,
        max_frames=args.frames,
        headless=args.headless
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
