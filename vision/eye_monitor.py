"""
Driver Fatigue & Distraction Safety Monitor
Standalone Prototype 2: Eye Landmark Detection and EAR-based Drowsiness Monitoring

Uses MediaPipe FaceLandmarker (Tasks Vision API) and OpenCV to:
1. Detect 3D facial landmarks on live webcam stream.
2. Extract eye contour landmarks for both left and right eyes.
3. Compute Eye Aspect Ratio (EAR) using Euclidean distance ratios.
4. Detect prolonged eye closure via configurable consecutive-frame thresholding.
5. Provide live visual telemetry: Eyes Open, Eyes Closed, and Drowsiness Warning.

Disclaimer:
This prototype is for educational, demonstration, and research purposes.
It is NOT a certified medical or automotive safety device. Detection thresholds
vary by individual facial geometry, camera angles, and lighting, requiring
practical testing and calibration.

Controls:
    Press 'q' in the video window to quit and release the webcam.
"""

import sys
import os
import argparse
import urllib.request
import numpy as np
import cv2

# Landmark indices for EAR calculation (MediaPipe Face Mesh)
# Right Eye (Subject's right / viewer's left side):
# p1: 33 (outer corner), p2: 160 (upper), p3: 158 (upper),
# p4: 133 (inner corner), p5: 153 (lower), p6: 144 (lower)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Left Eye (Subject's left / viewer's right side):
# p1: 362 (inner corner), p2: 385 (upper), p3: 387 (upper),
# p4: 263 (outer corner), p5: 373 (lower), p6: 380 (lower)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Full contour loop indices for drawing smooth eye boundaries
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def ensure_model_asset(model_path: str) -> str:
    """
    Ensures that the MediaPipe face_landmarker.task model file is available.
    Downloads it from official Google MediaPipe repository if missing.
    """
    if os.path.exists(model_path):
        return model_path

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    print(f"[*] Downloading MediaPipe model asset to {model_path}...")
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"[OK] Downloaded model asset ({os.path.getsize(model_path)} bytes)")
    except Exception as e:
        print(f"[ERROR] Failed to download model asset: {e}")
        print("Please check your internet connection or place face_landmarker.task manually.")
        return None

    return model_path


def initialize_landmarker(model_path: str):
    """
    Initializes MediaPipe FaceLandmarker using the Tasks Python Vision API.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    valid_model_path = ensure_model_asset(model_path)
    if not valid_model_path or not os.path.exists(valid_model_path):
        return None

    base_options = BaseOptions(model_asset_path=valid_model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5
    )
    return vision.FaceLandmarker.create_from_options(options)


def calculate_ear(eye_points: np.ndarray) -> float:
    """
    Calculates Eye Aspect Ratio (EAR) based on Soukupova and Cech (2016):
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    if len(eye_points) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_points[:6]

    # Vertical Euclidean distances
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)

    # Horizontal Euclidean distance
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal <= 1e-6:
        return 0.0

    return float((v1 + v2) / (2.0 * horizontal))


def draw_eye_overlay(frame: np.ndarray, points: np.ndarray, color: tuple):
    """
    Draws eye contour polygon and landmark points on the frame.
    """
    if len(points) == 0:
        return
    pts = points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
    for pt in points:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, color, -1)


def run_eye_monitor(
    camera_index: int = 0,
    ear_threshold: float = 0.22,
    consecutive_frames: int = 20,
    max_frames: int = None,
    headless: bool = False,
    model_path: str = "models/face_landmarker.task"
):
    """
    Main monitoring loop for eye landmark detection and drowsiness alerts.
    """
    import mediapipe as mp

    print("=" * 68)
    print(" Driver Fatigue & Distraction Safety Monitor - Eye Monitor (EAR) ")
    print("=" * 68)
    print(f"Camera Index:             {camera_index}")
    print(f"EAR Threshold:            {ear_threshold:.2f}")
    print(f"Consecutive Frame Limit:  {consecutive_frames} frames (~{consecutive_frames/25.0:.1f}s at 25 FPS)")
    print(f"Model Path:               {model_path}")
    print("=" * 68)

    # 1. Initialize MediaPipe Face Landmarker
    landmarker = initialize_landmarker(model_path)
    if landmarker is None:
        print("[ERROR] Could not initialize MediaPipe FaceLandmarker.")
        return 1

    # 2. Open camera (DirectShow on Windows for fast access)
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"\n[ERROR] Failed to open camera (index {camera_index}).")
        print("Please check that your webcam is connected and not in use by another app.")
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Camera connected ({width}x{height})")
    print("[*] Starting monitoring. Press 'q' in video window to exit.\n")

    frame_count = 0
    closed_frames = 0
    window_name = "Driver Safety Monitor - Prototype 2 (Eye & EAR Monitor)"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Could not read frame from camera. Exiting.")
                break

            frame_count += 1
            h, w = frame.shape[:2]

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect face landmarks
            detection_result = landmarker.detect(mp_image)

            face_detected = len(detection_result.face_landmarks) > 0
            ear_left = 0.0
            ear_right = 0.0
            avg_ear = 0.0

            if face_detected:
                landmarks = detection_result.face_landmarks[0]

                def to_pixel(idx):
                    lm = landmarks[idx]
                    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

                # Extract landmark coordinates for EAR
                r_pts = np.array([to_pixel(i) for i in RIGHT_EYE_INDICES])
                l_pts = np.array([to_pixel(i) for i in LEFT_EYE_INDICES])

                # Extract contour points for drawing
                r_contour = np.array([to_pixel(i) for i in RIGHT_EYE_CONTOUR])
                l_contour = np.array([to_pixel(i) for i in LEFT_EYE_CONTOUR])

                # Compute EAR
                ear_right = calculate_ear(r_pts)
                ear_left = calculate_ear(l_pts)
                avg_ear = (ear_right + ear_left) / 2.0

                # Evaluate eye closure state
                if avg_ear < ear_threshold:
                    closed_frames += 1
                else:
                    closed_frames = max(0, closed_frames - 1)

                # Determine driver alertness status
                if closed_frames >= consecutive_frames:
                    status_text = "DROWSINESS WARNING!"
                    status_color = (0, 0, 255)       # Red
                    contour_color = (0, 0, 255)
                    alert_active = True
                elif closed_frames > 0:
                    status_text = f"Eyes Closed ({closed_frames}/{consecutive_frames})"
                    status_color = (0, 165, 255)     # Orange / Amber
                    contour_color = (0, 165, 255)
                    alert_active = False
                else:
                    status_text = "Eyes Open"
                    status_color = (0, 220, 0)       # Green
                    contour_color = (0, 220, 0)
                    alert_active = False

                # Draw eye contours on live frame
                draw_eye_overlay(frame, r_contour, contour_color)
                draw_eye_overlay(frame, l_contour, contour_color)

            else:
                # No face detected in frame
                closed_frames = 0
                status_text = "No Face Detected"
                status_color = (128, 128, 128)
                alert_active = False

            # Render Telemetry HUD
            # Top card background
            cv2.rectangle(frame, (10, 10), (460, 115), (15, 15, 15), -1)
            cv2.rectangle(frame, (10, 10), (460, 115), (50, 50, 50), 1)

            # Status line
            cv2.putText(
                frame,
                f"Status: {status_text}",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                status_color,
                2
            )

            # EAR telemetry line
            ear_info = f"EAR: {avg_ear:.3f} (R: {ear_right:.2f}, L: {ear_left:.2f}) | Thresh: {ear_threshold:.2f}"
            cv2.putText(
                frame,
                ear_info,
                (20, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (230, 230, 230),
                1
            )

            # Frame counter line
            prog_info = f"Closure Frames: {closed_frames} / {consecutive_frames}"
            cv2.putText(
                frame,
                prog_info,
                (20, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (180, 180, 180),
                1
            )

            # Disclaimer banner
            cv2.putText(
                frame,
                "Prototype 2: EAR Drowsiness Monitor (Testing & Calibration Only)",
                (20, 107),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (130, 130, 130),
                1
            )

            # Flashing border warning when drowsiness alert is active
            if alert_active and (frame_count % 6 < 3):
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

            # Footer instructions
            cv2.putText(
                frame,
                "Press 'q' to exit",
                (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )

            # GUI display unless in headless mode
            if not headless:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[*] User pressed 'q'. Exiting eye monitor.")
                    break

            if max_frames and frame_count >= max_frames:
                print(f"[*] Reached max test frames ({max_frames}). Stopping.")
                break

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        print(f"[OK] Camera released and clean shutdown completed. (Processed {frame_count} frames)")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Driver Fatigue & Distraction Safety Monitor - Eye & EAR Monitor")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--ear-thresh", type=float, default=0.22, help="EAR threshold for eye closure (default: 0.22)")
    parser.add_argument("--consecutive-frames", type=int, default=20, help="Consecutive closed frames before warning (default: 20)")
    parser.add_argument("--frames", type=int, default=None, help="Exit after N frames (for automated verification)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI display window")
    parser.add_argument("--model-path", type=str, default="models/face_landmarker.task", help="Path to MediaPipe model asset")
    args = parser.parse_args()

    exit_code = run_eye_monitor(
        camera_index=args.camera,
        ear_threshold=args.ear_thresh,
        consecutive_frames=args.consecutive_frames,
        max_frames=args.frames,
        headless=args.headless,
        model_path=args.model_path
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
