"""
Driver Fatigue & Distraction Safety Monitor
Standalone Prototype 3: Mouth Landmark Detection and Yawning Detection (MAR)

Uses MediaPipe FaceLandmarker (Tasks Vision API) and OpenCV to:
1. Detect 3D facial landmarks on live webcam stream.
2. Extract mouth and lip contour landmarks.
3. Compute Mouth Aspect Ratio (MAR) using Euclidean distance ratios.
4. Detect prolonged wide mouth opening (indicative of yawning) via configurable consecutive-frame thresholding.
5. Provide live visual telemetry: Mouth Normal, Possible Yawn, and Yawning Alert.

Mouth Aspect Ratio (MAR) Formula:
    MAR = (||p81 - p178|| + ||p13 - p14|| + ||p311 - p402||) / (2 * ||p61 - p291||)

Where (MediaPipe Face Mesh indices):
    - p61:  Left mouth corner (Subject's right / viewer's left)
    - p291: Right mouth corner (Subject's left / viewer's right)
    - p13:  Upper inner lip center
    - p14:  Lower inner lip center
    - p81:  Upper inner lip left
    - p178: Lower inner lip left
    - p311: Upper inner lip right
    - p402: Lower inner lip right

Disclaimer:
This prototype is an educational/coursework demonstration and is NOT a certified
automotive safety or driver-monitoring product. Mouth opening alone cannot definitively
distinguish yawning from singing, talking, or laughing. Practical deployment requires
individual threshold calibration and multi-modal fusion.

Controls:
    Press 'q' in the video window to quit and release the webcam.
"""

import sys
import os
import argparse
import urllib.request
import numpy as np
import cv2

# MediaPipe Face Mesh landmark indices for MAR calculation
MOUTH_CORNER_LEFT = 61      # Subject's right / viewer's left
MOUTH_CORNER_RIGHT = 291   # Subject's left / viewer's right
UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14
UPPER_LIP_LEFT = 81
LOWER_LIP_LEFT = 178
UPPER_LIP_RIGHT = 311
LOWER_LIP_RIGHT = 402

KEY_MOUTH_INDICES = [
    MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT,
    UPPER_LIP_CENTER, LOWER_LIP_CENTER,
    UPPER_LIP_LEFT, LOWER_LIP_LEFT,
    UPPER_LIP_RIGHT, LOWER_LIP_RIGHT
]


def ensure_model_asset(model_path: str) -> str:
    """
    Ensures that the MediaPipe face_landmarker.task model file is available.
    Reuses existing file if present; downloads from Google repository if missing.
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


def calculate_mar(
    corner_left: np.ndarray,
    corner_right: np.ndarray,
    top_center: np.ndarray,
    bot_center: np.ndarray,
    top_left: np.ndarray,
    bot_left: np.ndarray,
    top_right: np.ndarray,
    bot_right: np.ndarray
) -> float:
    """
    Calculates Mouth Aspect Ratio (MAR):
        MAR = (||top_left - bot_left|| + ||top_center - bot_center|| + ||top_right - bot_right||) / (2 * ||corner_left - corner_right||)
    """
    # Vertical Euclidean distances
    v_center = np.linalg.norm(top_center - bot_center)
    v_left = np.linalg.norm(top_left - bot_left)
    v_right = np.linalg.norm(top_right - bot_right)

    # Horizontal mouth width
    horizontal = np.linalg.norm(corner_left - corner_right)

    if horizontal <= 1e-6:
        return 0.0

    return float((v_left + v_center + v_right) / (2.0 * horizontal))


def draw_mouth_mesh(frame: np.ndarray, landmarks, w: int, h: int, color: tuple):
    """
    Draws the full mouth mesh connections and key points on the frame.
    """
    from mediapipe.tasks.python.vision import FaceLandmarksConnections

    def to_px(idx):
        lm = landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))

    # Draw all 40 lips connection lines
    for conn in FaceLandmarksConnections.FACE_LANDMARKS_LIPS:
        pt1 = to_px(conn.start)
        pt2 = to_px(conn.end)
        cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)

    # Highlight the 8 key MAR measurement points
    for idx in KEY_MOUTH_INDICES:
        pt = to_px(idx)
        cv2.circle(frame, pt, 2, (0, 255, 255), -1)


def run_yawn_monitor(
    camera_index: int = 0,
    mar_threshold: float = 0.50,
    consecutive_frames: int = 25,
    max_frames: int = None,
    headless: bool = False,
    model_path: str = "models/face_landmarker.task"
):
    """
    Main monitoring loop for mouth landmark detection and yawning alerts.
    """
    import mediapipe as mp

    print("=" * 68)
    print(" Driver Fatigue & Distraction Safety Monitor - Yawn Monitor (MAR) ")
    print("=" * 68)
    print(f"Camera Index:             {camera_index}")
    print(f"MAR Threshold:            {mar_threshold:.2f}")
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
    yawn_frames = 0
    window_name = "Driver Safety Monitor - Prototype 3 (Yawn & MAR Monitor)"

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
            mar_value = 0.0

            if face_detected:
                landmarks = detection_result.face_landmarks[0]

                def to_pixel(idx):
                    lm = landmarks[idx]
                    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

                # Extract key landmarks for MAR calculation
                p_c_left = to_pixel(MOUTH_CORNER_LEFT)
                p_c_right = to_pixel(MOUTH_CORNER_RIGHT)
                p_top_c = to_pixel(UPPER_LIP_CENTER)
                p_bot_c = to_pixel(LOWER_LIP_CENTER)
                p_top_l = to_pixel(UPPER_LIP_LEFT)
                p_bot_l = to_pixel(LOWER_LIP_LEFT)
                p_top_r = to_pixel(UPPER_LIP_RIGHT)
                p_bot_r = to_pixel(LOWER_LIP_RIGHT)

                # Compute MAR
                mar_value = calculate_mar(
                    p_c_left, p_c_right,
                    p_top_c, p_bot_c,
                    p_top_l, p_bot_l,
                    p_top_r, p_bot_r
                )

                # Evaluate yawning condition
                if mar_value >= mar_threshold:
                    yawn_frames += 1
                else:
                    yawn_frames = max(0, yawn_frames - 1)

                # Determine driver yawning alertness state
                if yawn_frames >= consecutive_frames:
                    status_text = "YAWNING ALERT!"
                    status_color = (0, 0, 255)       # Red
                    mesh_color = (0, 0, 255)
                    alert_active = True
                elif yawn_frames > 0:
                    status_text = f"Possible Yawn ({yawn_frames}/{consecutive_frames})"
                    status_color = (0, 165, 255)     # Amber / Orange
                    mesh_color = (0, 165, 255)
                    alert_active = False
                else:
                    status_text = "Mouth Normal"
                    status_color = (0, 220, 0)       # Green
                    mesh_color = (0, 220, 0)
                    alert_active = False

                # Draw mouth mesh and key landmarks
                draw_mouth_mesh(frame, landmarks, w, h, mesh_color)

            else:
                # No face detected
                yawn_frames = 0
                status_text = "No Face Detected"
                status_color = (128, 128, 128)
                alert_active = False

            # Render Telemetry HUD Card
            cv2.rectangle(frame, (10, 10), (470, 115), (15, 15, 15), -1)
            cv2.rectangle(frame, (10, 10), (470, 115), (50, 50, 50), 1)

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

            # MAR telemetry line
            mar_info = f"MAR: {mar_value:.3f} | Thresh: {mar_threshold:.2f}"
            cv2.putText(
                frame,
                mar_info,
                (20, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (230, 230, 230),
                1
            )

            # Frame counter line
            prog_info = f"Yawn Frames: {yawn_frames} / {consecutive_frames}"
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
                "Prototype 3: MAR Yawn Monitor (Coursework Prototype - Calibration Required)",
                (20, 107),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (130, 130, 130),
                1
            )

            # Flashing border warning when yawning alert is active
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
                    print("[*] User pressed 'q'. Exiting yawn monitor.")
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
    parser = argparse.ArgumentParser(description="Driver Fatigue & Distraction Safety Monitor - Yawn & MAR Monitor")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--mar-thresh", type=float, default=0.50, help="MAR threshold for yawning (default: 0.50)")
    parser.add_argument("--consecutive-frames", type=int, default=25, help="Consecutive open frames before warning (default: 25)")
    parser.add_argument("--frames", type=int, default=None, help="Exit after N frames (for automated verification)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI display window")
    parser.add_argument("--model-path", type=str, default="models/face_landmarker.task", help="Path to MediaPipe model asset")
    args = parser.parse_args()

    exit_code = run_yawn_monitor(
        camera_index=args.camera,
        mar_threshold=args.mar_thresh,
        consecutive_frames=args.consecutive_frames,
        max_frames=args.frames,
        headless=args.headless,
        model_path=args.model_path
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
