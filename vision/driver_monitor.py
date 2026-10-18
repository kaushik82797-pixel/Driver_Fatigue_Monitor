"""
Driver Fatigue & Distraction Safety Monitor
Standalone Prototype 4: Combined Eye-Closure & Yawning Monitoring Engine

Performs simultaneous real-time driver state monitoring using a single webcam
stream and a single MediaPipe FaceLandmarker instance per frame.

Metrics:
    - Eye Aspect Ratio (EAR) for both eyes (Soukupova & Cech)
    - Mouth Aspect Ratio (MAR) for mouth opening
    - Dual independent state machines for drowsiness & yawning detection

Disclaimer:
This prototype is an educational and coursework demonstration. It is NOT
certified automotive safety or medical monitoring equipment. Thresholds require
individual testing and calibration.

Controls:
    Press 'q' in the video window to quit and release the webcam.
"""

import sys
import os
import argparse
import urllib.request
import numpy as np
import cv2

# =====================================================================
# Landmark Indices (MediaPipe Face Mesh)
# =====================================================================

# Right Eye (Subject's right / viewer's left side):
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Left Eye (Subject's left / viewer's right side):
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Full eye contour loops for visualization
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Mouth Landmarks for MAR
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


# =====================================================================
# Model Management & Landmarker Initialization
# =====================================================================

def ensure_model_asset(model_path: str = "models/face_landmarker.task") -> str:
    """
    Ensures that the MediaPipe face_landmarker.task model file is available.
    Reuses existing file; downloads from Google repository if missing.
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


def initialize_landmarker(model_path: str = "models/face_landmarker.task"):
    """
    Initializes a single MediaPipe FaceLandmarker instance using Tasks Vision API.
    """
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    valid_path = ensure_model_asset(model_path)
    if not valid_path or not os.path.exists(valid_path):
        return None

    base_options = BaseOptions(model_asset_path=valid_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5
    )
    return vision.FaceLandmarker.create_from_options(options)


# =====================================================================
# Geometric Calculation Functions (EAR & MAR)
# =====================================================================

def calculate_ear(eye_points: np.ndarray) -> float:
    """
    Calculates Eye Aspect Ratio (EAR) based on Soukupova & Cech (2016):
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    if len(eye_points) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_points[:6]

    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal <= 1e-6:
        return 0.0

    return float((v1 + v2) / (2.0 * horizontal))


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
    v_center = np.linalg.norm(top_center - bot_center)
    v_left = np.linalg.norm(top_left - bot_left)
    v_right = np.linalg.norm(top_right - bot_right)
    horizontal = np.linalg.norm(corner_left - corner_right)

    if horizontal <= 1e-6:
        return 0.0

    return float((v_left + v_center + v_right) / (2.0 * horizontal))


# =====================================================================
# Core Decision / Evaluation Function
# =====================================================================

def evaluate_driver_state(
    ear_left: float,
    ear_right: float,
    mar: float,
    eye_closed_frames: int,
    mouth_open_frames: int,
    ear_threshold: float = 0.22,
    mar_threshold: float = 0.50,
    eye_consecutive_frames: int = 20,
    mouth_consecutive_frames: int = 25,
    face_detected: bool = True
) -> dict:
    """
    Single, documented decision/status function for driver state evaluation.
    This pure-function contract can be consumed directly by a FastAPI backend
    or WebSocket streamer without parsing UI strings.

    Returns:
        dict: Standardized driver state evaluation payload.
    """
    ear_avg = float((ear_left + ear_right) / 2.0) if face_detected else 0.0

    if not face_detected:
        return {
            "face_detected": False,
            "metrics": {
                "ear_left": 0.0,
                "ear_right": 0.0,
                "ear_avg": 0.0,
                "mar": 0.0
            },
            "counters": {
                "eye_closed_frames": 0,
                "mouth_open_frames": 0
            },
            "states": {
                "eye_state": "NO_FACE",
                "mouth_state": "NO_FACE",
                "primary_status": "No Face Detected"
            },
            "alerts": {
                "drowsiness_warning": False,
                "yawn_alert": False,
                "dual_alert": False
            }
        }

    # Evaluate eye closure state
    if eye_closed_frames >= eye_consecutive_frames:
        eye_state = "DROWSINESS_WARNING"
        drowsiness_warning = True
    elif eye_closed_frames > 0:
        eye_state = "EYES_CLOSED"
        drowsiness_warning = False
    else:
        eye_state = "EYES_OPEN"
        drowsiness_warning = False

    # Evaluate mouth yawning state
    if mouth_open_frames >= mouth_consecutive_frames:
        mouth_state = "YAWNING_ALERT"
        yawn_alert = True
    elif mouth_open_frames > 0:
        mouth_state = "POSSIBLE_YAWN"
        yawn_alert = False
    else:
        mouth_state = "MOUTH_NORMAL"
        yawn_alert = False

    dual_alert = drowsiness_warning and yawn_alert

    # Determine primary readable status without overwriting concurrent alerts
    if dual_alert:
        primary_status = "CRITICAL: DROWSINESS & YAWNING!"
    elif drowsiness_warning:
        primary_status = "DROWSINESS WARNING!"
    elif yawn_alert:
        primary_status = "YAWNING ALERT!"
    elif eye_state == "EYES_CLOSED" and mouth_state == "POSSIBLE_YAWN":
        primary_status = f"Closed ({eye_closed_frames}) & Yawn ({mouth_open_frames})"
    elif eye_state == "EYES_CLOSED":
        primary_status = f"Eyes Closed ({eye_closed_frames}/{eye_consecutive_frames})"
    elif mouth_state == "POSSIBLE_YAWN":
        primary_status = f"Possible Yawn ({mouth_open_frames}/{mouth_consecutive_frames})"
    else:
        primary_status = "NORMAL (Driver Alert)"

    return {
        "face_detected": True,
        "metrics": {
            "ear_left": round(float(ear_left), 3),
            "ear_right": round(float(ear_right), 3),
            "ear_avg": round(float(ear_avg), 3),
            "mar": round(float(mar), 3)
        },
        "counters": {
            "eye_closed_frames": eye_closed_frames,
            "mouth_open_frames": mouth_open_frames
        },
        "states": {
            "eye_state": eye_state,
            "mouth_state": mouth_state,
            "primary_status": primary_status
        },
        "alerts": {
            "drowsiness_warning": drowsiness_warning,
            "yawn_alert": yawn_alert,
            "dual_alert": dual_alert
        }
    }


# =====================================================================
# Visualization / Drawing Helpers
# =====================================================================

def draw_driver_mesh(frame: np.ndarray, landmarks, w: int, h: int, eye_color: tuple, mouth_color: tuple):
    """
    Renders eye contours and mouth connections in one drawing pass.
    """
    from mediapipe.tasks.python.vision import FaceLandmarksConnections

    def to_px(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h], dtype=np.int32)

    # 1. Draw Eye Contours
    r_pts = np.array([to_px(i) for i in RIGHT_EYE_CONTOUR]).reshape((-1, 1, 2))
    l_pts = np.array([to_px(i) for i in LEFT_EYE_CONTOUR]).reshape((-1, 1, 2))
    cv2.polylines(frame, [r_pts], isClosed=True, color=eye_color, thickness=1, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [l_pts], isClosed=True, color=eye_color, thickness=1, lineType=cv2.LINE_AA)

    # 2. Draw Mouth Connections
    for conn in FaceLandmarksConnections.FACE_LANDMARKS_LIPS:
        pt1 = tuple(to_px(conn.start))
        pt2 = tuple(to_px(conn.end))
        cv2.line(frame, pt1, pt2, mouth_color, 1, cv2.LINE_AA)


# =====================================================================
# Main Engine Video Loop
# =====================================================================

def run_driver_monitor(
    camera_index: int = 0,
    ear_threshold: float = 0.22,
    mar_threshold: float = 0.50,
    eye_consecutive_frames: int = 20,
    mouth_consecutive_frames: int = 25,
    max_frames: int = None,
    headless: bool = False,
    model_path: str = "models/face_landmarker.task"
):
    """
    Runs combined webcam ingestion loop, MediaPipe landmark extraction,
    and concurrent drowsiness/yawning detection.
    """
    import mediapipe as mp

    print("=" * 72)
    print(" Driver Fatigue & Distraction Safety Monitor - Combined Driver Engine ")
    print("=" * 72)
    print(f"Camera Index:                {camera_index}")
    print(f"EAR Threshold:               {ear_threshold:.2f} (Consecutive Frames: {eye_consecutive_frames})")
    print(f"MAR Threshold:               {mar_threshold:.2f} (Consecutive Frames: {mouth_consecutive_frames})")
    print(f"Model Path:                  {model_path}")
    print("=" * 72)

    # 1. Initialize Landmarker
    landmarker = initialize_landmarker(model_path)
    if landmarker is None:
        print("[ERROR] Failed to initialize MediaPipe FaceLandmarker.")
        return 1

    # 2. Open Camera
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"\n[ERROR] Failed to open camera at index {camera_index}.")
        print("Please check camera connection and privacy permissions.")
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Camera connected ({width}x{height})")
    print("[*] Engine running. Press 'q' in video window to exit.\n")

    frame_count = 0
    eye_closed_frames = 0
    mouth_open_frames = 0
    window_name = "Driver Safety Monitor - Combined Driver Engine (Prototype 4)"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Could not read frame from camera. Exiting.")
                break

            frame_count += 1
            h, w = frame.shape[:2]

            # Ingest frame into MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect(mp_image)

            face_detected = len(detection_result.face_landmarks) > 0
            ear_left = 0.0
            ear_right = 0.0
            mar_val = 0.0

            if face_detected:
                landmarks = detection_result.face_landmarks[0]

                def to_pixel(idx):
                    lm = landmarks[idx]
                    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

                # Eye landmarks
                r_pts = np.array([to_pixel(i) for i in RIGHT_EYE_INDICES])
                l_pts = np.array([to_pixel(i) for i in LEFT_EYE_INDICES])
                ear_right = calculate_ear(r_pts)
                ear_left = calculate_ear(l_pts)
                ear_avg = (ear_right + ear_left) / 2.0

                # Mouth landmarks
                p_c_left = to_pixel(MOUTH_CORNER_LEFT)
                p_c_right = to_pixel(MOUTH_CORNER_RIGHT)
                p_top_c = to_pixel(UPPER_LIP_CENTER)
                p_bot_c = to_pixel(LOWER_LIP_CENTER)
                p_top_l = to_pixel(UPPER_LIP_LEFT)
                p_bot_l = to_pixel(LOWER_LIP_LEFT)
                p_top_r = to_pixel(UPPER_LIP_RIGHT)
                p_bot_r = to_pixel(LOWER_LIP_RIGHT)
                mar_val = calculate_mar(
                    p_c_left, p_c_right,
                    p_top_c, p_bot_c,
                    p_top_l, p_bot_l,
                    p_top_r, p_bot_r
                )

                # Update frame counters
                if ear_avg < ear_threshold:
                    eye_closed_frames += 1
                else:
                    eye_closed_frames = max(0, eye_closed_frames - 1)

                if mar_val >= mar_threshold:
                    mouth_open_frames += 1
                else:
                    mouth_open_frames = max(0, mouth_open_frames - 1)

            else:
                eye_closed_frames = 0
                mouth_open_frames = 0

            # 3. Evaluate unified driver state via documented decision function
            state = evaluate_driver_state(
                ear_left=ear_left,
                ear_right=ear_right,
                mar=mar_val,
                eye_closed_frames=eye_closed_frames,
                mouth_open_frames=mouth_open_frames,
                ear_threshold=ear_threshold,
                mar_threshold=mar_threshold,
                eye_consecutive_frames=eye_consecutive_frames,
                mouth_consecutive_frames=mouth_consecutive_frames,
                face_detected=face_detected
            )

            # Determine contour colors based on states
            if state["alerts"]["drowsiness_warning"]:
                eye_color = (0, 0, 255)      # Red
            elif state["states"]["eye_state"] == "EYES_CLOSED":
                eye_color = (0, 165, 255)    # Amber
            else:
                eye_color = (0, 220, 0)      # Green

            if state["alerts"]["yawn_alert"]:
                mouth_color = (0, 0, 255)    # Red
            elif state["states"]["mouth_state"] == "POSSIBLE_YAWN":
                mouth_color = (0, 165, 255)  # Amber
            else:
                mouth_color = (0, 220, 0)    # Green

            # Draw visual landmarks if face present
            if face_detected:
                draw_driver_mesh(frame, landmarks, w, h, eye_color, mouth_color)

            # 4. Render Telemetry HUD Card
            cv2.rectangle(frame, (10, 10), (490, 140), (15, 15, 15), -1)
            cv2.rectangle(frame, (10, 10), (490, 140), (55, 55, 55), 1)

            # Primary status line
            overall_color = (0, 220, 0)
            if state["alerts"]["dual_alert"]:
                overall_color = (0, 0, 255)
            elif state["alerts"]["drowsiness_warning"] or state["alerts"]["yawn_alert"]:
                overall_color = (0, 0, 255)
            elif state["states"]["eye_state"] == "EYES_CLOSED" or state["states"]["mouth_state"] == "POSSIBLE_YAWN":
                overall_color = (0, 165, 255)

            cv2.putText(
                frame,
                f"Status: {state['states']['primary_status']}",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                overall_color,
                2
            )

            # Line 2: Eye Telemetry
            eye_info = (
                f"Eyes: {state['states']['eye_state']} | "
                f"EAR: {state['metrics']['ear_avg']:.3f} (L:{state['metrics']['ear_left']:.2f}, R:{state['metrics']['ear_right']:.2f}) | "
                f"Closed: {state['counters']['eye_closed_frames']}/{eye_consecutive_frames}"
            )
            cv2.putText(frame, eye_info, (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)

            # Line 3: Mouth Telemetry
            mouth_info = (
                f"Mouth: {state['states']['mouth_state']} | "
                f"MAR: {state['metrics']['mar']:.3f} (Thresh: {mar_threshold:.2f}) | "
                f"Yawn: {state['counters']['mouth_open_frames']}/{mouth_consecutive_frames}"
            )
            cv2.putText(frame, mouth_info, (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)

            # Line 4: Concurrent Alert Badges
            drowsy_badge = "[DROWSY: ON]" if state["alerts"]["drowsiness_warning"] else "[DROWSY: OFF]"
            yawn_badge = "[YAWN: ON]" if state["alerts"]["yawn_alert"] else "[YAWN: OFF]"
            cv2.putText(
                frame,
                f"Alert Flags: {drowsy_badge}  {yawn_badge}",
                (20, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 0, 255) if (state["alerts"]["drowsiness_warning"] or state["alerts"]["yawn_alert"]) else (0, 200, 0),
                1
            )

            # Line 5: Coursework Disclaimer
            cv2.putText(
                frame,
                "Prototype 4: Combined Drowsiness & Yawn Monitor (Coursework Prototype)",
                (20, 131),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (120, 120, 120),
                1
            )

            # Flashing border warning when any alert is active
            if (state["alerts"]["drowsiness_warning"] or state["alerts"]["yawn_alert"]) and (frame_count % 6 < 3):
                border_color = (0, 0, 255)
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 6)

            # Footer
            cv2.putText(frame, "Press 'q' to exit", (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # GUI output
            if not headless:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[*] User pressed 'q'. Exiting driver monitor.")
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


# =====================================================================
# Synthetic Tests (EAR, MAR, Decision Engine)
# =====================================================================

def run_synthetic_tests() -> bool:
    """
    Executes automated synthetic unit tests verifying EAR math, MAR math,
    and the driver state decision logic without requiring a camera or person.
    """
    print("\n" + "=" * 60)
    print(" Running Synthetic Validation Tests for Driver Monitor ")
    print("=" * 60)

    # 1. EAR Test
    open_eye = np.array([
        [10.0, 30.0], [15.0, 24.0], [25.0, 24.0],
        [30.0, 30.0], [25.0, 36.0], [15.0, 36.0]
    ])
    closed_eye = np.array([
        [10.0, 30.0], [15.0, 29.5], [25.0, 29.5],
        [30.0, 30.0], [25.0, 30.5], [15.0, 30.5]
    ])
    ear_open = calculate_ear(open_eye)
    ear_closed = calculate_ear(closed_eye)
    assert ear_open > 0.25, f"Expected open eye EAR > 0.25, got {ear_open}"
    assert ear_closed < 0.10, f"Expected closed eye EAR < 0.10, got {ear_closed}"
    print(f"[PASS] calculate_ear(): open={ear_open:.3f}, closed={ear_closed:.3f}")

    # 2. MAR Test
    corner_l = np.array([20.0, 50.0])
    corner_r = np.array([80.0, 50.0])
    top_c_norm, bot_c_norm = np.array([50.0, 48.5]), np.array([50.0, 51.5])
    top_l_norm, bot_l_norm = np.array([35.0, 49.0]), np.array([35.0, 51.0])
    top_r_norm, bot_r_norm = np.array([65.0, 49.0]), np.array([65.0, 51.0])
    mar_normal = calculate_mar(corner_l, corner_r, top_c_norm, bot_c_norm, top_l_norm, bot_l_norm, top_r_norm, bot_r_norm)

    top_c_yawn, bot_c_yawn = np.array([50.0, 25.0]), np.array([50.0, 75.0])
    top_l_yawn, bot_l_yawn = np.array([35.0, 30.0]), np.array([35.0, 70.0])
    top_r_yawn, bot_r_yawn = np.array([65.0, 30.0]), np.array([65.0, 70.0])
    mar_yawn = calculate_mar(corner_l, corner_r, top_c_yawn, bot_c_yawn, top_l_yawn, bot_l_yawn, top_r_yawn, bot_r_yawn)

    assert mar_normal < 0.20, f"Expected normal MAR < 0.20, got {mar_normal}"
    assert mar_yawn > 0.60, f"Expected yawn MAR > 0.60, got {mar_yawn}"
    print(f"[PASS] calculate_mar(): normal={mar_normal:.3f}, yawning={mar_yawn:.3f}")

    # 3. Decision Engine: Normal State
    s_norm = evaluate_driver_state(0.30, 0.30, 0.08, 0, 0, 0.22, 0.50, 20, 25)
    assert s_norm["states"]["eye_state"] == "EYES_OPEN"
    assert s_norm["states"]["mouth_state"] == "MOUTH_NORMAL"
    assert s_norm["alerts"]["drowsiness_warning"] is False
    assert s_norm["alerts"]["yawn_alert"] is False
    print("[PASS] evaluate_driver_state(): Normal State Verified")

    # 4. Decision Engine: Drowsiness Warning Only
    s_drowsy = evaluate_driver_state(0.15, 0.14, 0.08, 20, 0, 0.22, 0.50, 20, 25)
    assert s_drowsy["states"]["eye_state"] == "DROWSINESS_WARNING"
    assert s_drowsy["alerts"]["drowsiness_warning"] is True
    assert s_drowsy["alerts"]["yawn_alert"] is False
    assert "DROWSINESS WARNING" in s_drowsy["states"]["primary_status"]
    print("[PASS] evaluate_driver_state(): Drowsiness Warning Verified")

    # 5. Decision Engine: Yawn Alert Only
    s_yawn = evaluate_driver_state(0.30, 0.31, 0.65, 0, 25, 0.22, 0.50, 20, 25)
    assert s_yawn["states"]["mouth_state"] == "YAWNING_ALERT"
    assert s_yawn["alerts"]["yawn_alert"] is True
    assert s_yawn["alerts"]["drowsiness_warning"] is False
    assert "YAWNING ALERT" in s_yawn["states"]["primary_status"]
    print("[PASS] evaluate_driver_state(): Yawning Alert Verified")

    # 6. Decision Engine: Dual Concurrent Alert
    s_dual = evaluate_driver_state(0.15, 0.14, 0.65, 20, 25, 0.22, 0.50, 20, 25)
    assert s_dual["alerts"]["dual_alert"] is True
    assert s_dual["alerts"]["drowsiness_warning"] is True
    assert s_dual["alerts"]["yawn_alert"] is True
    assert "CRITICAL" in s_dual["states"]["primary_status"]
    print("[PASS] evaluate_driver_state(): Dual Concurrent Alert Verified")

    print("=" * 60)
    print(" All Synthetic Unit Tests PASSED Successfully! ")
    print("=" * 60 + "\n")
    return True


# =====================================================================
# Main CLI Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Driver Fatigue & Distraction Safety Monitor - Combined Driver Engine")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--ear-thresh", type=float, default=0.22, help="EAR threshold for eye closure (default: 0.22)")
    parser.add_argument("--mar-thresh", type=float, default=0.50, help="MAR threshold for yawning (default: 0.50)")
    parser.add_argument("--eye-frames", type=int, default=20, help="Consecutive closed-eye frames before warning (default: 20)")
    parser.add_argument("--yawn-frames", type=int, default=25, help="Consecutive open-mouth frames before alert (default: 25)")
    parser.add_argument("--frames", type=int, default=None, help="Exit after N frames (for automated verification)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI display window")
    parser.add_argument("--model-path", type=str, default="models/face_landmarker.task", help="Path to MediaPipe model asset")
    parser.add_argument("--run-tests", action="store_true", help="Run automated synthetic validation tests and exit")
    args = parser.parse_args()

    if args.run_tests:
        success = run_synthetic_tests()
        sys.exit(0 if success else 1)

    exit_code = run_driver_monitor(
        camera_index=args.camera,
        ear_threshold=args.ear_thresh,
        mar_threshold=args.mar_thresh,
        eye_consecutive_frames=args.eye_frames,
        mouth_consecutive_frames=args.yawn_frames,
        max_frames=args.frames,
        headless=args.headless,
        model_path=args.model_path
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
