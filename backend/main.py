"""
FastAPI application for Driver Fatigue & Distraction Safety Monitor.
Includes headless background worker for live video processing and telemetry endpoints.
"""

from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import os
import sys
import time
import threading
import pathlib
import numpy as np
import cv2

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

# Ensure project root is in sys.path to access the vision module
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.driver_monitor import (
    initialize_landmarker,
    calculate_ear,
    calculate_mar,
    evaluate_driver_state,
    RIGHT_EYE_INDICES,
    LEFT_EYE_INDICES,
    MOUTH_CORNER_LEFT,
    MOUTH_CORNER_RIGHT,
    UPPER_LIP_CENTER,
    LOWER_LIP_CENTER,
    UPPER_LIP_LEFT,
    LOWER_LIP_LEFT,
    UPPER_LIP_RIGHT,
    LOWER_LIP_RIGHT,
)


# =====================================================================
# Request & Response Schemas
# =====================================================================

class StartMonitorRequest(BaseModel):
    camera_index: int = Field(0, description="Webcam device index (default: 0)")
    ear_threshold: float = Field(0.22, ge=0.05, le=0.50, description="EAR threshold for eye closure")
    mar_threshold: float = Field(0.50, ge=0.10, le=1.50, description="MAR threshold for yawning")
    eye_consecutive_frames: int = Field(20, ge=1, le=300, description="Consecutive closed frames for drowsiness warning")
    mouth_consecutive_frames: int = Field(25, ge=1, le=300, description="Consecutive open frames for yawning alert")


# =====================================================================
# Background Driver Monitor Worker Manager
# =====================================================================

class DriverMonitorManager:
    """
    Manages the lifecycle of a single headless camera worker thread,
    ensuring thread-safe state access and exclusive hardware ownership.
    """

    def __init__(self, model_path: str = "models/face_landmarker.task"):
        self.model_path = model_path
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False
        self._starting = False
        self._latest_frame: Optional[np.ndarray] = None

        # Current session configuration
        self._config: Dict[str, Any] = {
            "camera_index": 0,
            "ear_threshold": 0.22,
            "mar_threshold": 0.50,
            "eye_consecutive_frames": 20,
            "mouth_consecutive_frames": 25,
        }

        # Live telemetry statistics
        self._stats: Dict[str, Any] = {
            "frames_processed": 0,
            "fps": 0.0,
            "error": None,
            "started_at": None,
        }

        # Initial baseline evaluation payload
        self._latest_evaluation = evaluate_driver_state(
            ear_left=0.0,
            ear_right=0.0,
            mar=0.0,
            eye_closed_frames=0,
            mouth_open_frames=0,
            face_detected=False
        )

    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    def start(self, req: StartMonitorRequest) -> tuple[bool, str]:
        """
        Starts the background camera processing thread.
        Guarantees single-worker exclusivity and validates camera initialization.
        """
        with self._lock:
            if self._is_running or self._starting or (self._worker_thread and self._worker_thread.is_alive()):
                return False, "Driver monitoring worker is already active."
            abs_model_path = os.path.join(str(PROJECT_ROOT), self.model_path)
            if not os.path.isfile(abs_model_path):
                self._stats["error"] = f"Model file not found at {abs_model_path}."
                return False, self._stats["error"]
            self._stop_event.clear()
            self._latest_frame = None
            self._starting = True
            self._config = {
                "camera_index": req.camera_index,
                "ear_threshold": req.ear_threshold,
                "mar_threshold": req.mar_threshold,
                "eye_consecutive_frames": req.eye_consecutive_frames,
                "mouth_consecutive_frames": req.mouth_consecutive_frames,
            }
            self._stats = {
                "frames_processed": 0,
                "fps": 0.0,
                "error": None,
                "started_at": None,
            }
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                args=(abs_model_path,),
                daemon=True,
                name="DriverMonitorWorker"
            )
            try:
                self._worker_thread.start()
            except Exception as exc:
                self._starting = False
                self._worker_thread = None
                self._stats["error"] = f"Could not start monitoring worker: {exc}"
                return False, self._stats["error"]

        # Wait briefly for initialization so failures are returned by this request.
        ready_deadline = time.monotonic() + 5.0
        while time.monotonic() < ready_deadline:
            with self._lock:
                if not self._starting:
                    if self._is_running:
                        return True, "Driver monitoring started successfully in background worker."
                    worker = self._worker_thread
                    error = self._stats.get("error") or "Monitoring initialization failed."
                    break
            time.sleep(0.01)
        else:
            worker = None
            error = None

        if error is not None:
            if worker and worker.is_alive():
                worker.join()
            with self._lock:
                self._worker_thread = None
            return False, error

        with self._lock:
            self._stop_event.set()
            self._stats["error"] = "Timed out initializing camera monitoring worker."
        if self._worker_thread:
            self._worker_thread.join()
        with self._lock:
            self._worker_thread = None
            self._starting = False
            self._is_running = False
        return False, "Timed out initializing camera monitoring worker."

    def stop(self) -> tuple[bool, str]:
        """
        Signals the worker thread to stop and waits for clean hardware release.
        """
        with self._lock:
            if not self._is_running and not self._starting:
                return False, "Driver monitoring is not currently running."
            self._stop_event.set()
            worker = self._worker_thread

        if worker and worker.is_alive():
            worker.join()

        with self._lock:
            self._is_running = False
            self._starting = False
            self._worker_thread = None
            self._latest_frame = None

        return True, "Driver monitoring stopped successfully."

    def get_status_payload(self) -> Dict[str, Any]:
        """
        Returns a thread-safe snapshot of monitoring status and telemetry.
        """
        with self._lock:
            return {
                "monitor_running": self._is_running,
                "monitor_starting": self._starting,
                "message": (
                    "Monitoring active in background."
                    if self._is_running
                    else "Monitoring is idle. Call POST /api/monitor/start to begin."
                ),
                "config": dict(self._config),
                "stats": dict(self._stats),
                "evaluation": dict(self._latest_evaluation)
            }

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Returns an independent copy of the latest annotated frame, if available."""
        with self._lock:
            if not self._is_running or self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def _worker_loop(self, abs_model_path: str):
        """
        Background worker processing camera frames headlessly without GUI calls.
        """
        cap = None
        landmarker = None
        try:
            import mediapipe as mp
            # 1. Initialize Landmarker
            landmarker = initialize_landmarker(abs_model_path)
            if landmarker is None:
                with self._lock:
                    self._stats["error"] = "Failed to initialize MediaPipe FaceLandmarker."
                return

            # 2. Open Camera
            cam_idx = self._config["camera_index"]
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(cam_idx)

            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(cam_idx)

            if not cap.isOpened():
                with self._lock:
                    self._stats["error"] = f"Failed to open camera index {cam_idx} in worker thread."
                return

            with self._lock:
                self._starting = False
                self._is_running = True
                self._stats["started_at"] = time.time()

            eye_closed_frames = 0
            mouth_open_frames = 0
            frames_counted = 0
            start_time = time.time()

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                frames_counted += 1
                h, w = frame.shape[:2]

                # Convert to RGB and process with MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)

                face_detected = len(result.face_landmarks) > 0
                ear_left, ear_right, mar_val = 0.0, 0.0, 0.0

                if face_detected:
                    lm = result.face_landmarks[0]

                    def to_px(idx):
                        pt = lm[idx]
                        return np.array([pt.x * w, pt.y * h], dtype=np.float32)

                    # Compute EAR
                    r_pts = np.array([to_px(i) for i in RIGHT_EYE_INDICES])
                    l_pts = np.array([to_px(i) for i in LEFT_EYE_INDICES])
                    ear_right = calculate_ear(r_pts)
                    ear_left = calculate_ear(l_pts)
                    ear_avg = (ear_right + ear_left) / 2.0

                    # Compute MAR
                    p_c_l = to_px(MOUTH_CORNER_LEFT)
                    p_c_r = to_px(MOUTH_CORNER_RIGHT)
                    p_top_c = to_px(UPPER_LIP_CENTER)
                    p_bot_c = to_px(LOWER_LIP_CENTER)
                    p_top_l = to_px(UPPER_LIP_LEFT)
                    p_bot_l = to_px(LOWER_LIP_LEFT)
                    p_top_r = to_px(UPPER_LIP_RIGHT)
                    p_bot_r = to_px(LOWER_LIP_RIGHT)
                    mar_val = calculate_mar(
                        p_c_l, p_c_r, p_top_c, p_bot_c, p_top_l, p_bot_l, p_top_r, p_bot_r
                    )

                    # Draw a minimal face mesh overlay on the same frame being monitored.
                    from mediapipe.tasks.python.vision import FaceLandmarksConnections
                    for connection in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
                        start_pt = lm[connection.start]
                        end_pt = lm[connection.end]
                        cv2.line(
                            frame,
                            (int(start_pt.x * w), int(start_pt.y * h)),
                            (int(end_pt.x * w), int(end_pt.y * h)),
                            (70, 220, 150),
                            1,
                            cv2.LINE_AA,
                        )

                    # Update counters
                    if ear_avg < self._config["ear_threshold"]:
                        eye_closed_frames += 1
                    else:
                        eye_closed_frames = max(0, eye_closed_frames - 1)

                    if mar_val >= self._config["mar_threshold"]:
                        mouth_open_frames += 1
                    else:
                        mouth_open_frames = max(0, mouth_open_frames - 1)
                else:
                    eye_closed_frames = 0
                    mouth_open_frames = 0

                # Evaluate state using shared logic from vision/driver_monitor.py
                evaluation = evaluate_driver_state(
                    ear_left=ear_left,
                    ear_right=ear_right,
                    mar=mar_val,
                    eye_closed_frames=eye_closed_frames,
                    mouth_open_frames=mouth_open_frames,
                    ear_threshold=self._config["ear_threshold"],
                    mar_threshold=self._config["mar_threshold"],
                    eye_consecutive_frames=self._config["eye_consecutive_frames"],
                    mouth_consecutive_frames=self._config["mouth_consecutive_frames"],
                    face_detected=face_detected
                )

                # Compute approximate FPS
                elapsed = time.time() - start_time
                current_fps = round(frames_counted / elapsed, 1) if elapsed > 0 else 0.0

                # Thread-safe telemetry update
                with self._lock:
                    self._latest_evaluation = evaluation
                    self._stats["frames_processed"] = frames_counted
                    self._stats["fps"] = current_fps
                    self._latest_frame = frame.copy()

        except Exception as exc:
            with self._lock:
                self._stats["error"] = f"Worker runtime exception: {str(exc)}"
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception as exc:
                    with self._lock:
                        self._stats["error"] = self._stats["error"] or f"Failed to release camera: {exc}"
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception as exc:
                    with self._lock:
                        self._stats["error"] = self._stats["error"] or f"Failed to close FaceLandmarker: {exc}"
            with self._lock:
                self._starting = False
                self._is_running = False
                self._latest_frame = None


# Instantiate central monitor manager singleton
monitor_manager = DriverMonitorManager()


# =====================================================================
# FastAPI Application & Lifecycle
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Clean shutdown of background worker if active
    monitor_manager.stop()


app = FastAPI(
    title="Driver Fatigue & Distraction Safety Monitor",
    description="REST API for real-time driver drowsiness and yawning telemetry.",
    version="0.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Driver Fatigue & Distraction Safety Monitor API",
        "version": "0.2.0",
        "ready": True,
        "note": "Vision monitoring engine integrated. Call /api/monitor/status to inspect telemetry."
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/api/monitor/status")
def get_monitor_status():
    """
    Returns the current monitoring status, active configuration,
    and latest driver telemetry (EAR, MAR, frame counters, and alert states).
    """
    return monitor_manager.get_status_payload()


def _mjpeg_frames():
    while monitor_manager.is_running():
        frame = monitor_manager.get_latest_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            time.sleep(0.05)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"


@app.get("/api/monitor/stream")
def stream_monitor():
    if not monitor_manager.is_running():
        raise HTTPException(status_code=409, detail="Monitoring is stopped; start monitoring before requesting the camera stream.")
    if monitor_manager.get_latest_frame() is None:
        raise HTTPException(status_code=503, detail="Monitoring is active, but no camera frame is available yet.")
    return StreamingResponse(
        _mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
    )


@app.post("/api/monitor/start")
def start_monitor(req: StartMonitorRequest = StartMonitorRequest()):
    """
    Starts the headless background video processing worker.
    Rejects duplicate start requests and verifies camera access safely.
    """
    success, msg = monitor_manager.start(req)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg
        )
    return {
        "status": "started",
        "message": msg,
        "config": req.dict()
    }


@app.post("/api/monitor/stop")
def stop_monitor():
    """
    Stops the active background video processing worker and releases camera hardware.
    """
    success, msg = monitor_manager.stop()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg
        )
    return {
        "status": "stopped",
        "message": msg
    }
