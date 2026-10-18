"""
Automated unit tests for backend FastAPI monitoring integration.
Verifies status payload structure, idle state, invalid camera handling,
duplicate start protection, and clean stop behavior without requiring a physical camera.
"""

import sys
import pathlib
import pytest
import numpy as np
import threading

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import (
    app,
    read_root,
    health,
    get_monitor_status,
    start_monitor,
    stop_monitor,
    monitor_manager,
    StartMonitorRequest,
)
from fastapi import HTTPException
from unittest.mock import Mock
import backend.main as backend_main


def test_root_endpoint():
    """Verify GET / returns online status and service metadata."""
    res = read_root()
    assert res["status"] == "online"
    assert "version" in res
    assert res["ready"] is True


def test_health_endpoint():
    """Verify GET /health returns healthy status."""
    res = health()
    assert res["status"] == "healthy"


def test_initial_monitor_status_is_idle():
    """Verify GET /api/monitor/status returns monitor_running: False initially."""
    # Ensure manager is not running
    if monitor_manager.is_running():
        monitor_manager.stop()

    payload = get_monitor_status()
    assert payload["monitor_running"] is False
    assert "Monitoring is idle" in payload["message"]
    assert "config" in payload
    assert "stats" in payload
    assert "evaluation" in payload
    assert payload["evaluation"]["face_detected"] is False
    assert payload["evaluation"]["states"]["primary_status"] == "No Face Detected"


def test_stop_when_idle_raises_http_400():
    """Verify calling stop when monitoring is idle raises HTTP 400 gracefully."""
    if monitor_manager.is_running():
        monitor_manager.stop()

    with pytest.raises(HTTPException) as exc_info:
        stop_monitor()
    assert exc_info.value.status_code == 400
    assert "not currently running" in exc_info.value.detail


def test_start_invalid_camera_fails_gracefully():
    """Verify camera initialization failures are returned and never reported as running."""
    if monitor_manager.is_running():
        monitor_manager.stop()
    first_cap = Mock()
    first_cap.isOpened.return_value = False
    retry_cap = Mock()
    retry_cap.isOpened.return_value = False
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(backend_main.cv2, "VideoCapture", Mock(side_effect=[first_cap, retry_cap]))
    monkeypatch.setattr(backend_main, "initialize_landmarker", Mock(return_value=Mock(close=Mock())))
    bad_req = StartMonitorRequest(camera_index=999)
    with pytest.raises(HTTPException) as exc_info:
        start_monitor(bad_req)
    assert exc_info.value.status_code == 400
    assert "Failed to open camera" in exc_info.value.detail
    assert monitor_manager.is_running() is False
    first_cap.release.assert_called_once()
    retry_cap.release.assert_called_once()
    monkeypatch.undo()


def test_missing_model_fails_without_opening_camera(monkeypatch, tmp_path):
    manager = backend_main.DriverMonitorManager(str(tmp_path / "missing.task"))
    camera = Mock()
    monkeypatch.setattr(backend_main.cv2, "VideoCapture", camera)
    ok, message = manager.start(StartMonitorRequest())
    assert ok is False
    assert "Model file not found" in message
    camera.assert_not_called()
    assert manager.get_status_payload()["monitor_running"] is False


def test_start_and_stop_with_mocked_camera(monkeypatch, tmp_path):
    model = tmp_path / "model.task"
    model.touch()
    manager = backend_main.DriverMonitorManager(str(model))
    cap = Mock()
    cap.isOpened.return_value = True
    cap.read.side_effect = [(True, np.zeros((4, 4, 3), dtype=np.uint8))]
    landmarker = Mock()
    landmarker.detect.return_value = Mock(face_landmarks=[])
    monkeypatch.setattr(backend_main.cv2, "VideoCapture", Mock(return_value=cap))
    monkeypatch.setattr(backend_main, "initialize_landmarker", Mock(return_value=landmarker))

    ok, message = manager.start(StartMonitorRequest())
    assert ok is True, message
    assert manager.get_status_payload()["monitor_running"] is True
    stopped, _ = manager.stop()
    assert stopped is True
    assert manager.get_status_payload()["monitor_running"] is False
    cap.release.assert_called_once()
    landmarker.close.assert_called_once()


def test_concurrent_start_only_creates_one_worker(monkeypatch, tmp_path):
    model = tmp_path / "model.task"
    model.touch()
    manager = backend_main.DriverMonitorManager(str(model))
    entered = threading.Event()
    release = threading.Event()
    cap = Mock()
    cap.isOpened.return_value = True
    cap.read.side_effect = lambda: (release.wait(2) or True, np.zeros((4, 4, 3), dtype=np.uint8))
    landmarker = Mock()
    landmarker.detect.return_value = Mock(face_landmarks=[])
    monkeypatch.setattr(backend_main.cv2, "VideoCapture", Mock(return_value=cap))

    def init(_path):
        entered.set()
        return landmarker

    monkeypatch.setattr(backend_main, "initialize_landmarker", init)
    results = []
    first = threading.Thread(target=lambda: results.append(manager.start(StartMonitorRequest())))
    first.start()
    assert entered.wait(2)
    second = manager.start(StartMonitorRequest())
    assert second[0] is False
    assert "already active" in second[1]
    release.set()
    first.join(3)
    if manager.is_running():
        manager.stop()
    assert len(results) == 1 and results[0][0] is True


def test_duplicate_start_protection():
    """Verify that if manager is running, a duplicate start is rejected."""
    # Directly test the manager's state machine lock
    with monitor_manager._lock:
        monitor_manager._is_running = True

    try:
        success, msg = monitor_manager.start(StartMonitorRequest())
        assert success is False
        assert "already active" in msg
    finally:
        with monitor_manager._lock:
            monitor_manager._is_running = False


if __name__ == "__main__":
    test_root_endpoint()
    test_health_endpoint()
    test_initial_monitor_status_is_idle()
    test_stop_when_idle_raises_http_400()
    test_start_invalid_camera_fails_gracefully()
    test_duplicate_start_protection()
    print("=" * 60)
    print(" All API Integration Unit Tests PASSED Successfully! ")
    print("=" * 60)
