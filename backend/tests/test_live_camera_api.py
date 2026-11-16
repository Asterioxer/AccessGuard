"""
Tests for Live Camera and Video Stream API endpoints (/api/v1/cv/*).
"""

import base64
import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_list_adapters():
    res = client.get("/api/v1/cv/adapters")
    assert res.status_code == 200
    data = res.json()
    assert "available_adapters" in data
    assert "heuristic" in data["available_adapters"]
    assert "onnx" in data["available_adapters"]
    assert "active_adapter" in data

def test_set_active_adapter():
    # Set to onnx
    res = client.post("/api/v1/cv/active-adapter", json={"adapter_name": "onnx"})
    assert res.status_code == 200
    assert res.json()["active_adapter"] == "onnx"

    # Reset back to heuristic
    res = client.post("/api/v1/cv/active-adapter", json={"adapter_name": "heuristic"})
    assert res.status_code == 200
    assert res.json()["active_adapter"] == "heuristic"

    # Invalid adapter
    res_bad = client.post("/api/v1/cv/active-adapter", json={"adapter_name": "nonexistent_adapter"})
    assert res_bad.status_code == 400

def test_live_frame_base64_inference():
    # Create synthetic test image
    img = np.full((480, 640, 3), 120, dtype=np.uint8)
    # Draw horizontal stairs lines
    for y in range(200, 380, 20):
        cv2.line(img, (150, y), (450, y), (30, 30, 30), 2)

    _, buf = cv2.imencode(".jpg", img)
    b64_str = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    res = client.post(
        "/api/v1/cv/live-frame",
        data={
            "base64_image": b64_str,
            "stream_id": "test_stream_001",
            "timestamp": 123456.78
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert "detections" in data
    assert "latency_ms" in data
    assert data["latency_ms"] > 0
    assert "fps_estimate" in data
    assert "image_quality" in data
    assert data["privacy_redacted"] is True

def test_live_frame_multipart_upload():
    img = np.full((320, 320, 3), 150, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)

    res = client.post(
        "/api/v1/cv/live-frame",
        files={"file": ("frame.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
        data={"stream_id": "test_stream_002"}
    )
    assert res.status_code == 200
    data = res.json()
    assert "detections" in data
    assert "adapter_used" in data
