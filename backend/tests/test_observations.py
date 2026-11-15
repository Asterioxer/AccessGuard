import io
import cv2
import numpy as np
import pytest

def create_test_image_bytes():
    img = np.full((300, 300, 3), 180, dtype=np.uint8)
    # Add horizontal step lines
    for y in [100, 140, 180, 220]:
        cv2.line(img, (20, y), (280, y), (30, 30, 30), 3)
    _, enc = cv2.imencode(".jpg", img)
    return enc.tobytes()

def test_upload_observation_image(client, contributor_token):
    img_bytes = create_test_image_bytes()
    response = client.post(
        "/api/v1/observations/upload",
        files={"file": ("street_scene.jpg", img_bytes, "image/jpeg")},
        data={"latitude": "37.7765", "longitude": "-122.4194", "adapter": "heuristic"},
        headers={"Authorization": f"Bearer {contributor_token}"}
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["latitude"] == 37.7765
    assert data["longitude"] == -122.4194
    assert data["status"] == "tentative"
    assert len(data["detections"]) >= 1

def test_upload_invalid_file_rejected(client):
    fake_exe_bytes = b"MZ\x90\x00\x03\x00\x00\x00evil_binary"
    response = client.post(
        "/api/v1/observations/upload",
        files={"file": ("malicious.exe", fake_exe_bytes, "application/octet-stream")},
        data={"latitude": "37.7765", "longitude": "-122.4194"}
    )
    assert response.status_code == 400
    assert "validation failed" in response.json()["detail"].lower()

def test_list_observations(client, contributor_token):
    # Upload one first
    img_bytes = create_test_image_bytes()
    client.post(
        "/api/v1/observations/upload",
        files={"file": ("sidewalk.jpg", img_bytes, "image/jpeg")},
        headers={"Authorization": f"Bearer {contributor_token}"}
    )

    response = client.get("/api/v1/observations")
    assert response.status_code == 200
    obs_list = response.json()
    assert len(obs_list) >= 1
    assert "detections" in obs_list[0]

def test_get_observation_detail(client, contributor_token):
    img_bytes = create_test_image_bytes()
    create_resp = client.post(
        "/api/v1/observations/upload",
        files={"file": ("curb.jpg", img_bytes, "image/jpeg")},
        headers={"Authorization": f"Bearer {contributor_token}"}
    )
    obs_id = create_resp.json()["id"]

    detail_resp = client.get(f"/api/v1/observations/{obs_id}")
    assert detail_resp.status_code == 200
    data = detail_resp.json()
    assert data["id"] == obs_id
    assert "media_url" in data
