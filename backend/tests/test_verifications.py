import pytest
import cv2
import numpy as np

def create_image():
    img = np.full((200, 200, 3), 150, dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", img)
    return enc.tobytes()

def test_human_verification_submission(client, contributor_token, verifier_token):
    # Upload an observation
    img_bytes = create_image()
    up_resp = client.post(
        "/api/v1/observations/upload",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"latitude": "37.7765", "longitude": "-122.4194"},
        headers={"Authorization": f"Bearer {contributor_token}"}
    )
    obs_id = up_resp.json()["id"]

    # Submit verification review
    review_resp = client.post(
        "/api/v1/verifications/submit",
        json={
            "observation_id": obs_id,
            "vote": "confirm",
            "notes": "Verified on-site: curb ramp is present and conforms to ADA slope standards."
        },
        headers={"Authorization": f"Bearer {verifier_token}"}
    )
    assert review_resp.status_code == 200
    data = review_resp.json()
    assert data["observation_id"] == obs_id
    assert data["verification_score"] == 1

def test_duplicate_review_prevented(client, verifier_token, contributor_token):
    img_bytes = create_image()
    up_resp = client.post(
        "/api/v1/observations/upload",
        files={"file": ("test2.jpg", img_bytes, "image/jpeg")},
        headers={"Authorization": f"Bearer {contributor_token}"}
    )
    obs_id = up_resp.json()["id"]

    # First vote
    client.post(
        "/api/v1/verifications/submit",
        json={"observation_id": obs_id, "vote": "confirm"},
        headers={"Authorization": f"Bearer {verifier_token}"}
    )
    
    # Second duplicate vote by same user
    dup_resp = client.post(
        "/api/v1/verifications/submit",
        json={"observation_id": obs_id, "vote": "confirm"},
        headers={"Authorization": f"Bearer {verifier_token}"}
    )
    assert dup_resp.status_code == 400
    assert "already submitted" in dup_resp.json()["detail"].lower()

def test_verification_queue(client, contributor_token):
    # Queue should return unverified items
    img_bytes = create_image()
    client.post(
        "/api/v1/observations/upload",
        files={"file": ("queue_test.jpg", img_bytes, "image/jpeg")},
        headers={"Authorization": f"Bearer {contributor_token}"}
    )
    
    q_resp = client.get("/api/v1/verifications/queue")
    assert q_resp.status_code == 200
    items = q_resp.json()
    assert len(items) >= 1
    assert all(item["status"] == "tentative" for item in items)
