import pytest
from app.core.security import create_access_token, Role

def test_unauthenticated_admin_override_rejected(client):
    res = client.post("/api/v1/verifications/fake_obs_id/admin-override?new_status=verified")
    assert res.status_code == 401

def test_contributor_admin_override_forbidden(client):
    # Create token for standard contributor
    token = create_access_token({"sub": "user_contrib", "role": Role.CONTRIBUTOR})
    headers = {"Authorization": f"Bearer {token}"}
    res = client.post(
        "/api/v1/verifications/fake_obs_id/admin-override?new_status=verified",
        headers=headers
    )
    # Contributor should be denied with 403 Forbidden
    assert res.status_code == 403

def test_verifier_admin_override_forbidden(client):
    # Create token for verifier
    token = create_access_token({"sub": "user_verifier", "role": Role.VERIFIER})
    headers = {"Authorization": f"Bearer {token}"}
    res = client.post(
        "/api/v1/verifications/fake_obs_id/admin-override?new_status=verified",
        headers=headers
    )
    # Verifier should be denied with 403 Forbidden because admin is required
    assert res.status_code == 403

def test_admin_override_allowed_role(client, admin_token):
    # Calling with admin credentials succeeds the auth/role check (returns 404 for non-existent ID)
    headers = {"Authorization": f"Bearer {admin_token}"}
    res = client.post(
        "/api/v1/verifications/non_existent_id/admin-override?new_status=verified",
        headers=headers
    )
    assert res.status_code == 404
