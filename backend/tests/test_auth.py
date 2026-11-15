import pytest

def test_user_registration(client):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "newuser@test.org", "password": "password123", "role": "contributor"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "newuser@test.org"
    assert data["role"] == "contributor"
    assert "id" in data

def test_duplicate_user_registration(client, contributor_user):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": contributor_user.email, "password": "password123"}
    )
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]

def test_user_login(client, contributor_user):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": contributor_user.email, "password": "testpass123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["email"] == contributor_user.email

def test_invalid_login(client, contributor_user):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": contributor_user.email, "password": "wrongpassword"}
    )
    assert response.status_code == 401

def test_get_current_user_profile(client, contributor_token):
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {contributor_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "contributor@test.org"
