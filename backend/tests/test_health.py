import pytest

def test_root_endpoint(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["platform"] == "AccessGuard"
    assert "metrics_url" in data

def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["healthy", "ok"]

def test_system_info_endpoint(client):
    res = client.get("/system-info")
    assert res.status_code == 200
    data = res.json()
    assert data["project"] == "AccessGuard"
    assert "privacy_blurring_enabled" in data

def test_prometheus_metrics_endpoint(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "text/plain" in res.headers.get("content-type", "")
    assert len(res.text) > 0
