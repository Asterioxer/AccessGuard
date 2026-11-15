import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_segments_pagination(client):
    # Fetch with limit=2, offset=0
    res1 = client.get("/api/v1/segments?limit=2&offset=0")
    assert res1.status_code == 200
    data1 = res1.json()
    assert len(data1) <= 2

    # Fetch next page with limit=2, offset=2
    res2 = client.get("/api/v1/segments?limit=2&offset=2")
    assert res2.status_code == 200
    data2 = res2.json()
    
    # Verify different records between pages if enough segments exist
    if len(data1) > 0 and len(data2) > 0:
        ids_page1 = {s["id"] for s in data1}
        ids_page2 = {s["id"] for s in data2}
        assert ids_page1.isdisjoint(ids_page2)

def test_observations_pagination(client):
    res = client.get("/api/v1/observations?limit=10&offset=0")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

def test_invalid_pagination_bounds(client):
    # Negative offset should fail validation
    res = client.get("/api/v1/segments?limit=10&offset=-1")
    assert res.status_code == 422

    # Exceeding max limit (200) should fail validation
    res_exceed = client.get("/api/v1/segments?limit=500&offset=0")
    assert res_exceed.status_code == 422
