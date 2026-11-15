import pytest

def test_accessible_route_calculation(client):
    # Route from start of Main St to Market St
    payload = {
        "start": {"latitude": 37.7749, "longitude": -122.4194},
        "destination": {"latitude": 37.7778, "longitude": -122.4194},
        "wheelchair_profile": "manual_wheelchair",
        "avoid_unverified_barriers": True
    }
    response = client.post("/api/v1/routing/accessible-route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["total_distance_meters"] > 0
    assert len(data["path_coordinates"]) >= 2
    assert "geojson" in data
    assert data["geojson"]["type"] == "FeatureCollection"
    assert len(data["segments_traversed"]) >= 1

def test_route_avoids_stairs_when_accessible_bypass_exists(client, db_session):
    from app.db.models import Observation, BarrierDetection, PedestrianSegment
    
    # Mark seg_central_plaza_stairs as having stairs
    stairs_obs = Observation(
        id="obs_stairs_demo",
        segment_id="seg_central_plaza_stairs",
        media_path="./dummy.jpg",
        status="verified",
        latitude=37.7771,
        longitude=-122.4194
    )
    db_session.add(stairs_obs)
    det = BarrierDetection(
        id="det_stairs_demo",
        observation_id=stairs_obs.id,
        barrier_type="stairs",
        confidence=0.95,
        uncertainty_score=0.10,
        is_confirmed=True,
        bbox_ymin=0.3, bbox_xmin=0.1, bbox_ymax=0.8, bbox_xmax=0.9
    )
    db_session.add(det)
    
    seg = db_session.query(PedestrianSegment).filter(PedestrianSegment.id == "seg_central_plaza_stairs").first()
    if seg:
        seg.accessibility_score = 30.0
        seg.score_grade = "F"
    db_session.commit()

    # Request route from Main St (37.7765, -122.4194) to Market St (37.7778, -122.4194)
    payload = {
        "start": {"latitude": 37.7765, "longitude": -122.4194},
        "destination": {"latitude": 37.7778, "longitude": -122.4194},
        "wheelchair_profile": "manual_wheelchair"
    }
    response = client.post("/api/v1/routing/accessible-route", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Router must take the ramp bypass segments rather than the stairs segment!
    assert "seg_central_plaza_stairs" not in data["segments_traversed"]
    assert "seg_plaza_bypass_ramp_east" in data["segments_traversed"] or "seg_plaza_bypass_ramp_north" in data["segments_traversed"]
    assert data["avoids_stairs"] is True
