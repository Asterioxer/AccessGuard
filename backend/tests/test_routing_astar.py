import pytest
from app.gis.routing_engine import AccessibilityRoutingEngine
from app.gis.osm_importer import create_pedestrian_segment_models
from app.db.models import Observation, BarrierDetection

def test_astar_pathfinding():
    segments = create_pedestrian_segment_models()
    engine = AccessibilityRoutingEngine(segments)
    
    # Test A* search between start and destination
    res_astar = engine.find_accessible_route(
        start_lat=37.7765,
        start_lon=-122.4194,
        end_lat=37.7778,
        end_lon=-122.4194,
        algorithm="astar"
    )
    assert res_astar.success is True
    assert res_astar.algorithm_used == "astar"
    assert res_astar.total_distance_meters > 0.0

def test_alternative_route_direct_comparison():
    segments = create_pedestrian_segment_models()
    # Add stairs on the direct path (seg_central_plaza_stairs)
    direct_seg = next(s for s in segments if s.id == "seg_central_plaza_stairs")
    obs = Observation(id="obs_test_stairs", segment_id=direct_seg.id, status="verified", latitude=37.777, longitude=-122.4194)
    obs.detections = [BarrierDetection(barrier_type="stairs", confidence=0.95, uncertainty_score=0.1)]
    direct_seg.observations = [obs]

    engine = AccessibilityRoutingEngine(segments)
    res = engine.find_accessible_route(
        start_lat=37.7765,
        start_lon=-122.4194,
        end_lat=37.7778,
        end_lon=-122.4194,
        algorithm="dijkstra",
        include_alternatives=True
    )

    assert res.success is True
    assert res.direct_route_comparison is not None
    assert "direct_distance_meters" in res.direct_route_comparison
    assert "accessible_detour_meters" in res.direct_route_comparison
    assert res.direct_route_comparison["direct_route_has_stairs"] is True
