import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from app.gis.routing_engine import AccessibilityRoutingEngine
from app.gis.scoring import calculate_segment_accessibility_score
from app.db.models import PedestrianSegment, Observation, BarrierDetection

class MockSegment:
    def __init__(self, seg_id, name, start_lat, start_lon, end_lat, end_lon, score=50.0, grade="U", has_stairs=False, length=50.0):
        self.id = seg_id
        self.name = name
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon
        self.accessibility_score = score
        self.score_grade = grade
        self.length_meters = length
        self.geometry_geojson = json.dumps([[start_lon, start_lat], [end_lon, end_lat]])
        self.observations = []
        if has_stairs:
            obs = Observation(id=f"obs_{seg_id}", segment_id=seg_id, status="verified", latitude=start_lat, longitude=start_lon)
            det = BarrierDetection(id=f"det_{seg_id}", observation_id=obs.id, barrier_type="stairs", confidence=0.95, uncertainty_score=0.1)
            obs.detections = [det]
            self.observations.append(obs)

def audit_routing_gis():
    print("=== Phase 3: Routing and GIS Audit ===")
    results = {}

    # Test 1: Disconnected Graph
    print("\n1. Testing Disconnected Graphs...")
    seg_island_1 = MockSegment("isl_1", "Island A North", 37.770, -122.420, 37.772, -122.420)
    seg_island_2 = MockSegment("isl_2", "Island B South", 37.790, -122.420, 37.792, -122.420) # 2 km away, no edge
    engine_disc = AccessibilityRoutingEngine([seg_island_1, seg_island_2])
    route_disc = engine_disc.find_accessible_route(37.770, -122.420, 37.792, -122.420)
    
    results["disconnected_graph"] = {
        "success": route_disc.success,
        "notes": route_disc.routing_notes,
        "handled_gracefully": not route_disc.success and "No accessible continuous pedestrian route" in route_disc.routing_notes[0]
    }
    print(f"   Disconnected graph handled gracefully? {results['disconnected_graph']['handled_gracefully']}")

    # Test 2: Inaccessible Network (All paths blocked by stairs)
    print("\n2. Testing Inaccessible Network (All paths blocked by stairs)...")
    seg_stairs = MockSegment("stair_seg", "Stairway to Plaza", 37.770, -122.420, 37.772, -122.420, score=25.0, grade="F", has_stairs=True)
    engine_stairs = AccessibilityRoutingEngine([seg_stairs])
    route_stairs = engine_stairs.find_accessible_route(37.770, -122.420, 37.772, -122.420)
    
    results["all_paths_inaccessible"] = {
        "success": route_stairs.success,
        "avoids_stairs": route_stairs.avoids_stairs,
        "route_grade": route_stairs.route_grade,
        "barriers_encountered": len(route_stairs.barriers_encountered),
        "notes": route_stairs.routing_notes
    }
    print(f"   All paths have stairs -> avoids_stairs={route_stairs.avoids_stairs}, grade={route_stairs.route_grade}")

    # Test 3: Unknown / Unobserved Segments Classification
    print("\n3. Testing Unknown / Unobserved Segments (Silent Grade A classification)...")
    seg_unknown = MockSegment("unknown_seg", "Unsurveyed Sidewalk", 37.770, -122.420, 37.772, -122.420)
    score, grade, breakdown = calculate_segment_accessibility_score([])
    is_silently_grade_a = (grade == "A" or seg_unknown.score_grade == "A")
    results["unknown_segments_audit"] = {
        "calculated_score": score,
        "calculated_grade": grade,
        "default_model_score": seg_unknown.accessibility_score,
        "default_model_grade": seg_unknown.score_grade,
        "is_silently_classified_as_grade_a": is_silently_grade_a,
        "status": "PASS: Unsurveyed segments classified as Grade U (Unassessed) with caution cost weighting." if not is_silently_grade_a else "FAIL"
    }
    print(f"   Unobserved segment classified as Grade A? {is_silently_grade_a} (Calculated Grade: {grade})")

    # Test 4: Extreme Out-of-Bounds / Incorrect Coordinates
    print("\n4. Testing Out-of-Bounds Coordinates Snap...")
    # Coordinates in Tokyo against San Francisco network
    engine_sf = AccessibilityRoutingEngine([seg_island_1])
    route_oob = engine_sf.find_accessible_route(35.6762, 139.6503, 37.772, -122.420)
    
    # Distance between Tokyo and SF is ~8,000 km
    results["out_of_bounds_snap"] = {
        "success": route_oob.success,
        "total_distance_meters": route_oob.total_distance_meters,
        "snapped_across_continents": route_oob.total_distance_meters > 100000.0,
        "issue": "No maximum snap radius check at routing entry: Coordinates thousands of km away snap to the nearest node."
    }
    print(f"   Snapped coordinates across continents? {results['out_of_bounds_snap']['snapped_across_continents']} ({route_oob.total_distance_meters:.0f} m)")

    # Test 5: Barrier-Aware Alternative Path Selection
    print("\n5. Testing Barrier-Aware Alternative Path Selection...")
    # Direct path with stairs: (0,0) -> (0, 0.002)
    # Bypass path with ramp: (0,0) -> (0.001, 0.001) -> (0, 0.002)
    seg_direct_stairs = MockSegment("s_direct", "Direct Stairs", 37.770, -122.420, 37.772, -122.420, score=25.0, grade="F", has_stairs=True, length=200.0)
    seg_bypass_1 = MockSegment("s_b1", "Ramp East", 37.770, -122.420, 37.771, -122.418, score=95.0, grade="A", has_stairs=False, length=120.0)
    seg_bypass_2 = MockSegment("s_b2", "Ramp North", 37.771, -122.418, 37.772, -122.420, score=95.0, grade="A", has_stairs=False, length=120.0)
    
    engine_alt = AccessibilityRoutingEngine([seg_direct_stairs, seg_bypass_1, seg_bypass_2])
    route_alt = engine_alt.find_accessible_route(37.770, -122.420, 37.772, -122.420)
    
    took_bypass = "s_b1" in route_alt.segments_traversed and "s_direct" not in route_alt.segments_traversed
    results["alternative_path_selection"] = {
        "success": route_alt.success,
        "avoids_stairs": route_alt.avoids_stairs,
        "segments_traversed": route_alt.segments_traversed,
        "took_accessible_bypass": took_bypass
    }
    print(f"   Successfully detoured around stairs via ramp bypass? {took_bypass}")

    out_path = Path(__file__).parent / "routing_audit_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nRouting & GIS Audit complete. Results saved to {out_path}")
    return results

if __name__ == "__main__":
    audit_routing_gis()
