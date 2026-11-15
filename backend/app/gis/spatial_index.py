import math
from typing import List, Tuple, Optional, Any
from shapely.geometry import Point, LineString
import json

def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two GPS coordinates in meters."""
    R = 6371000.0 # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def snap_point_to_segments(
    lat: float,
    lon: float,
    segments: List[Any],
    max_distance_meters: float = 30.0
) -> Optional[Tuple[Any, float]]:
    """
    Finds the nearest pedestrian segment within max_distance_meters.
    Segments must have .geometry_geojson or .start_lat/.end_lat attributes.
    Returns (nearest_segment, distance_meters) or None if no segment is within range.
    """
    best_segment = None
    min_dist = float('inf')

    obs_point = Point(lon, lat)

    for seg in segments:
        try:
            # Parse coordinates
            if hasattr(seg, 'geometry_geojson') and seg.geometry_geojson:
                coords = json.loads(seg.geometry_geojson)
                line = LineString(coords) # [[lon, lat], ...]
            else:
                line = LineString([(seg.start_lon, seg.start_lat), (seg.end_lon, seg.end_lat)])

            # Distance in degrees approx
            nearest_geom_pt = line.interpolate(line.project(obs_point))
            
            # Convert to actual metric distance using haversine
            dist_m = haversine_distance_meters(lat, lon, nearest_geom_pt.y, nearest_geom_pt.x)
            
            if dist_m < min_dist and dist_m <= max_distance_meters:
                min_dist = dist_m
                best_segment = seg
        except Exception:
            continue

    if best_segment is not None:
        return best_segment, round(min_dist, 2)
    return None
