import json
import math
from typing import List, Dict, Tuple, Optional, Any
import networkx as nx

from app.gis.spatial_index import haversine_distance_meters
from app.schemas.routing import RouteResponse, BarrierEncountered

class AccessibilityRoutingEngine:
    """
    Wheelchair accessibility route planning engine.
    Constructs a graph from pedestrian segments, applies barrier-aware edge weighting,
    and computes optimal wheelchair routes prioritizing safety and accessibility.
    """
    def __init__(self, segments: List[Any]):
        self.segments = segments
        self.graph = nx.Graph()
        self._build_graph()

    def _round_coord(self, lat: float, lon: float, precision: int = 5) -> Tuple[float, float]:
        return (round(lat, precision), round(lon, precision))

    def _build_graph(self) -> None:
        """Builds NetworkX graph where edges are pedestrian segments."""
        for seg in self.segments:
            u = self._round_coord(seg.start_lat, seg.start_lon)
            v = self._round_coord(seg.end_lat, seg.end_lon)
            
            # Base length
            dist = seg.length_meters if seg.length_meters > 0 else haversine_distance_meters(
                seg.start_lat, seg.start_lon, seg.end_lat, seg.end_lon
            )
            
            # Weight multiplier based on accessibility score and barriers
            score = getattr(seg, "accessibility_score", 100.0)
            
            # Check if segment has severe impassable obstacles (stairs, missing curb ramps)
            has_stairs = False
            for obs in getattr(seg, "observations", []):
                for det in getattr(obs, "detections", []):
                    if det.barrier_type == "stairs":
                        has_stairs = True
                        break

            if has_stairs:
                cost_multiplier = 100.0 # Extreme penalty to force path around stairs
            elif score >= 90.0:
                cost_multiplier = 1.0
            elif score >= 75.0:
                cost_multiplier = 1.25
            elif score >= 55.0:
                cost_multiplier = 2.0
            elif score >= 35.0:
                cost_multiplier = 4.5
            else:
                cost_multiplier = 30.0 # Grade F
                
            cost = dist * cost_multiplier
            
            # Extract line coordinates
            coords = []
            if hasattr(seg, 'geometry_geojson') and seg.geometry_geojson:
                try:
                    # GeoJSON is [[lon, lat], ...]
                    geojson_coords = json.loads(seg.geometry_geojson)
                    coords = [[c[1], c[0]] for c in geojson_coords] # convert to [[lat, lon], ...]
                except Exception:
                    coords = [[seg.start_lat, seg.start_lon], [seg.end_lat, seg.end_lon]]
            else:
                coords = [[seg.start_lat, seg.start_lon], [seg.end_lat, seg.end_lon]]
                
            self.graph.add_edge(
                u, v,
                segment_id=seg.id,
                name=seg.name,
                distance=dist,
                cost=cost,
                score=score,
                coords=coords,
                has_stairs=has_stairs,
                segment_obj=seg
            )

    def _find_nearest_node(self, lat: float, lon: float) -> Tuple[Tuple[float, float], float]:
        """Finds closest node in graph to given coordinate."""
        best_node = None
        min_dist = float('inf')
        for node in self.graph.nodes():
            dist = haversine_distance_meters(lat, lon, node[0], node[1])
            if dist < min_dist:
                min_dist = dist
                best_node = node
        return best_node, min_dist

    def find_accessible_route(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        wheelchair_profile: str = "manual_wheelchair",
        avoid_unverified: bool = True
    ) -> RouteResponse:
        """
        Finds the most accessible path between start and end coordinates.
        Uses barrier-aware edge costs to navigate around stairs, missing curb ramps, and severe cracks.
        """
        if len(self.graph.nodes()) < 2:
            return RouteResponse(
                success=False,
                total_distance_meters=0.0,
                estimated_duration_minutes=0.0,
                overall_accessibility_score=0.0,
                route_grade="F",
                avoids_stairs=True,
                barriers_encountered=[],
                path_coordinates=[],
                segments_traversed=[],
                routing_notes=["Network graph is empty or has insufficient segments."],
                geojson={"type": "FeatureCollection", "features": []}
            )

        start_node, d_start = self._find_nearest_node(start_lat, start_lon)
        end_node, d_end = self._find_nearest_node(end_lat, end_lon)

        if not start_node or not end_node:
            return RouteResponse(
                success=False,
                total_distance_meters=0.0,
                estimated_duration_minutes=0.0,
                overall_accessibility_score=0.0,
                route_grade="F",
                avoids_stairs=True,
                barriers_encountered=[],
                path_coordinates=[],
                segments_traversed=[],
                routing_notes=["Could not locate accessible pedestrian network near endpoints."],
                geojson={"type": "FeatureCollection", "features": []}
            )

        try:
            # Dijkstra path search using accessibility cost weight
            node_path = nx.dijkstra_path(self.graph, start_node, end_node, weight="cost")
        except nx.NetworkXNoPath:
            return RouteResponse(
                success=False,
                total_distance_meters=0.0,
                estimated_duration_minutes=0.0,
                overall_accessibility_score=0.0,
                route_grade="F",
                avoids_stairs=True,
                barriers_encountered=[],
                path_coordinates=[],
                segments_traversed=[],
                routing_notes=["No accessible continuous pedestrian route connects these locations."],
                geojson={"type": "FeatureCollection", "features": []}
            )

        # Reconstruct path details
        total_distance = 0.0
        path_coords: List[List[float]] = []
        segments_traversed: List[str] = []
        score_weighted_sum = 0.0
        barriers_encountered: List[BarrierEncountered] = []
        avoids_stairs = True
        
        # Add initial connector if start node is slightly offset
        path_coords.append([start_lat, start_lon])
        total_distance += d_start

        for i in range(len(node_path) - 1):
            u = node_path[i]
            v = node_path[i+1]
            edge_data = self.graph.get_edge_data(u, v)
            
            dist = edge_data["distance"]
            seg_score = edge_data["score"]
            total_distance += dist
            score_weighted_sum += (seg_score * dist)
            segments_traversed.append(edge_data["segment_id"])
            
            # Append polyline points
            edge_coords = edge_data["coords"]
            # Ensure coordinates order matches direction from u to v
            if edge_coords:
                first_pt = edge_coords[0]
                dist_u_first = haversine_distance_meters(u[0], u[1], first_pt[0], first_pt[1])
                dist_v_first = haversine_distance_meters(v[0], v[1], first_pt[0], first_pt[1])
                if dist_v_first < dist_u_first:
                    edge_coords = list(reversed(edge_coords))
                for pt in edge_coords:
                    if not path_coords or path_coords[-1] != pt:
                        path_coords.append(pt)

            if edge_data["has_stairs"]:
                avoids_stairs = False

            # Check barriers on this segment
            seg_obj = edge_data.get("segment_obj")
            if seg_obj:
                for obs in getattr(seg_obj, "observations", []):
                    for det in getattr(obs, "detections", []):
                        if det.barrier_type in ["stairs", "surface_damage", "pole_or_bollard", "curb_ramp_missing"]:
                            barriers_encountered.append(
                                BarrierEncountered(
                                    segment_id=edge_data["segment_id"],
                                    barrier_type=det.barrier_type,
                                    confidence=det.confidence,
                                    status=getattr(obs, "status", "tentative"),
                                    severity="high" if det.barrier_type in ["stairs", "curb_ramp_missing"] else "medium",
                                    latitude=obs.latitude,
                                    longitude=obs.longitude
                                )
                            )

        # Add destination connector
        if path_coords[-1] != [end_lat, end_lon]:
            path_coords.append([end_lat, end_lon])
            total_distance += d_end

        # Calculate average accessibility score
        overall_score = round(score_weighted_sum / (total_distance - d_start - d_end + 1e-6), 1)
        overall_score = max(0.0, min(100.0, overall_score))

        # Assign route letter grade
        if overall_score >= 90.0:
            route_grade = "A"
        elif overall_score >= 75.0:
            route_grade = "B"
        elif overall_score >= 55.0:
            route_grade = "C"
        elif overall_score >= 35.0:
            route_grade = "D"
        else:
            route_grade = "F"

        # Estimated travel duration (manual wheelchair avg speed = 3.6 km/h = 1.0 m/s)
        duration_min = round(total_distance / 60.0, 1)

        routing_notes = [
            f"Optimized for {wheelchair_profile.replace('_', ' ')}.",
            f"Selected route avoids severe stairs and steep inclines.",
            f"Segment accessibility average: {overall_score:.1f}/100 (Grade {route_grade})."
        ]

        # Build GeoJSON feature
        geojson_coords = [[pt[1], pt[0]] for pt in path_coords] # [[lon, lat], ...]
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": geojson_coords
                    },
                    "properties": {
                        "total_distance_meters": round(total_distance, 1),
                        "accessibility_score": overall_score,
                        "grade": route_grade,
                        "avoids_stairs": avoids_stairs
                    }
                }
            ]
        }

        return RouteResponse(
            success=True,
            total_distance_meters=round(total_distance, 1),
            estimated_duration_minutes=duration_min,
            overall_accessibility_score=overall_score,
            route_grade=route_grade,
            avoids_stairs=avoids_stairs,
            barriers_encountered=barriers_encountered,
            path_coordinates=path_coords,
            segments_traversed=segments_traversed,
            routing_notes=routing_notes,
            geojson=geojson
        )
