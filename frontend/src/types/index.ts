export interface BoundingBox {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
}

export interface BarrierEvidence {
  feature_type: string;
  measured_metric: number;
  threshold_applied: number;
  explanation: string;
  metadata?: Record<string, any>;
}

export interface BarrierDetection {
  id: string;
  barrier_type: string;
  confidence: number;
  uncertainty_score: number;
  is_confirmed: boolean;
  is_uncertain: boolean;
  bbox: BoundingBox;
  polygon?: number[][];
  evidence?: BarrierEvidence;
  created_at: string;
}

export interface Observation {
  id: string;
  contributor_id?: string;
  segment_id?: string;
  media_url: string;
  media_type: 'image' | 'video';
  latitude: number;
  longitude: number;
  status: 'tentative' | 'verified' | 'disputed' | 'rejected';
  verification_score: number;
  captured_at?: string;
  created_at: string;
  detections: BarrierDetection[];
}

export interface ScoringCriteriaBreakdown {
  base_score: number;
  stairs_penalty: number;
  curb_ramp_missing_penalty: number;
  surface_damage_penalty: number;
  obstruction_penalty: number;
  curb_ramp_bonus: number;
  verified_weight_applied: number;
  net_score: number;
  penalties_detail: string[];
  bonuses_detail: string[];
}

export interface PedestrianSegment {
  id: string;
  osm_way_id?: string;
  name: string;
  start_lat: number;
  start_lon: number;
  end_lat: number;
  end_lon: number;
  geometry_geojson: string;
  length_meters: number;
  accessibility_score: number;
  score_grade: string;
  score_breakdown: ScoringCriteriaBreakdown;
  barrier_count: number;
  last_evaluated_at: string;
}

export interface BarrierEncountered {
  segment_id: string;
  barrier_type: string;
  confidence: number;
  status: string;
  severity: string;
  latitude?: number;
  longitude?: number;
}

export interface DirectRouteComparison {
  direct_distance_meters: number;
  accessible_detour_meters: number;
  direct_route_has_stairs: boolean;
  stairs_avoided_by_accessible_route: number;
}

export interface RouteResponse {
  success: boolean;
  total_distance_meters: number;
  estimated_duration_minutes: number;
  overall_accessibility_score: number;
  route_grade: string;
  avoids_stairs: boolean;
  barriers_encountered: BarrierEncountered[];
  path_coordinates: [number, number][]; // [lat, lon][]
  segments_traversed: string[];
  routing_notes: string[];
  geojson: any;
  algorithm_used?: string;
  direct_route_comparison?: DirectRouteComparison;
}

export interface User {
  id: string;
  email: string;
  role: string;
  reputation_score: number;
  is_active: boolean;
  created_at: string;
}

export interface AnalyticsSummary {
  total_observations: number;
  verified_observations: number;
  tentative_observations: number;
  disputed_observations: number;
  verification_rate: number;
  average_city_accessibility_score: number;
  barrier_distribution: Record<string, number>;
  segment_grade_distribution: Record<string, number>;
}
