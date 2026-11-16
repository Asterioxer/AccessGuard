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

export interface LiveDetectionDTO {
  barrier_type: string;
  confidence: number;
  uncertainty_score: number;
  bbox: [number, number, number, number]; // [ymin, xmin, ymax, xmax]
  is_uncertain: boolean;
  is_confirmed: boolean;
  track_id?: string;
  evidence?: Record<string, any>;
}

export interface LiveFrameResponse {
  detections: LiveDetectionDTO[];
  latency_ms: number;
  fps_estimate: number;
  adapter_used: string;
  image_quality: {
    is_blurry: boolean;
    blur_variance: number;
    clahe_applied: boolean;
    resolution?: string;
  };
  privacy_redacted: boolean;
  active_tracks_count: number;
}

export interface IngestionProviderInfo {
  provider: string;
  is_available: boolean;
  is_authenticated: boolean;
  circuit_state: string;
  rate_limit_per_min: number;
  requests_last_minute: number;
  attribution: string;
  license: string;
}

export interface IngestionJob {
  id: string;
  provider: string;
  status: string;
  priority: string;
  bbox: [number, number, number, number];
  interval_minutes: number;
  items_fetched: number;
  items_processed: number;
  items_failed: number;
  error_message?: string;
  last_run_at?: string;
  next_run_at?: string;
  created_at: string;
}

export interface SegmentedRegion {
  class_id: number;
  class_name: string;
  polygon: [number, number][]; // [[x, y], ...] normalized 0-1
  confidence: number;
  bbox: [number, number, number, number];
  area_pixels: number;
  area_ratio: number;
  color_hex: string;
  color_rgb: [number, number, number];
  model_identifier: string;
  model_version: string;
}

export interface SegmentationResponse {
  image_width: number;
  image_height: number;
  region_count: number;
  regions: SegmentedRegion[];
  model_identifier: string;
  model_version: string;
}

export interface VideoJobProgress {
  job_id: string;
  status: string;
  progress_percent: number;
  total_frames: number;
  sampled_frames: number;
  duration_seconds: number;
  native_fps: number;
  unique_obstacles_count: number;
  confirmed_tracks: Array<{
    track_id: string;
    barrier_type: string;
    first_seen_sec: number;
    last_seen_sec: number;
    duration_sec: number;
    frame_count: number;
    max_confidence: number;
    bbox: [number, number, number, number];
    is_physical_obstacle: boolean;
  }>;
  error_message?: string;
}

export interface RealityScorecard {
  scorecard_id: string;
  evaluation_timestamp: string;
  dataset_manifest: string;
  model_version: string;
  total_scenes_evaluated: number;
  pipeline_execution_rate: number;
  consensus_grade: string;
  detection_metrics: {
    macro_precision: number;
    macro_recall: number;
    macro_f1: number;
    map_50: number;
    map_50_95: number;
    expected_calibration_error: number;
    total_ground_truth: number;
    total_predictions: number;
    per_class: Record<string, any>;
  };
  segmentation_metrics: {
    mean_iou: number;
    per_class_miou: Record<string, number>;
  };
  geospatial_accuracy: {
    mean_association_error_meters: number;
    percent_correctly_snapped: number;
  };
  temporal_deduplication: {
    suppression_efficiency_percent: number;
    frame_to_obstacle_reduction_ratio: string;
  };
  environmental_robustness: Record<string, Record<string, number>>;
  failed_predictions_count: number;
  failed_predictions_sample: Array<{
    sample_id: string;
    title: string;
    expected_hazard: string;
    expected_bbox: [number, number, number, number];
    environmental_context: Record<string, any>;
    diagnostic_reason: string;
  }>;
}

export interface ProvenanceRecord {
  id: string;
  observation_id?: string;
  provider: string;
  external_image_id?: string;
  sequence_id?: string;
  capture_timestamp?: string;
  ingestion_timestamp: string;
  latitude?: number;
  longitude?: number;
  heading?: number;
  source_attribution: string;
  license_metadata: string;
  model_version: string;
  raw_metadata: Record<string, any>;
}

