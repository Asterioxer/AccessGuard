# Changelog — AccessGuard

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.3.0] - 2026-11-16

### Added
- **External API Connectors:** Built modular, SSRF-safe connectors for Mapillary v4, KartaView public imagery, Authorized Live RTSP/Webcam streams, Copernicus STAC satellite observation, and Pexels.
- **SSRF Protection & Host Whitelisting:** Enforced strict private IP / link-local / loopback blocking in `validate_remote_url` preventing internal network probing or cloud metadata leaks.
- **Automated Ingestion Engine & Priority Scheduler:** Built `AcquisitionEngine` and `IngestionScheduler` with spatial-temporal duplicate detection, exponential backoff, circuit-breaking, and dead-letter retry queues.
- **Pixel-Level Segmentation Engine:** Built `SegmentationAdapter` interface and `HeuristicSegmentationAdapter`, `ONNXSegmentationAdapter`, and `YOLOSegmentationAdapter` emitting genuine multi-point contour polygons and RLE masks across 10 standard accessibility classes.
- **Asynchronous Video Intelligence Pipeline:** Built `VideoProcessor` with configurable sampling rates, temporal obstacle tracking, and multi-frame deduplication (100 video frames -> 1 physical obstacle record).
- **Mission 005 Addendum Reality Test Engine:** Built `RealityEngine` evaluating independent unseen benchmarks against CPWD standards, measuring mAP, mIoU, calibration ECE, geospatial association accuracy, and environmental robustness.
- **Expanded Indian Streetscape Benchmark:** Expanded benchmark collection from 10 to 25 verified scenes with ground truth bounding boxes, segmentation polygons, and physical environmental slices.
- **Routing Engine Freshness & Unknown-Condition Guardrails:** Enhanced `routing_engine.py` to penalize stale observations (> 180 days) and prevent unassessed segments from silently certifying a route as accessible.
- **Autonomous Intelligence Frontend Dashboard:** Created `AutonomousDashboard.tsx` with Ingestion Management, Pixel Segmentation & Video Tracker, Reality Scorecard, and Geospatial Provenance views.
- **Automated Test Expansion:** Expanded backend test suite from 62 to **79 passing tests** across 23 test modules.

### Fixed
- **Benchmark Adapter Identical mAP Anomaly:** Diagnosed and resolved the root cause of identical mAP values across all 4 adapters by properly decoupling model-driven inference from heuristic fallbacks and loading native ONNX models directly.

---

## [1.2.0] - 2026-11-16

### Added
- **Hugging Face Dataset Ingestion:** Built streaming and selective sample loader for `Abtinzandi/Obstacle-Detection-Dataset-YOLO`, `ProjectSidewalk`, `ProjectSidewalk/RampNet`, and `ShuoZheLi/SVRDD_YOLO`.
- **Storage Safety Enforcer:** Added disk usage verification (`check_storage_safety()`) preventing unbounded dataset downloads from filling disk.
- **Dataset Manifest Generation:** Formalized provenance, license attribution, and split distributions in JSON manifests.
- **Unified Annotation Normalizer:** Standardized class mapping into 6 core categories (`curb_ramp`, `stairs`, `surface_damage`, `obstacle`, `crosswalk`, `unknown_infrastructure`) while strictly separating roadway asphalt cracks from pedestrian sidewalk defects.
- **ONNX Runtime Acceleration:** Built `ONNXAdapter` with dynamic batching, letterbox preprocessing, and hardware acceleration on CPU/CUDA.
- **Ultralytics YOLO Integration:** Updated `YOLOAdapter` with native YOLOv8n inference and seamless ONNX fallback.
- **Reproducible Google Colab Notebooks:** Authored 5 end-to-end notebooks in `notebooks/` covering exploration, normalization, training, validation, and ONNX packaging.
- **Live Camera Testing Interface:** Created `LiveCamera.tsx` in frontend and `POST /api/v1/cv/live-frame` in backend supporting webcam access, configurable FPS (snapshot, 1, 2, 5 FPS), bounding box canvas overlays, latency HUD, and camera flip.
- **Temporal Video Tracking & Deduplication:** Built `TemporalObstacleTracker` preventing duplicate reports across 30 FPS video streams via spatial IoU clustering ($\text{IoU} \ge 0.35$).
- **Evaluation & Benchmarks Dashboard:** Built `EvaluationDashboard.tsx` with ground-truth benchmark comparison, interactive confusion matrices, false positive/negative analysis, and live telemetry view.
- **Indian Street-Scene Benchmark:** Curated 10 verified evaluation scenes reflecting real Indian pedestrian hazards (open drains, vendor pushcarts, two-wheelers on footpaths) adhering to CPWD 2021 accessibility guidelines.
- **Model Card:** Added `docs/model_cards/onnx_yolo_adapter.md`.

---

## [1.1.0] - 2026-11-15

### Added
- **A\* Pathfinding Algorithm:** Added goal-directed A* routing with Haversine admissible heuristic alongside Dijkstra.
- **Alternative Direct Route Tradeoffs:** Route responses calculate both safest accessible path and direct shortest path, reporting detour distances and avoided staircases.
- **Image Preprocessing & Quality Safeguards:** Added blur detection ($Var(\nabla^2 I) < 80$) with uncertainty penalty and adaptive CLAHE contrast enhancement on low-contrast/fog scenes.
- **Prometheus Observability:** Added `/metrics` endpoint exposing HTTP request counts, status codes, and CV inference latencies.
- **Alembic Database Migrations:** Initialized versioned schema migrations (`146609277add_initial_accessguard_schema.py`) and documented upgrade/downgrade workflows.
- **Pagination:** Implemented `limit` and `offset` query parameters on `/api/v1/segments` and `/api/v1/observations`.
- **Role-Based Access Control:** Added `require_role(Role.ADMIN)` on `/api/v1/verifications/{id}/admin-override`.
- **Model Cards & Documentation:** Created `docs/model_cards/heuristic_cv_adapter.md`, `docs/model_cards/torchvision_adapter.md`, `docs/gis_and_routing.md`, `docs/dataset_documentation.md`, and `docs/interview_preparation.md`.

### Fixed
- **Stair False Positives on Zebra Crosswalks:** Fixed false detections by analyzing signed vertical gradient directions (`cv2.Sobel`), distinguishing flat alternating stripes from physical stair risers.
- **Silent Grade A Default for Unsurveyed Sidewalks:** Fixed unobserved segments silently claiming Grade A ($100.0/100$); unobserved segments now assign Grade `"U"` (Unassessed) with caution cost weighting ($2.5\times$).
- **Extreme Coordinate Snapping:** Enforced `MAX_SNAP_METERS = 500.0m` boundary on routing endpoints.
- **Unsafe PyTorch Deserialization:** Added `weights_only=True` to `torch.load()` to neutralize Bandit B614 code execution risk.
- **CORS Lockdown:** Restricted origins in `main.py` to explicit authorized origins.
- **Decompression Bomb Protection:** Added `MAX_IMAGE_PIXELS = 16_000_000` header checks.
- **API Rate Limiting:** Added sliding-window rate limiting on `/auth` and `/observations/upload` via `slowapi`.

---

## [1.0.0] - 2026-11-14

### Added
- Initial public-benefit release of AccessGuard (Mission 001).
- 10 core product requirements implemented across React/Leaflet frontend and FastAPI backend.
- Classical OpenCV heuristic detection for stairs, curb ramps, potholes, and bollards.
- PostGIS pedestrian network database schema.
- Transparent wheelchair accessibility scoring engine ($0 - 100$).
- Community human verification queue with consensus promotion thresholds.
- Docker Compose multi-container deployment stack.
