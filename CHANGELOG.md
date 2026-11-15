# Changelog — AccessGuard

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
