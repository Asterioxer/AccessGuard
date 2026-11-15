# AccessGuard — Release Candidate Checklist (Gate A to G)

This checklist tracks readiness across all 7 acceptance gates defined in Mission 003 Section 13.

---

## Gate A — Functional Correctness
- [x] All 10 Mission 001 product requirements traced to working code.
- [x] End-to-end user journey verified (Register -> Upload -> CV -> GIS association -> Scoring -> Verification -> Routing).
- [x] Pagination on observations and segments implemented and tested.
- [x] Alternative route generation verified.
- [x] Database migrations formalized with Alembic.

## Gate B — Computer Vision
- [x] All adapters audited (`Heuristic`, `Torchvision`, `YOLO`, `Dummy`).
- [x] Evaluation dataset separated into 20-scene benchmark without data leakage.
- [x] Precision, Recall, F1, mAP@0.5 ($0.618$), and ECE ($0.268$) calibrated on held-out test data.
- [x] Image preprocessing guards added (CLAHE, blur detection, orientation).
- [x] Model cards created for Heuristic and Torchvision adapters.

## Gate C — GIS & Spatial Intelligence
- [x] Maximum endpoint snap distance enforced ($500\text{ m}$).
- [x] Unknown/unobserved segments assigned Grade "U" (Unassessed) with $2.5\times$ caution multiplier.
- [x] A* routing algorithm implemented with Haversine distance heuristic.
- [x] Disconnected graph and impassable network handling verified.
- [x] EPSG coordinate system standards documented in `docs/gis_and_routing.md`.

## Gate D — Cybersecurity & Privacy
- [x] Bandit SAST scan reports 0 High and 0 Medium vulnerabilities.
- [x] Restricted CORS policy enforced.
- [x] Production `SECRET_KEY` guard active.
- [x] Image decompression bomb protection ($16\text{ MP}$ limit) active.
- [x] EXIF privacy metadata stripped and faces/plates blurred.
- [x] API rate limiting active on `/auth` and `/observations/upload`.
- [x] RBAC permission matrix verified with dedicated integration tests (`test_rbac_matrix.py`).

## Gate E — Reliability & Performance
- [x] Full automated test suite passes (43/43 tests, 100% pass rate).
- [x] Frontend TypeScript build passes with zero errors (`dist/assets/index-DyOJLpG9.js: 362.04 kB`).
- [x] Prometheus metrics endpoint active (`/metrics`).
- [x] Video processing and concurrent upload benchmarks verified.
- [x] Docker Compose configuration validated (`docker compose config`).

## Gate F — Documentation & Artifacts
- [x] 7 audit documents generated in `docs/audit/`.
- [x] Updated `README.md`, `CONTRIBUTING.md`, `LICENSE`, `CHANGELOG.md`, `SECURITY.md`.
- [x] Updated architecture, database, API, and GIS documentation.
- [x] Demonstration walkthrough and senior interview question bank prepared.

## Gate G — Release Candidate Handover
- [x] Version bumped to 1.1.0 in `backend/app/config.py` and `frontend/package.json`.
- [x] All tracking files (`MISSION_STATUS.md`, `TASK_REGISTER.md`, `ENGINEERING_DECISIONS.md`, `SECURITY_FINDINGS.md`, `TEST_RESULTS.md`, `RELEASE_CHECKLIST.md`) complete.
- [x] Final completion report produced.
