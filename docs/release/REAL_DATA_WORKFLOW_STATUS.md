# AccessGuard Real-Data Workflow Status

## Purpose

This note defines the honest runtime contract for the image, video, and pedestrian-network workflows. Synthetic benchmark scenes may remain in test/evaluation code, but they are not production observations and must never be inserted into the live map.

## Data sources

| Data | Permitted source | Runtime behavior |
|---|---|---|
| Street geometry | Explicit OpenStreetMap Overpass import | Imported with OSM way ID; grade remains Unassessed |
| Street photographs | User upload or a provider response with source ID, attribution, and license metadata | Sanitized, hashed, processed, and persisted with provenance |
| Street videos | User upload | Frames sampled and passed through the canonical image-processing service; temporal tracking consolidates repeated detections |
| Detection labels | Genuine loaded model inference, or explicitly selected development heuristic | Missing weights are Unavailable; no dummy output for user uploads |
| Segmentation masks | Genuine loaded segmentation checkpoint | Missing weights are Unavailable; zero masks after successful inference remain a valid empty result |
| Synthetic scenes | Test fixtures / benchmark datasets only | Must be labelled synthetic and excluded from production map and real-world accuracy claims |

## Model asset audit

The repository currently contains paths named `data/models/yolov8n.onnx`, `data/models/yolov8n.pt`, and `data/models/yolov8n-seg.pt`, but GitHub returns zero-byte content for each path. They are not usable model checkpoints. The Torchvision adapter has no configured callable checkpoint. RampNet, GuideCane, CitySurfaces, and Sidewalk Validator model assets are not established by this repository inspection.

Therefore, the strict runtime must report these adapters as Unavailable until genuine, compatible weights are provisioned and validated. Do not interpret an empty prediction list from an unavailable model as a successful negative detection.

## User workflow

1. Import a small OpenStreetMap pedestrian bounding box from Autonomous → Ingestion. This endpoint requires an administrator session.
2. Upload a genuine street photograph or short video through Upload Sidewalk Observation. The asynchronous `/video/process` endpoint also uses the canonical `ImageProcessingService.process_video` path and persists the resulting observation.
3. Inspect the original sanitized media and detections.
4. Inspect Processing Provenance to see each adapter's status, execution type, and output count. Video jobs also return a per-adapter summary across sampled frames and the persisted observation ID.
5. Treat unassessed OSM ways and tentative model predictions as unverified until reviewed by people with appropriate local knowledge.

## Privacy and provenance

- Do not add fabricated GPS coordinates or infer a location when the media has no GPS and the user supplied no coordinates.
- Preserve original and sanitized content hashes and the media source attribution.
- Keep provider imagery license metadata separate from software/model code licenses.
- Do not claim live imagery from archived sources.

## Verification completed in CI

On PR #2 head `b56be63a01b77056dd35906c5112dd488f909177`, GitHub Actions run #27 passed all three jobs:

- Backend Tests & CV Evaluation: 130 tests passed; CV evaluation runner completed successfully.
- Frontend Build & Typecheck: TypeScript and production build passed.
- Docker Compose Validation: `docker compose config` passed.

The async video route, OSM importer, strict empty-checkpoint behavior, and frontend build are covered by automated tests or build validation. CI does not replace a real deployment smoke test.

## Verification still required before release

- Exercise an actual image upload and an actual video upload in a clean local deployment, then inspect persisted media and provenance.
- Verify an actual small-area Overpass query and map rendering against live OSM data.
- Provision compatible accessibility-specific model weights, record their source, license, version, and SHA-256, and verify that model inference is genuine.
- Provision compatible accessibility-specific model weights, record their source, license, version, and SHA-256, and verify that model inference is genuine.
- Verify OSM import against a real small-area query and confirm the map zooms to imported geometry.
- Inspect an existing local database separately; this change prevents new fixture seeding but intentionally does not delete old rows or media.

## Release statement

The code changes make the product fail honestly instead of fabricating data. They do not, by themselves, supply trained accessibility-specific weights or prove detection accuracy. Release claims must remain limited to the capabilities verified with genuine source media and compatible model assets.
