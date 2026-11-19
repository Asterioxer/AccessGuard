# AccessGuard Real-Data Workflow Status

## Purpose

This note defines the runtime contract for image, video, and pedestrian-network workflows. Synthetic benchmark scenes may remain in evaluation code, but they are not production observations and must never be inserted into the live map.

## Data sources

| Data | Permitted source | Runtime behavior |
|---|---|---|
| Street geometry | Explicit OpenStreetMap Overpass import | Imported with OSM way ID and ODbL attribution; grade starts Unassessed |
| Street photographs | User upload or provider response with source ID, attribution, and license metadata | Sanitized, hashed, processed, persisted with provenance |
| Street videos | User upload | Sampled frames pass through the canonical image-processing service; temporal tracking consolidates recurring detections |
| Detection labels | Genuine loaded model inference, or explicitly selected development heuristic | Missing/invalid weights are Unavailable; no dummy output for user uploads |
| Segmentation masks | Genuine loaded segmentation checkpoint | Missing weights are Unavailable; zero masks after successful inference remain a valid empty result |
| Synthetic scenes | Test fixtures / benchmark datasets only | Must be labelled synthetic and excluded from production map and real-world accuracy claims |

## Model asset inventory

The Mission 006.6 repository ledger records these assets as present and SHA-256 verified:

| Model asset | Repository path | Size | SHA-256 |
|---|---|---:|---|
| RampNet ConvNeXt-V2 | `data/models/rampnet.safetensors` | 360,247,780 bytes | `f2119e3…617866b5` |
| YOLOv8n ONNX | `data/models/yolov8n.onnx` | 12,851,105 bytes | `fbb1139…d7531d76` |
| YOLOv8n PyTorch | `data/models/yolov8n.pt` | 6,549,796 bytes | `f59b3d8…fc83b36` |
| YOLOv8n segmentation | `data/models/yolov8n-seg.pt` | 7,071,756 bytes | `a7cd8f9…c63c423c` |

GuideCane and CitySurfaces weights were reported unavailable in that ledger. Torchvision requires a configured compatible callable checkpoint. Asset presence and hashes establish file identity, not task accuracy. The Mission 006.6 audit's synthetic stairs fixture produced no fused barrier detections; that is not evidence of real-world performance.

## User workflow

1. Configure a private `ADMIN_INITIAL_PASSWORD` before startup. No default admin password is embedded in the application.
2. Import a small OpenStreetMap pedestrian bounding box from Autonomous → Ingestion. This endpoint requires an administrator session.
3. Upload a genuine street photograph or short video through Upload Sidewalk Observation. The asynchronous `/video/process` endpoint also uses the canonical `ImageProcessingService.process_video` path and persists the resulting observation.
4. Inspect the sanitized media, detections, and Processing Provenance panel. Video provenance includes per-frame adapter status, execution type, and output counts.
5. Treat unassessed OSM ways and tentative model predictions as unverified until reviewed by people with appropriate local knowledge.

## Privacy and provenance

- Do not add fabricated GPS coordinates or infer a location when the media has no GPS and the user supplied no coordinates.
- Preserve original and sanitized content hashes and media source attribution.
- Keep provider imagery license metadata separate from software/model code licenses.
- Do not claim live imagery from archived sources.

## Verification completed in CI

On PR #3 head `4ee2b3492bf971ada9a9152fa6685e85e4bbd2a1`, GitHub Actions run #34 passed all three jobs:

- Backend Tests & CV Evaluation: **148 tests passed**; CV evaluation runner completed.
- Frontend Build & Typecheck: **success**.
- Docker Compose Validation: **success**.

Run: https://github.com/Asterioxer/AccessGuard/actions/runs/44510001202

CI does not replace a real deployment smoke test.

## Remaining verification before release

- Exercise actual image and video uploads in a clean local deployment; inspect persisted media and provenance.
- Verify an actual small-area Overpass query and map rendering against live OSM data.
- Validate the installed model weights on a representative, independently labelled real streetscape dataset. Record per-class precision/recall, false positives, false negatives, and calibration. Do not use generated scenes as the sole evidence.
- Confirm local database cleanup only after backup. The included cleanup script defaults to dry-run and requires `--apply`.

## Release statement

The code changes remove fabricated production seed data, make real OSM import explicit, route video through the canonical image pipeline, and expose honest model execution telemetry. They do not establish accessibility detection accuracy without real-world evaluation and human verification.
