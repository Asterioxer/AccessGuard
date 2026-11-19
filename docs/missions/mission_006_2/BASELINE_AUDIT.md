# ACCESSGUARD — MISSION 006.2 FORENSIC BASELINE AUDIT
## Architectural Inconsistencies, Dual Inference Paths, Heuristic Masking & Execution Integrity

**Document ID:** `AG-M006-2-AUDIT-001`  
**Date:** 2026-11-17  
**Branch Audited:** `feature/reality-execution-repair` (Commit: `ac518a55280b11c6d37617b4c4e7f339f4d1e21b`)  
**Hardware Environment:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), 24 GB System RAM, Windows 11  
**Lead Auditor:** Agent 1 (Chief Architect & Mission Controller)

---

### Executive Summary

Mission 006.1 successfully repaired synthetic sample masquerading in dataset streaming, loaded authentic RampNet ConvNeXt-V2 weights on CUDA, and enforced basic failure semantics. However, an exhaustive forensic baseline audit of the codebase reveals critical architectural bifurcations and integrity flaws across model execution, segmentation, evidence fusion, and evaluation.

The system currently violates the foundational principle of a unified accessibility intelligence platform: **different entry points for real-world imagery execute completely different inference pipelines, produce inconsistent evidence structures, and apply diverging validation standards.**

---

### Confirmed Architectural Defects & Forensic Inventory

#### Defect 1: Dual Inference Architectures for Real-World Imagery
- **Files & Locations:**
  - `backend/app/api/v1/observations.py` (`upload_observation`, lines 143–161)
  - `backend/app/ingestion/engine.py` (`execute_job`, lines 192–195)
- **Existing Behavior:**
  - When a user uploads a real street photo via `/api/v1/observations/upload`, the backend instantiates `CVPipeline(adapter_name=adapter or settings.CV_ADAPTER)`. This executes a single monolithic model (defaulting to legacy heuristic or single YOLO) without multi-model specialist adapters, without candidate verification via Sidewalk Validator, and without evidence fusion.
  - In contrast, automated external ingestion in `engine.py` invokes `HybridVisionAdapter(mode="RESOURCE_CONSTRAINED")` running YOLO, RampNet, and EvidenceFusionEngine.
- **Why Defective:**
  - An image uploaded by a wheelchair user follows a degraded, legacy inference path, while the same street photograph retrieved via Mapillary or KartaView executes the multi-expert hybrid vision pipeline.
  - Outputs, confidence calibrations, uncertainty scores, and evidence metadata diverge depending solely on image ingress route.
- **Required Correction:**
  - Create a single, canonical `ImageProcessingService` (`backend/app/cv/orchestration/image_processing_service.py`).
  - Both user uploads and autonomous ingestion must delegate all decoding, privacy sanitization, model execution, segmentation, evidence fusion, and persistence to this unified service.
- **Dependencies & Regressions:**
  - `upload_observation` API signature must maintain backward compatibility while returning structured `ImageProcessingResult` telemetry.

---

#### Defect 2: Heuristic Contour Segmentation Masquerading as Model Inference
- **Files & Locations:**
  - `backend/app/ingestion/engine.py` (lines 197, 239–251)
  - `backend/app/cv/segmentation/yolo_seg_adapter.py` (lines 51–66)
  - `backend/app/cv/segmentation/onnx_seg_adapter.py` (lines 43, 58–60)
- **Existing Behavior:**
  - In `engine.py`, `self.seg_adapter = HeuristicSegmentationAdapter()` runs OpenCV edge detection and color thresholding, persisting generated contours into `segmented_observations` table with `model_identifier="heuristic_segmentation_adapter"` and presenting them alongside genuine detections.
  - In `yolo_seg_adapter.py` and `onnx_seg_adapter.py`, when upstream weights are missing, the adapters silently catch the absence, set `_fallback = HeuristicSegmentationAdapter()`, and return heuristic contours while stamping them with `model_identifier="yolo_segmentation_adapter"` or `"onnx_segmentation_adapter"`.
- **Why Defective:**
  - Procedural morphological thresholding is being falsely presented as deep neural network segmentation inference.
  - When genuine segmentation weights are missing, the system silently fakes success instead of reporting `AdapterStatus.UNAVAILABLE`.
- **Required Correction:**
  - Decouple object detection from semantic segmentation.
  - Enforce strict failure semantics: if neural segmentation weights are missing, the segmentation pipeline must explicitly report `AdapterStatus.UNAVAILABLE` with `MODEL_WEIGHTS_NOT_FOUND`.
  - Heuristic contours must never be saved into `SegmentedObservation` or attributed to neural models.
- **Dependencies & Regressions:**
  - UI must gracefully handle `segmentation_status: "UNAVAILABLE"` without crashing or assuming masks are always present.

---

#### Defect 3: Reality Engine Evaluates Legacy Pipeline Rather Than Unified Hybrid Architecture
- **Files & Locations:**
  - `backend/app/cv/evaluation/reality_engine.py` (`__init__` lines 40–45, `run_reality_evaluation` lines 114, 129)
- **Existing Behavior:**
  - `RealityEngine` explicitly initializes `self.cv_pipeline = cv_pipeline or CVPipeline()` and `self.seg_adapter = seg_adapter or HeuristicSegmentationAdapter()`.
  - In line 114, it calls `preds = self.cv_pipeline.adapter.predict(img)` and in line 129 calls `seg_regions = self.seg_adapter.predict_masks(img)`.
- **Why Defective:**
  - The benchmark and validation engine is completely disconnected from the hybrid vision and evidence fusion architecture.
  - Evaluation scorecards measure the legacy single-adapter pipeline and heuristic segmenter, providing zero scientific evidence regarding the performance of the actual multi-model hybrid orchestrator.
- **Required Correction:**
  - Refactor `RealityEngine` to evaluate the unified `ImageProcessingService` and `HybridVisionAdapter`.
  - Report separate metric dimensions: Execution Integrity, Model Integrity, Output Integrity, Accuracy (mAP@50, mAP@50:95, mIoU), and Hardware Performance.
- **Dependencies & Regressions:**
  - Test suites (`test_reality_engine.py`) must be updated to expect unified service outputs.

---

#### Defect 4: Evidence Fusion Synthesizes Hardcoded Bounding Boxes & Unconstrained Spatial Merging
- **Files & Locations:**
  - `backend/app/cv/fusion/evidence_fusion.py` (lines 167–180)
  - `backend/app/cv/fusion/spatial_association.py` (lines 118–130)
- **Existing Behavior:**
  - In `evidence_fusion.py`, if surface roughness exceeds 0.65, it automatically appends a `FusedBarrierEvidence` with a fabricated fixed bounding box: `bbox=(0.60, 0.20, 0.95, 0.80)`.
  - In `spatial_association.py`, `cluster_spatial_observations` merges any two detections with bounding box `IoU >= 0.40` into the same `SpatialCluster` without checking class taxonomy compatibility or label semantics.
- **Why Defective:**
  - Synthesizing arbitrary spatial coordinates `(0.60, 0.20, 0.95, 0.80)` violates the directive: *"A bounding box must never be converted into a purported genuine segmentation mask. Do not invent bounding boxes."*
  - Merging disparate classes (e.g. `pole_or_bollard` with `stairs` or `curb_ramp`) because their bounding boxes overlap corrupts the fused taxonomy and distorts confidence calculations.
- **Required Correction:**
  - Excised all synthetic coordinate injection in `evidence_fusion.py`.
  - Enforce class-aware spatial association: only merge compatible barrier categories (e.g., pedestrian infrastructure vs obstacles vs surface qualities).
  - Explicitly preserve contradictions and disagreements between models without collapsing them into arbitrary selections.
- **Dependencies & Regressions:**
  - `test_evidence_fusion_engine_conflict_and_consensus` must be updated to assert class-aware clustering.

---

#### Defect 5: Unsafe Weight Loading & Missing Checkpoint Validation in Specialist Adapters
- **Files & Locations:**
  - `backend/app/cv/adapters/citysurfaces_adapter.py` (lines 101–103)
  - `backend/app/cv/adapters/guidecane_adapter.py` (lines 116–122)
- **Existing Behavior:**
  - In `citysurfaces_adapter.py`, line 101 executes `self.model = torch.load(valid_path, map_location=self._device, weights_only=True); self.model.eval()`. If the file is a PyTorch `state_dict` (standard for upstream CitySurfaces checkpoints), `torch.load` returns a dictionary, causing `AttributeError: 'dict' object has no attribute 'eval'`.
  - In `guidecane_adapter.py`, any `.pt` file placed at `data/models/guidecane_yolov8.pt` is loaded via `YOLO(valid_path)` without verifying that the network architecture has 17 obstacle classes or matches the GuideCane taxonomy.
- **Why Defective:**
  - File existence at a path does not verify checkpoint architecture, weight provenance, or tensor compatibility.
  - A generic 80-class COCO YOLO model could be renamed `guidecane_yolov8.pt` and falsely executed as GuideCane.
- **Required Correction:**
  - Implement checkpoint identity and architectural validation in all adapters.
  - Verify tensor keys, model classes, and output heads. If weights fail validation or are missing, strictly set `AdapterStatus.UNAVAILABLE`.
- **Dependencies & Regressions:**
  - Model verification scripts must assert against checkpoint integrity rules.

---

#### Defect 6: Ingestion Engine & Observation Endpoint Coordinate Invention
- **Files & Locations:**
  - `backend/app/api/v1/observations.py` (lines 134–137)
- **Existing Behavior:**
  - When an uploaded image lacks EXIF GPS coordinates and no form latitude/longitude is provided, line 135 defaults the coordinates to downtown San Francisco:
    `final_lat = 37.7765`, `final_lon = -122.4194`.
- **Why Defective:**
  - Automatically inventing geospatial coordinates places unanchored field observations onto a real-world sidewalk segment in California, corrupting the GIS accessibility scoring database.
- **Required Correction:**
  - Preserve `latitude: Optional[float] = None` and `longitude: Optional[float] = None` when unavailable.
  - Never fabricate geographic coordinates. If coordinates are absent, mark the observation as unanchored; skip pedestrian segment snapping without failing image processing.
- **Dependencies & Regressions:**
  - GIS scoring recalculation must gracefully ignore unanchored observations.

---

#### Defect 7: Execution Ledger Commit & Cleanliness Discrepancies
- **Files & Locations:**
  - `data/runs/mission_006_1/run_1794911069_116131/run_manifest.json`
- **Existing Behavior:**
  - The run manifest recorded git state prior to final staging and documentation authoring, leaving potential divergence between recorded SHA and the final merged commit.
- **Why Defective:**
  - The ledger must refer to the exact, clean git commit SHA under which tests and benchmarks were executed.
- **Required Correction:**
  - Build `scripts/generate_mission_ledger.py` to verify working tree cleanliness, compute exact commit SHA, and record all runtime artifact hashes.

---

### Baseline Defect Matrix

| ID | Component | Defective Mechanism | Impact | Priority |
| :--- | :--- | :--- | :--- | :--- |
| **D1** | `api/v1/observations.py` | Invokes legacy `CVPipeline` instead of `HybridVisionAdapter` | Bypasses multi-model fusion for user uploads | **CRITICAL** |
| **D2** | `cv/segmentation/*` | Heuristic contours stamped as model inference; silent fallback | Fabricates neural segmentation evidence | **CRITICAL** |
| **D3** | `cv/evaluation/reality_engine.py` | Evaluates legacy pipeline & heuristic masks directly | Invalidates benchmark and reality scorecard claims | **CRITICAL** |
| **D4** | `cv/fusion/evidence_fusion.py` | Injects synthetic bbox `(0.60, 0.20, 0.95, 0.80)`; class-agnostic box merging | Synthesizes fake barrier boxes and merges conflicting classes | **HIGH** |
| **D5** | `cv/adapters/*` | Missing checkpoint validation; crashes on state_dict in CitySurfaces | Risk of loading incompatible weights or misidentifying models | **HIGH** |
| **D6** | `api/v1/observations.py` | Defaults missing GPS to `(37.7765, -122.4194)` | Corrupts GIS database with false geographic associations | **MEDIUM** |
| **D7** | `data/runs/` | Run ledger captured during intermediate execution state | Incomplete reproducibility chain | **MEDIUM** |

---

### Audit Sign-off

The architectural defects documented above are confirmed by static and dynamic code inspection. The implementation phases of Mission 006.2 will systematically resolve these defects, unifying all entry points under a single verified processing contract.
