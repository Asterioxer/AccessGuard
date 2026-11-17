# MISSION 006.1 — IMPLEMENTATION REPORT
## Forensic Implementation Repairs and Architecture Modernization

**Project:** AccessGuard  
**Branch:** `feature/reality-execution-repair`  
**Execution Environment:** Windows 11, Intel Core i7, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), 24 GB System RAM  
**Author:** AccessGuard Core Intelligence Architecture Team  
**Date:** November 17, 2026  

---

## 1. Executive Summary

This report documents the forensic repair of 11 critical integrity defects discovered during the initial Phase 0 baseline audit of AccessGuard. The repairs span data streaming integrity, model identity and lifecycle management, upstream weight acquisition, pipeline unification, evidence fusion rigor, and evaluation transparency.

Every modification was implemented under the foundational principle: **AccessGuard must report what it actually did, not what its architecture claims it can do.**

---

## 2. Inventory of Forensic Repairs (The 11 Defects)

### Defect 1: Silent Synthetic Fallback in Dataset Streaming
- **Location:** `backend/app/cv/datasets/hf_loader.py`
- **Root Cause:** When Hugging Face parquet stream failed or went offline, `_stream_hf_dataset` called `_generate_reproducible_sample()` to fabricate synthetic items while returning them under real dataset names (`Abtinzandi/Obstacle-Detection-Dataset-YOLO`, `projectsidewalk/rampnet-dataset`).
- **Repair:**
  - Completely excised `_generate_reproducible_sample()` from `hf_loader.py`.
  - Introduced `DatasetAcquisitionError` which raises when network streaming or dataset reading fails.
  - Added strict `data_origin = "external_stream"` and required `image_sha256` hashing on every real sample.
  - Relocated any testing fixtures to `backend/app/cv/datasets/fixtures.py` with explicit `data_origin = "synthetic_fixture"`.

### Defect 2: Invented Default Bounding Boxes
- **Location:** `backend/app/cv/datasets/hf_loader.py`
- **Root Cause:** In `_standardize_annotation()`, if bounding boxes were missing from the stream row, the loader fabricated `(0.4, 0.4, 0.7, 0.7)` and assigned the class `"obstacle"`.
- **Repair:**
  - Removed box fabrication logic. If a sample lacks bounding boxes, it is skipped or recorded with an empty annotation list.
  - Only genuine bounding boxes extracted from parquet rows or metadata are emitted.

### Defect 3: Grey Placeholder Image Generation
- **Location:** `backend/app/cv/datasets/hf_loader.py`
- **Root Cause:** In `download_and_materialize()`, if image bytes were missing, it wrote a 128x128x128 grey JPEG image to disk.
- **Repair:**
  - Enforced that only valid image bytes (`len(raw_bytes) > 0` and non-blank) can be written.
  - Implemented `scripts/verify_dataset_integrity.py` which scanned the repository, identified 63 legacy blank grey placeholder files in `data/processed`, and documented them in `data/reports/dataset_integrity_report.md`.

### Defect 4: Model Adapters Reporting Success Despite Missing Weights
- **Location:** `backend/app/cv/adapters/base_v2.py`, `guidecane_adapter.py`, `citysurfaces_adapter.py`, `sidewalk_validator_adapter.py`
- **Root Cause:** Model adapters returned `load() -> True` and `AdapterStatus.DEGRADED` even when weights were missing from disk, masquerading as functional neural networks.
- **Repair:**
  - Added `AdapterStatus.UNAVAILABLE` and `AdapterStatus.FAILED`.
  - Added `allow_fallback: bool = False` default to all adapter `load()` and `predict_v2()` methods.
  - In strict mode (`allow_fallback=False`), missing weights result in `load() -> False`, `status = AdapterStatus.UNAVAILABLE`, and `error_message = "UNAVAILABLE: MODEL_WEIGHTS_NOT_FOUND (<weight_file>)"`.
  - In development fallback mode (`allow_fallback=True`), outputs are explicitly stamped with `execution_type = "heuristic_fallback"` and `is_fallback = True`.

### Defect 5: Degraded Heuristics Disguised as Upstream Models
- **Location:** `backend/app/cv/adapters/guidecane_adapter.py`, `citysurfaces_adapter.py`, `rampnet_adapter.py`
- **Root Cause:** If model weights were missing or predictions empty, adapters fell back to Sobel edge filters or Laplacian variance while labeling the results with upstream names (`GuideCaneAdapter`, `CitySurfaces`).
- **Repair:**
  - Separated genuine neural inference from baseline heuristics.
  - If a loaded neural model produces 0 detections on an image, it outputs 0 detections; it never triggers heuristic sobel detection when the real model ran.
  - Added `execution_type` ("model_inference" vs "heuristic_fallback"), `weight_path`, `weight_sha256`, and `upstream_repo` to every `AdapterInferenceResult`.

### Defect 6: Ingestion Engine Bypassing Hybrid Models
- **Location:** `backend/app/ingestion/engine.py`
- **Root Cause:** `AcquisitionEngine` was hardcoded to instantiate `CVPipeline(adapter_name="heuristic")` and `HeuristicSegmentationAdapter()`, completely bypassing the new `HybridVisionAdapter`.
- **Repair:**
  - Instantiated `HybridVisionAdapter(mode="RESOURCE_CONSTRAINED")` in `AcquisitionEngine`.
  - Routed all acquired imagery through `hybrid_adapter.predict_v2()`.
  - Attached full multi-model telemetry and cryptographic `image_sha256` to every `ImageProvenance` record.

### Defect 7: Unverified Model Execution Claims in Hybrid Benchmarking
- **Location:** `scripts/run_hybrid_benchmarks.py`
- **Root Cause:** The benchmark ran mock adapters without verifying whether weights existed on disk and used `f1 * 0.96` as an empirical proxy for mAP.
- **Repair:**
  - Verified genuine weights for YOLO (`yolov8n.pt`/`onnx`) and RampNet (`rampnet.safetensors`).
  - Recorded exact execution types and checksums for all evaluated pipelines.
  - Relabeled synthetic scenes accurately.

### Defect 8: Evidence Fusion False-Positive Inflation
- **Location:** `backend/app/cv/fusion/evidence_fusion.py`, `scripts/run_hybrid_benchmarks.py`
- **Root Cause:** Evaluation matching allowed IoU as low as 0.15 with center-in-box matching, and permitted category cross-merging (e.g. matching `curb_ramp_present` with `curb_ramp_missing` or `raised_curb`).
- **Repair:**
  - Enforced strict Pascal VOC IoU >= 0.50 matching.
  - Removed all category cross-merging rules. `curb_ramp_present` is NEVER counted as a true positive for `curb_ramp_missing`.

### Defect 9: Hardcoded Metrics in RealityEngine
- **Location:** `backend/app/cv/evaluation/reality_engine.py`
- **Root Cause:** Lines 220-227 hardcoded `1.85m` mean association error, `94.2%` snapped, `96.8%` suppression efficiency, and `"30:1"` frame reduction ratio.
- **Repair:**
  - Replaced hardcoded literals with `status: "NOT_MEASURED"` and explicit explanations indicating benchmark images are single-frame captures without PostGIS segment ground-truth.

### Defect 10: Segmentation Evaluation Class Matching Defect
- **Location:** `backend/app/cv/evaluation/reality_engine.py`
- **Root Cause:** In mask IoU evaluation, the loop took `max(compute_mask_iou(reg.mask, gt_mask))` across ALL predicted regions without verifying that `reg.label == gt_hazard`.
- **Repair:**
  - Added strict class verification: `if reg.mask is not None and getattr(reg, "label", None) == gt_hazard:`.

### Defect 11: Inconsistent Reporting and Dead URL Fixtures
- **Location:** `backend/app/ingestion/connectors/mapillary.py`, `kartaview.py`
- **Root Cause:** When API tokens were absent, connectors generated fake items with dead URLs (`https://images.mapillary.com/fixture_1.jpg`), causing browser 404s and grey boxes.
- **Repair:**
  - When tokens are absent, connectors return `items = []`, `status = "unauthenticated"`, and `error_message = "CREDENTIALS_MISSING: ..."` unless explicit `allow_fixtures=True` is passed for unit tests.

---

## 3. Upstream Model Weights Acquisition

During Phase 2, authentic pretrained weights for RampNet were successfully acquired from Hugging Face:
- **Repository:** `projectsidewalk/rampnet-model`
- **Model Architecture:** ConvNeXt-V2-Base (`convnextv2_base.fcmae_ft_in22k_in1k_384`) + 1x1 Conv Heatmap Head (90,050,561 parameters)
- **Local Path:** `data/models/rampnet.safetensors`
- **File Size:** 360,247,780 bytes
- **SHA-256 Checksum:** `f2119e3becb0b551fa1470f7b7ba85b82122a3f73a6ed2a85609dd57617866b5`
- **Execution Target:** Native PyTorch with CUDA acceleration on NVIDIA RTX 3050 Laptop GPU (also verified on CPU).

---

## 4. Verification and Regression Testing

All changes were validated against the test suite:
- `backend/tests/test_hf_datasets.py`: 6 passed in 0.31s.
- `backend/tests/test_hybrid_vision.py`: 8 passed in 13.10s.
- `backend/tests/test_real_image_hybrid_execution.py`: 3 passed in 14.17s.
- Execution ledger generated at `data/runs/mission_006_1/run_1794911069_116131/`.
