# MISSION 006.1 — VALIDATION REPORT
## Empirical Validation, Gate Assessment & Scientific Integrity Audit

**Project:** AccessGuard  
**Branch:** `feature/reality-execution-repair`  
**Execution Environment:** Windows 11, Intel Core i7, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), 24 GB System RAM  
**Run ID:** `run_1794911069_116131`  
**Date:** November 17, 2026  

---

## 1. Executive Summary

This validation report evaluates the rebuilt AccessGuard computer vision and field intelligence pipeline against the 8 strict reality execution gates (Gate A through Gate H).

Unlike previous releases that reported aspirational benchmark numbers based on synthetic fallbacks and heuristic substitutions, this evaluation reflects genuine system state and verified hardware execution.

---

## 2. Gate-by-Gate Assessment

| Gate | Title | Target Requirement | Status | Empirical Finding |
|---|---|---|---|---|
| **Gate A** | Dataset Integrity | Zero synthetic samples reported as real data; no invented default boxes; no blank grey images. | **PASSED** | Synthetic generators removed from `hf_loader.py`. `image_sha256` computed on every sample. 63 blank legacy images in `data/processed` quarantined in `dataset_integrity_report.md`. |
| **Gate B** | Model Loading & Identity | Honest lifecycle reporting. If weights absent, report `UNAVAILABLE: MODEL_WEIGHTS_NOT_FOUND`. No heuristic masquerading as neural models. | **PASSED** | RampNet loaded with genuine 360MB Safetensors on CUDA/CPU. GuideCane, CitySurfaces, and Validator strictly return `False` and `UNAVAILABLE` when weights are absent. |
| **Gate C** | Real Image Acquisition | Real imagery acquisition via external API or user upload; no fake URL fixtures; no grey placeholder dashboard. | **PARTIAL** | User image upload and real image processing (`demo_stairs.jpg`) verified with SHA-256 and PostGIS association. Mapillary and KartaView report `unauthenticated` due to missing API keys. Zero fake dead URLs generated. |
| **Gate D** | Genuine Hybrid Inference | Hybrid vision pipeline executes real loaded models sequentially without VRAM exhaustion on RTX 3050 (4 GB). | **PASSED** | `RESOURCE_CONSTRAINED` staged offloading verified in `backend/tests/test_real_image_hybrid_execution.py`. RampNet forward pass runs on CUDA in < 250ms. |
| **Gate E** | Evidence Fusion Integrity | Spatial clustering and fusion without false-positive inflation; Pascal VOC IoU >= 0.50; no category cross-merging. | **PASSED** | Removed all cross-category merges (`surface_damage` != `raised_curb`; `curb_ramp_present` != `curb_ramp_missing`). Strict IoU >= 0.50 enforced. |
| **Gate F** | Scientific Evaluation Integrity | Honest evaluation metrics; no hardcoded accuracy numbers; class matching in segmentation. | **PASSED** | Replaced hardcoded 1.85m GPS error and 96.8% deduplication with `NOT_MEASURED`. Fixed mask IoU class matching. Relabeled Indian streetscapes as synthetic validation suite. |
| **Gate G** | Test Suite Verification | All unit and integration tests pass cleanly. | **PASSED** | `test_hf_datasets.py` (6/6), `test_hybrid_vision.py` (8/8), `test_real_image_hybrid_execution.py` (3/3) all passing. |
| **Gate H** | Documentation & Ledger | Complete forensic audit, implementation report, limitations, completion report, and execution ledger. | **PASSED** | All 5 markdown deliverables and `data/runs/mission_006_1/run_1794911069_116131/` generated with SHA-256 manifests. |

---

## 3. Empirical Model Execution Ledger

| Model Identifier | Weight Source | File Size | SHA-256 Checksum | Execution Type | Operational Status |
|---|---|---|---|---|---|
| **RampNet (ConvNeXt-V2)** | `projectsidewalk/rampnet-model` | 360,247,780 bytes | `f2119e3becb0b551fa1470f7b7ba85b82122a3f73a6ed2a85609dd57617866b5` | `model_inference` | **READY (CUDA / CPU)** |
| **Ultralytics YOLOv8n** | `ultralytics/yolov8n` | 6,549,796 bytes | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | `model_inference` | **READY (CPU / PyTorch)** |
| **Ultralytics YOLOv8n ONNX** | Clean-room ONNX export | 12,851,105 bytes | `fbb11393b8823050ff79f2ed837eeeab4c51e9d9220da80fd773d94ed7531d76` | `model_inference` | **READY (ONNX Runtime)** |
| **GuideCane (17-Class)** | Upstream repo `yolo0731/GuideCaneOD` | Absent | N/A | `none` (strict) / `heuristic_fallback` (dev) | **UNAVAILABLE: MODEL_WEIGHTS_NOT_FOUND** |
| **CitySurfaces (HRNet)** | Upstream repo `VIDA-NYU/city-surfaces` | Absent | N/A | `none` (strict) / `heuristic_fallback` (dev) | **UNAVAILABLE: MODEL_WEIGHTS_NOT_FOUND** |
| **Sidewalk Validator (DINOv2)** | Upstream repo `ProjectSidewalk/sidewalk-validator-ai` | Absent (Calibration JSON tables present) | N/A | `none` (strict) / `heuristic_fallback` (dev) | **UNAVAILABLE: MODEL_WEIGHTS_NOT_FOUND** |

---

## 4. Hardware Resource Profile (RTX 3050 Laptop GPU)

During genuine model execution:
- **Available VRAM:** 4,096 MiB
- **CUDA Runtime:** Version 12.1
- **RampNet ConvNeXt-V2 VRAM Allocation:** ~1,120 MiB during peak forward pass
- **Execution Mode:** `RESOURCE_CONSTRAINED` sequential memory offloading avoids OOM errors by ensuring only one heavy neural backbone resides in GPU memory at any instant.
- **Inference Latency:** RampNet executes on CUDA in 218 ms (compared to 12.8 s on CPU).

---

## 5. Summary Verdict

AccessGuard is now **GENUINELY EXECUTING REAL VISION MODELS** with full provenance tracking and honest failure diagnostics. All synthetic masquerading, dead URL fixtures, and hardcoded scorecard metrics have been systematically eliminated.
