# Upstream Compatibility & Integration Matrix
**AccessGuard Hybrid Vision Intelligence (Mission 006)**  
**Feature Branch:** `feature/hybrid-vision-intelligence`  
**Date:** November 2026  
**Status:** COMPLETE — RIGOROUS VERIFICATION

---

## 1. Overview & Evaluation Methodology

This matrix provides the authoritative due diligence assessment of all upstream candidate components across technical, legal, computational, and architectural dimensions. Every component is evaluated against AccessGuard's production requirements:
- **Language / Runtime:** Python 3.11+, PyTorch 2.0+, FastAPI, pydantic v2.
- **Licensing Constraint:** AccessGuard is published under the **MIT License**. No upstream dependencies or imported code may impose viral copyleft obligations (e.g. AGPL/GPL) or conflict with open commercial/public usage without explicit containment.
- **Hardware Constraint:** Must function efficiently on an **NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)** and standard 16–24GB system RAM, with seamless **CPU fallback** (`torch.device("cpu")`).
- **Failure Semantics:** Models must report structured health/lifecycle states. Failure or uninitialized state must never silently output an empty list of detections.

---

## 2. Component Evaluation Matrix

| Component Identifier | Upstream Repository & Lineage | License | Architecture & Tech Stack | 4GB VRAM Fit (RTX 3050) | Maintenance Status | Integration Verdict | Notes & Action Required |
|---|---|---|---|---|---|---|---|
| **PS-RAMP-01** (RampNet Detector) | `ProjectSidewalk/RampNet` | MIT | ConvNeXt-V2 Base (PyTorch, Hugging Face Hub) | **Moderate** (~1.8GB FP16 / ~2.6GB FP32). Runs comfortably solo. | Active (ICCV'25 / 2026 updates) | `APPROVED FOR PROTOTYPE` | Wrap in `RampNetAdapter`. Must use corrected 1-to-1 matching and address equirectangular seam. Sequential load or CPU fallback. |
| **PS-VAL-01** (Sidewalk Validator) | `ProjectSidewalk/sidewalk-validator-ai` | MIT | DINOv2-Large Classifier (PyTorch, Depth-Anything-V2) | **Heavy** (~2.1GB FP16 / ~3.4GB FP32). Marginal on 4GB VRAM with other nets. | Active (Round 3 Oct 2026) | `APPROVED FOR PROTOTYPE` | Integrate as verification stage. Apply calibration tables from `sidewalk-ai-api`. Offload detector before running. |
| **PS-API-01** (Sidewalk AI API) | `ProjectSidewalk/sidewalk-ai-api` | MIT | Flask REST API, Torch, OpenCV | N/A (Server container) | Active (v1.2.0 Oct 2026) | `APPROVED (REFERENCE ARCHITECTURE)` | Do not vendor Flask API directly; adopt its calibration tables and schema into AccessGuard FastAPI. |
| **PS-WEB-01** (Sidewalk Webpage) | `ProjectSidewalk/SidewalkWebpage` | MIT | Scala, Play Framework, PostgreSQL / PostGIS | N/A (Web platform) | Active (develop branch) | `APPROVED (REFERENCE ARCHITECTURE)` | Consult for label ontology, spatial schema, and crowdsourcing verification flow. No code copying. |
| **GC-DATA-01** (GuideCane Taxonomy & Data) | `yolo0731/GuideCaneOD` | CC BY-NC 4.0 | YAML, COCO/YOLO annotations, 17 classes | N/A (Data/Ontology) | Maintained (v1.0 2025) | `APPROVED WITH RESTRICTIONS (RESEARCH ONLY)` | Map 17 classes to AccessGuard ontology. Non-commercial restriction must be explicitly noted. |
| **GC-CODE-01** (GuideCane Training Repo) | `yolo0731/GuideCaneOD-YOLO` | **AGPL v3** | Ultralytics YOLOv8/v11 wrapper, Python 3.10+ | **Light** (~0.9GB FP16 for YOLOv8s). | Maintained (2025) | **`REJECTED (CODE COPY PROHIBITED)`** | **AGPL v3 viral copyleft.** Do NOT import or vendor any code. Implement clean-room adapter using native Ultralytics YOLO API. |
| **CS-SEG-01** (CitySurfaces Segmenter) | `VIDA-NYU/city-surfaces` | BSD 3-Clause | HRNet-W48 / OCRNet (PyTorch 1.7 / 2.x, NVIDIA seg) | **Heavy** (~2.8GB FP32). High memory footprint at $1024 \times 2048$. | Archived / Research (2022) | `APPROVED FOR PROTOTYPE` | Permissive BSD license. Map 8 paving materials to roughness metrics. Run with FP16 or CPU fallback on 4GB VRAM. |

---

## 3. Verdict Definitions

- `APPROVED FOR PROTOTYPE`: The component has a permissive license (MIT, Apache 2.0, BSD), directly addresses an AccessGuard perception capability, passes security audit, and can be integrated cleanly into AccessGuard's `ModelAdapter` interface with appropriate hardware safeguards.
- `APPROVED WITH RESTRICTIONS`: The component provides valuable taxonomy or benchmarking datasets, but carries non-commercial or specific operational limitations (e.g. CC BY-NC 4.0). Can be used for research, evaluation, and non-commercial validation, but cannot be bundled as unencumbered commercial software.
- `APPROVED (REFERENCE ARCHITECTURE)`: Upstream implementation patterns, database schemas, calibration tables, or API contracts are studied and adapted into AccessGuard's native FastAPI / PostGIS architecture without vendoring external codebases.
- `REJECTED`: The component poses unacceptable licensing hazards (e.g. AGPL v3 copyleft virality), malicious dependencies, or extreme technical debt that conflicts with AccessGuard's architectural guidelines.

---

## 4. Hardware Sizing & Memory Budget (RTX 3050 4GB VRAM)

Running multi-model hybrid inference on a single 4GB GPU requires strict memory budgeting:

```
+-----------------------------------------------------------------------------+
|                     RTX 3050 LAPTOP GPU (4096 MB VRAM BUDGET)               |
+-----------------------------------------------------------------------------+
| PyTorch CUDA Context & Base Overhead:                    ~450 MB            |
| Free VRAM Available for Models & Tensors:               ~3640 MB            |
+-----------------------------------------------------------------------------+
| Concurrent Execution Risk:                                                  |
|   - ConvNeXt-V2 (RampNet):                              ~1800 MB (FP16)     |
|   - DINOv2-Large (Validator):                           ~2100 MB (FP16)     |
|   - HRNet-W48 (CitySurfaces):                           ~2200 MB (FP16)     |
|   - YOLOv8s (GuideCane / Obstacles):                     ~900 MB (FP16)     |
| Total if all 4 models loaded simultaneously:            ~7000 MB  --> OOM!  |
+-----------------------------------------------------------------------------+
| MANDATED SOLUTION: Dynamic Memory Offloading & Sequential Pipelining        |
|   1. `RESOURCE_CONSTRAINED` mode executes models in sequential phases.      |
|   2. Only ONE heavy backbone (ConvNeXt, DINOv2, HRNet) in VRAM at a time.   |
|   3. Call `torch.cuda.empty_cache()` and `gc.collect()` between phases.     |
|   4. Provide transparent CPU fallback (`DEVICE = "cpu"`) when VRAM < 1GB.  |
+-----------------------------------------------------------------------------+
```
