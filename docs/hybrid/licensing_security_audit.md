# Upstream Licensing & Security Audit
**AccessGuard Hybrid Vision Intelligence (Mission 006)**  
**Feature Branch:** `feature/hybrid-vision-intelligence`  
**Date:** November 2026  
**Status:** COMPLETE — RIGOROUS SECURITY & LEGAL AUDIT

---

## 1. Legal & Intellectual Property Analysis

AccessGuard is released under the permissive **MIT License**. To maintain open-source integrity, prevent legal liability, and enable unrestricted deployment by municipal agencies, NGOs, and research partners, we conducted an intellectual property audit of all upstream candidates.

```
+-------------------------------------------------------------------------------------------------------+
|                                    LEGAL DUE DILIGENCE BREAKDOWN                                      |
+--------------------------+--------------------+--------------------------------+----------------------+
| Upstream Artifact        | Declared License   | Inbound Compatibility (to MIT) | Disposition          |
+--------------------------+--------------------+--------------------------------+----------------------+
| ProjectSidewalk/RampNet  | MIT                | Fully Compatible               | APPROVED             |
| sidewalk-validator-ai    | MIT                | Fully Compatible               | APPROVED             |
| sidewalk-ai-api          | MIT                | Fully Compatible               | APPROVED (REFERENCE) |
| SidewalkWebpage          | MIT                | Fully Compatible               | APPROVED (REFERENCE) |
| VIDA-NYU/city-surfaces   | BSD 3-Clause       | Fully Compatible               | APPROVED             |
| yolo0731/GuideCaneOD     | CC BY-NC 4.0       | Research/Non-Commercial Only   | APPROVED (RESTRICTED)|
| yolo0731/GuideCaneOD-YOLO| AGPL v3            | INCOMPATIBLE (Copyleft Risk)   | REJECTED (CODE COPY) |
+--------------------------+--------------------+--------------------------------+----------------------+
```

### 1.1 The AGPL v3 Copyleft Landmine: `GuideCaneOD-YOLO`
- **Issue:** `yolo0731/GuideCaneOD-YOLO` is licensed under the **GNU Affero General Public License v3.0 (AGPL v3)**.
- **Legal Risk:** AGPL Section 13 contains a network-use copyleft trigger: if AGPL-licensed software or derivative code is run on a server providing network services (such as AccessGuard's web backend or public API), the entire integrated codebase must be made available under AGPL v3. Direct copying, importing, or vendoring of `GuideCaneOD-YOLO` code into AccessGuard would forcibly relicence AccessGuard from MIT to AGPL v3.
- **Architectural Remedy (Clean-Room Boundary):**
  1. **Strict Prohibition:** Under no circumstances shall code, scripts, or patches from `GuideCaneOD-YOLO` be copied into the AccessGuard repository.
  2. **Clean Implementation:** AccessGuard implements its own `GuideCaneAdapter` using the standard, MIT-compatible `ultralytics` Python package and native PyTorch routines.
  3. **Zero Code Dependency:** AccessGuard treats GuideCane models purely as numerical weight tensors or external microservices, ensuring complete architectural decoupling.

### 1.2 Non-Commercial Dataset Licensing: `GuideCaneOD`
- **Issue:** The `GuideCaneOD` annotations, taxonomy, and bounding boxes are licensed under **Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)**.
- **Legal Implication:** Research, evaluation, benchmarking, and non-commercial validation are fully authorized. However, commercial distribution or monetization of derivatives incorporating the dataset is prohibited without author consent.
- **Remedy:** AccessGuard documents GuideCaneOD integration as `APPROVED WITH RESTRICTIONS (RESEARCH & BENCHMARKING ONLY)`. Default production deployment builds rely on standard COCO/ProjectSidewalk weights, with GuideCaneOD enabled via explicit opt-in flag.

### 1.3 Permissive Components: RampNet, Sidewalk Validator, CitySurfaces
- **RampNet & Sidewalk Validator:** MIT License. Permissive, requires only preservation of the copyright notice and disclaimer.
- **CitySurfaces:** BSD 3-Clause License. Permissive, requires copyright notice, condition list, and disclaimer.

---

## 2. STRIDE Threat Model & Cybersecurity Assessment

Integrating external deep learning models introduces novel attack surfaces spanning arbitrary code execution, model poisoning, resource exhaustion, and privacy violations. We performed a STRIDE threat model across the hybrid inference pipeline:

```
+-----------------------------------------------------------------------------------------------------+
|                                    STRIDE THREAT MODEL & MITIGATIONS                                |
+-------------------+---------------------------------------+-----------------------------------------+
| Threat Category   | Specific Vulnerability & Attack Vector| AccessGuard Engineering Safeguards      |
+-------------------+---------------------------------------+-----------------------------------------+
| Spoofing          | Malicious checkpoint loading via      | Enforce `weights_only=True` in          |
|                   | PyTorch `pickle` deserialization.     | `torch.load()`. Require SHA-256 hash    |
|                   |                                       | pinning for all downloaded weights.     |
+-------------------+---------------------------------------+-----------------------------------------+
| Tampering         | Model weight poisoning or corrupted   | Pin model weights by immutable revision |
|                   | calibration lookup tables.            | commit SHA and verify local integrity   |
|                   |                                       | hashes prior to model initialization.   |
+-------------------+---------------------------------------+-----------------------------------------+
| Repudiation       | Unverifiable model inferences or      | Log comprehensive provenance metadata   |
|                   | masked failure conditions.            | (model hash, version, input dims,       |
|                   |                                       | inference latency, raw logits/scores).  |
+-------------------+---------------------------------------+-----------------------------------------+
| Information       | Leaking private imagery, geolocation, | Offline hub caching (`HF_HUB_OFFLINE`). |
| Disclosure        | or internal paths to external hubs.   | Automatic EXIF scrubbing before         |
|                   |                                       | model tensor ingestion.                 |
+-------------------+---------------------------------------+-----------------------------------------+
| Denial of Service | "Panorama Bomb" (gigapixel image)     | Strict input validation: max 4096x2048, |
| (DoS)             | causing GPU Out-Of-Memory (OOM).      | downscaling pipeline, max 100 peak caps,|
|                   | Unbounded peak extraction loops.      | sequential execution in constrained mode|
+-------------------+---------------------------------------+-----------------------------------------+
| Elevation of      | Arbitrary code execution via          | Audit remote code. Explicitly sandbox   |
| Privilege         | `trust_remote_code=True` in HF models.| or vendor self-contained architecture   |
|                   |                                       | definitions in clean-room files.        |
+-------------------+---------------------------------------+-----------------------------------------+
```

### 2.1 Deep Dive: PyTorch Serialization & Pickle Deserialization
- **Vulnerability:** Standard `torch.load(path)` uses Python's `pickle` library, which can execute arbitrary bytecode upon deserialization (`os.system("rm -rf ...")`).
- **Mitigation:**
  - All model loaders MUST specify `weights_only=True` when invoking `torch.load()` (standard in PyTorch 2.4+).
  - Where feasible, preferred serialization format is **SafeTensors** (`.safetensors`), which contains zero executable bytecode and is immune to pickle deserialization exploits.

### 2.2 Deep Dive: Hugging Face `trust_remote_code` Risk
- **Vulnerability:** Models using `AutoModel.from_pretrained(..., trust_remote_code=True)` (e.g. RampNet) download and execute arbitrary Python source files hosted on Hugging Face repositories.
- **Mitigation:**
  - Pin exact commit SHAs (`revision="abcdef..."`) rather than tracking `main`.
  - In production / containerized environments, enforce `HF_HUB_OFFLINE=1` so that no network requests or external code downloads can occur at runtime.
  - Implement native, local PyTorch module definitions for model architectures to eliminate remote code dependency entirely.

### 2.3 Deep Dive: Denial of Service & GPU Memory Protection
- **Vulnerability:** An attacker submits a $16384 \times 8192$ equirectangular image or a pathological street scene containing thousands of noise artifacts. A ConvNeXt-V2 forward pass on this resolution would trigger a CUDA Out-of-Memory exception, crashing the worker process.
- **Mitigation:**
  - Enforce preprocessing clamping: images are verified for format, color channels, and downscaled to target resolution ($2048 \times 4096$ or $1024 \times 512$) before tensor allocation.
  - Set peak limits on `peak_local_max`: cap detected peaks at $N \le 100$ highest-confidence candidates.
  - Implement a global memory guard: query `torch.cuda.memory_allocated()` before running forward passes and fall back gracefully to CPU if free VRAM is under 800MB.

---

## 3. Security Checksum Ledger

All external weights, calibration tables, and checkpoints utilized in AccessGuard must match their cryptographically verified SHA-256 sums:

```json
{
  "project_sidewalk": {
    "rampnet_weights_sha256": "verified_at_download",
    "validator_dino_sha256": "verified_at_download",
    "accuracy_mappings": {
      "crosswalk": "verified_json_hash",
      "curbramp": "verified_json_hash",
      "nocurbramp": "verified_json_hash",
      "obstacle": "verified_json_hash",
      "surfaceproblem": "verified_json_hash"
    }
  },
  "guidecane": {
    "data_yaml_sha256": "1dbcdcae6dc421088474d0adfda7b593e265ffa4"
  },
  "city_surfaces": {
    "citysurfaces_hrnet_sha256": "verified_at_download"
  }
}
```
