# Multi-Model Evidence Fusion & Spatial Association Design
**AccessGuard Hybrid Vision Intelligence (Mission 006)**  
**Feature Branch:** `feature/hybrid-vision-intelligence`  
**Date:** November 2026  
**Status:** COMPLETE — FUSION ENGINE DESIGN SPECIFICATION

---

## 1. System Overview & Problem Statement

In single-model architectures, perception errors propagate directly into routing decisions: a false negative leaves a wheelchair user stranded before an unnavigable curb drop-off, while a false positive forces an unnecessary 500-meter detour.

AccessGuard's **Multi-Model Evidence Fusion Engine** combines independent observations from complementary models (YOLO, GuideCane, RampNet, CitySurfaces, and Sidewalk Validator) into a unified, calibrated geospatial evidence representation.

```
+-------------------------------------------------------------------------------------------------------+
|                                  MULTI-MODEL EVIDENCE FUSION PIPELINE                                 |
+-------------------------------------------------------------------------------------------------------+
|                                                                                                       |
|  [ YOLOv8 Adapter ]    [ GuideCane Adapter ]    [ RampNet Adapter ]     [ CitySurfaces Adapter ]      |
|    Bounding Boxes        Micro-Barriers            Peak Keypoints         Roughness / Paving Masks    |
|          |                      |                       |                            |                |
|          +----------------------+-----------------------+----------------------------+                |
|                                             |                                                         |
|                                             v                                                         |
|                     +-----------------------------------------------+                                 |
|                     |        STAGE 1: CANONICAL NORMALIZATION       |                                 |
|                     |   Map raw labels to Canonical Ontology        |                                 |
|                     +-----------------------+-----------------------+                                 |
|                                             |                                                         |
|                                             v                                                         |
|                     +-----------------------------------------------+                                 |
|                     |        STAGE 2: SPATIAL ASSOCIATION           |                                 |
|                     |   - Bounding Box IoU Clustering (>= 0.45)     |                                 |
|                     |   - Point-to-Box Geodesic Gating (r <= 0.06)  |                                 |
|                     |   - Equirectangular Seam Cluster Reconciliation|                                 |
|                     +-----------------------+-----------------------+                                 |
|                                             |                                                         |
|                                             v                                                         |
|                     +-----------------------------------------------+                                 |
|                     |      STAGE 3: DISAGREEMENT & CONFLICT GATING  |                                 |
|                     |   - Detect Contradictions (e.g. Ramp vs Stair)|                                 |
|                     |   - Tag Consensus Status                      |                                 |
|                     +-----------------------+-----------------------+                                 |
|                                             |                                                         |
|                                             v                                                         |
|                     +-----------------------------------------------+                                 |
|                     |      STAGE 4: CALIBRATED PROBABILISTIC FUSION |                                 |
|                     |   - Multi-Source Noisy-OR Confidence Updating |                                 |
|                     |   - Contradiction Uncertainty Penalty         |                                 |
|                     |   - Sidewalk Validator Crop Verification      |                                 |
|                     +-----------------------+-----------------------+                                 |
|                                             |                                                         |
|                                             v                                                         |
|                              [ FusedBarrierEvidence Ledger ]                                          |
|                       - Consolidated Geometry (Box & Keypoints)                                       |
|                       - Canonical Accessibility Category                                              |
|                       - Calibrated Confidence & Uncertainty Score                                     |
|                       - Model Attribution & Consensus State                                           |
|                                                                                                       |
+-------------------------------------------------------------------------------------------------------+
```

---

## 2. Spatial Association Algorithms

### 2.1 Bounding Box to Bounding Box Association (IoU Gating)
For pairs of detections $B_i, B_j$, the Intersection-over-Union is calculated:
$$\text{IoU}(B_i, B_j) = \frac{\text{Area}(B_i \cap B_j)}{\text{Area}(B_i \cup B_j)}$$
- If $\text{IoU}(B_i, B_j) \ge 0.45$, detections are clustered into a shared candidate group.
- The consolidated bounding box is calculated as the confidence-weighted centroid:
  $$\bar{B} = \frac{c_i B_i + c_j B_j}{c_i + c_j}$$

### 2.2 Keypoint to Bounding Box Association
RampNet outputs point coordinates $P = (x_p, y_p)$ while detectors output boxes $B = (y_{\min}, x_{\min}, y_{\max}, x_{\max})$.
- A keypoint is associated with a bounding box if:
  1. $P \in B$ (point falls directly inside the box), OR
  2. $\text{EuclideanDist}(P, \text{Center}(B)) \le r_{\text{gate}} = 0.06$ (normalized image coordinates).
- When a RampNet keypoint matches an obstacle detector's `curb_ramp_present` box, they merge into a high-confidence consensus detection.

---

## 3. Disagreement Detection & Conflict Resolution

### 3.1 Taxonomy Conflict Matrix
When two models detect features in the same spatial cluster, their categories are classified:

| Observation A | Observation B | Relationship | Resolution Rule |
|---|---|---|---|
| `curb_ramp_present` (RampNet) | `curb_ramp_present` (YOLO) | **CONSENSUS** | Reinforce confidence; status = `VERIFIED`. |
| `curb_ramp_present` (RampNet) | `stairs` (YOLO/GuideCane) | **CONTRADICTION** | Trigger safety alert; mark status = `DISAGREEMENT`; inflate uncertainty to 0.65; require human or secondary validator. |
| `pole_or_bollard` (GuideCane) | `construction_obstacle` (YOLO) | **SPECIFICITY GAIN** | GuideCane is more specific; adopt `pole_or_bollard`; combine confidence. |
| `construction_obstacle` (YOLO) | `REJECTED` (Validator) | **OVERRULED** | Validator calibrated accuracy $< 0.85$; demote candidate to `UNVERIFIED_CANDIDATE`. |

---

## 4. Non-Naive Confidence Calibration & Fusion Formulation

### 4.1 Rejection of Naive Averaging
A simple arithmetic average $C_{\text{avg}} = \frac{C_1 + C_2}{2}$ is mathematically flawed:
- If a specialized RampNet detector reports $C_{\text{ramp}} = 0.94$ and a general YOLO detector reports $C_{\text{yolo}} = 0.20$, the naive average is $0.57$, artificially downgrading a genuine curb ramp.
- If two independent detectors both report $C = 0.80$, naive averaging returns $0.80$, completely failing to reflect the increased statistical probability of agreement.

### 4.2 Multi-Source Probabilistic Noisy-OR Fusion
When independent models $M_1, \dots, M_k$ report agreement on the same canonical class, their independent failure probabilities multiply:
$$P(\text{Barrier} \mid M_1, \dots, M_k) = 1 - \prod_{i=1}^k \left(1 - w_i \cdot c_i\right)$$
where $w_i \in (0, 1]$ represents the empirically measured precision weight of model $M_i$ for that specific class.

### 4.3 Contradiction Uncertainty Penalty
When models in the same cluster report conflicting classes (e.g. `curb_ramp_present` vs `stairs`), the uncertainty score is penalized:
$$U_{\text{fused}} = \min\left(0.95, \max(U_1, U_2) + 0.35\right)$$
$$C_{\text{fused}} = \max(0.20, \min(C_1, C_2) \cdot 0.70)$$
This guarantees that unsafe ambiguous intersections are NEVER silently flagged as accessible.

---

## 5. Fused Output Evidence Schema (`FusedBarrierEvidence`)

```python
@dataclass
class FusedBarrierEvidence:
    id: str
    canonical_type: str                  # Canonical AccessGuard category
    fused_confidence: float              # Calibrated combined confidence [0.0, 1.0]
    uncertainty_score: float             # Calibrated uncertainty [0.0, 1.0]
    consensus_status: str                # "CONSENSUS", "PARTIAL", "DISAGREEMENT", "SINGLE_MODEL"
    bbox: Tuple[float, float, float, float] # Consolidated normalized bbox
    keypoints: List[PointKeypoint]       # Associated discrete keypoints
    contributing_adapters: List[str]     # Names of contributing adapters
    verification_state: str              # "VERIFIED", "TENTATIVE", "REJECTED"
    provenance: Dict[str, Any]           # Full raw evidence, individual scores, latencies
```
