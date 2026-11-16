"""
Multi-Model Evidence Fusion Engine for AccessGuard.
Reconciles heterogeneous outputs across YOLO, GuideCane, RampNet, CitySurfaces, and Sidewalk Validator.
Performs spatial clustering, conflict resolution, probabilistic confidence fusion, and verification state tagging.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import numpy as np

from app.cv.base import DetectedBarrier
from app.cv.adapters.base_v2 import (
    AdapterInferenceResult,
    AdapterStatus,
    PointKeypoint,
    SemanticMask,
    VerificationVerdict,
)
from app.cv.fusion.spatial_association import (
    SpatialCluster,
    cluster_spatial_observations,
)
from app.core.logging import logger

# Model reliability weights per class [0.0 to 1.0]
MODEL_CLASS_WEIGHTS = {
    "rampnet_adapter": {"curb_ramp_present": 0.95},
    "guidecane_adapter": {
        "pole_or_bollard": 0.90,
        "tactile_paving_present": 0.92,
        "stairs": 0.88,
        "construction_obstacle": 0.85
    },
    "yolo_adapter": {
        "construction_obstacle": 0.82,
        "stairs": 0.80,
        "pole_or_bollard": 0.78,
        "curb_ramp_present": 0.70
    },
    "torchvision_adapter": {
        "construction_obstacle": 0.75,
        "stairs": 0.72
    }
}


@dataclass
class FusedBarrierEvidence:
    """Consolidated multi-model accessibility evidence object."""
    id: str
    canonical_type: str
    fused_confidence: float              # [0.0, 1.0]
    uncertainty_score: float             # [0.0, 1.0]
    consensus_status: str                # "CONSENSUS", "PARTIAL_CONSENSUS", "DISAGREEMENT", "SINGLE_MODEL"
    bbox: Tuple[float, float, float, float]
    keypoints: List[PointKeypoint] = field(default_factory=list)
    contributing_adapters: List[str] = field(default_factory=list)
    verification_state: str = "TENTATIVE" # "VERIFIED", "TENTATIVE", "REJECTED"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_detected_barrier(self) -> DetectedBarrier:
        """Converts to standard DetectedBarrier for backward compatibility with routing engine."""
        return DetectedBarrier(
            barrier_type=self.canonical_type,
            confidence=round(self.fused_confidence, 4),
            uncertainty_score=round(self.uncertainty_score, 4),
            bbox=self.bbox,
            evidence={
                **self.provenance,
                "fused_id": self.id,
                "consensus_status": self.consensus_status,
                "verification_state": self.verification_state,
                "contributing_adapters": self.contributing_adapters
            }
        )


@dataclass
class FusedInferenceResult:
    """Complete output of the Evidence Fusion Engine."""
    fused_barriers: List[FusedBarrierEvidence]
    surface_roughness: float = 0.0
    surface_mask: Optional[SemanticMask] = None
    adapter_statuses: Dict[str, AdapterStatus] = field(default_factory=dict)
    total_execution_time_ms: float = 0.0
    is_successful: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvidenceFusionEngine:
    """
    Coordinates multi-model inference aggregation and calibrated evidence synthesis.
    """

    def __init__(
        self,
        iou_threshold: float = 0.40,
        consensus_boost: float = 0.15,
        conflict_penalty: float = 0.35,
        verification_acceptance_threshold: float = 0.85
    ):
        self.iou_threshold = iou_threshold
        self.consensus_boost = consensus_boost
        self.conflict_penalty = conflict_penalty
        self.verification_threshold = verification_acceptance_threshold

    def fuse(
        self,
        adapter_results: List[AdapterInferenceResult],
        validator_verdict: Optional[VerificationVerdict] = None
    ) -> FusedInferenceResult:
        """
        Merges results from multiple adapters into a coherent set of FusedBarrierEvidence.
        """
        adapter_statuses = {res.adapter_name: res.status for res in adapter_results}
        total_time = sum(res.execution_time_ms for res in adapter_results)

        # 1. Collect all detections and keypoints tagged with adapter names
        detections_by_adapter: List[Tuple[str, DetectedBarrier]] = []
        keypoints_by_adapter: List[Tuple[str, PointKeypoint]] = []
        surface_mask: Optional[SemanticMask] = None
        surface_roughness = 0.0

        for res in adapter_results:
            if not res.is_successful:
                logger.warning(f"Adapter {res.adapter_name} failed: {res.error_message}; skipping from fusion pool.")
                continue

            for d in res.detections:
                detections_by_adapter.append((res.adapter_name, d))

            for kp in res.keypoints:
                keypoints_by_adapter.append((res.adapter_name, kp))

            if res.mask is not None:
                surface_mask = res.mask
                surface_roughness = res.mask.roughness_score

        # 2. Cluster observations spatially
        clusters = cluster_spatial_observations(
            detections_by_adapter=detections_by_adapter,
            keypoints_by_adapter=keypoints_by_adapter,
            iou_threshold=self.iou_threshold
        )

        fused_barriers: List[FusedBarrierEvidence] = []

        # 3. Process each spatial cluster
        for cluster in clusters:
            fused_evidence = self._reconcile_cluster(cluster)
            
            # Apply validator verdict if available for this label type
            if validator_verdict is not None and fused_evidence.canonical_type in validator_verdict.evidence.get("original_label", ""):
                if validator_verdict.is_valid:
                    fused_evidence.verification_state = "VERIFIED"
                    fused_evidence.fused_confidence = max(fused_evidence.fused_confidence, validator_verdict.calibrated_accuracy)
                    fused_evidence.uncertainty_score = min(fused_evidence.uncertainty_score, 0.15)
                else:
                    fused_evidence.verification_state = "REJECTED"
                    fused_evidence.fused_confidence = round(fused_evidence.fused_confidence * 0.40, 4)
                    fused_evidence.uncertainty_score = 0.75

            fused_barriers.append(fused_evidence)

        # 4. If surface roughness indicates severe damage, ensure a surface barrier exists
        if surface_roughness > 0.65 and not any(b.canonical_type == "surface_damage" for b in fused_barriers):
            fused_barriers.append(
                FusedBarrierEvidence(
                    id=str(uuid.uuid4())[:8],
                    canonical_type="surface_damage",
                    fused_confidence=round(min(0.95, surface_roughness), 4),
                    uncertainty_score=0.15,
                    consensus_status="SINGLE_MODEL",
                    bbox=(0.60, 0.20, 0.95, 0.80),
                    contributing_adapters=["citysurfaces_adapter"],
                    verification_state="VERIFIED",
                    provenance={"roughness_score": surface_roughness, "source": "CitySurfaces"}
                )
            )

        return FusedInferenceResult(
            fused_barriers=fused_barriers,
            surface_roughness=surface_roughness,
            surface_mask=surface_mask,
            adapter_statuses=adapter_statuses,
            total_execution_time_ms=round(total_time, 2),
            is_successful=True,
            metadata={
                "cluster_count": len(clusters),
                "total_fused_barriers": len(fused_barriers)
            }
        )

    def _reconcile_cluster(self, cluster: SpatialCluster) -> FusedBarrierEvidence:
        """
        Reconciles observations inside a single spatial cluster.
        Detects agreement, contradictions, and computes noisy-OR fused confidence.
        """
        contributing_adapters = list(set(
            [a for a, _ in cluster.detections] + [a for a, _ in cluster.keypoints]
        ))

        # Collect candidate types and confidences
        candidates: List[Tuple[str, float, float, str]] = [] # (canonical_type, conf, uncertainty, adapter)

        for adapter_name, d in cluster.detections:
            candidates.append((d.barrier_type, d.confidence, d.uncertainty_score, adapter_name))

        for adapter_name, kp in cluster.keypoints:
            candidates.append((kp.label, kp.confidence, kp.uncertainty_score, adapter_name))

        if not candidates:
            # Fallback empty
            return FusedBarrierEvidence(
                id=str(uuid.uuid4())[:8],
                canonical_type="unknown",
                fused_confidence=0.0,
                uncertainty_score=1.0,
                consensus_status="SINGLE_MODEL",
                bbox=cluster.consolidated_bbox
            )

        # Check for contradictions: curb_ramp_present vs stairs
        types_present = set(c[0] for c in candidates)
        has_contradiction = ("curb_ramp_present" in types_present and "stairs" in types_present)

        if has_contradiction:
            # Significant disagreement detected!
            fused_type = "stairs" # Conservative safety-first heuristic: assume barrier until verified
            fused_conf = 0.50
            fused_uncertainty = 0.70
            consensus_status = "DISAGREEMENT"
            verification_state = "TENTATIVE"
        elif len(types_present) == 1:
            fused_type = list(types_present)[0]
            # Agreement or single model
            if len(contributing_adapters) > 1:
                consensus_status = "CONSENSUS"
                verification_state = "VERIFIED"
                # Multi-source Noisy-OR fusion: 1 - prod(1 - w*c)
                fail_prob = 1.0
                for c_type, conf, _, adapter in candidates:
                    weight = MODEL_CLASS_WEIGHTS.get(adapter, {}).get(c_type, 0.80)
                    fail_prob *= (1.0 - (weight * conf))
                fused_conf = round(min(0.98, max(0.20, 1.0 - fail_prob)), 4)
                fused_uncertainty = round(max(0.05, min(c[2] for c in candidates) - 0.10), 4)
            else:
                consensus_status = "SINGLE_MODEL"
                verification_state = "TENTATIVE"
                fused_conf = round(candidates[0][1], 4)
                fused_uncertainty = round(candidates[0][2], 4)
        else:
            # Partial consensus: pick dominant or highest confidence type
            consensus_status = "PARTIAL_CONSENSUS"
            verification_state = "TENTATIVE"
            # Sort by confidence
            best_candidate = max(candidates, key=lambda c: c[1])
            fused_type = best_candidate[0]
            fused_conf = round(best_candidate[1], 4)
            fused_uncertainty = round(best_candidate[2] + 0.10, 4)

        return FusedBarrierEvidence(
            id=str(uuid.uuid4())[:8],
            canonical_type=fused_type,
            fused_confidence=fused_conf,
            uncertainty_score=fused_uncertainty,
            consensus_status=consensus_status,
            bbox=cluster.consolidated_bbox,
            keypoints=[kp for _, kp in cluster.keypoints],
            contributing_adapters=contributing_adapters,
            verification_state=verification_state,
            provenance={
                "candidate_count": len(candidates),
                "raw_candidates": [
                    {"type": c[0], "conf": c[1], "adapter": c[3]} for c in candidates
                ],
                "has_contradiction": has_contradiction
            }
        )
