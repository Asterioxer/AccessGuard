"""
AccessGuard Multi-Model Evidence Fusion Package.
"""

from app.cv.fusion.spatial_association import (
    SpatialCluster,
    calculate_iou,
    point_near_box,
    cluster_spatial_observations,
)
from app.cv.fusion.evidence_fusion import (
    FusedBarrierEvidence,
    FusedInferenceResult,
    EvidenceFusionEngine,
    MODEL_CLASS_WEIGHTS,
)

__all__ = [
    "SpatialCluster",
    "calculate_iou",
    "point_near_box",
    "cluster_spatial_observations",
    "FusedBarrierEvidence",
    "FusedInferenceResult",
    "EvidenceFusionEngine",
    "MODEL_CLASS_WEIGHTS",
]
