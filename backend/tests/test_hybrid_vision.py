"""
Unit and Integration Tests for AccessGuard Hybrid Vision Intelligence (Mission 006).
Tests lifecycle governance, failure semantics, seam suppression, calibration lookups,
and multi-model evidence fusion.
"""

import numpy as np
import pytest

from app.cv.adapters.base_v2 import (
    AdapterStatus,
    OutputModality,
    PointKeypoint,
    SemanticMask,
    VerificationVerdict,
    AdapterInferenceResult,
)
from app.cv.base import DetectedBarrier
from app.cv.adapters.rampnet_adapter import RampNetAdapter
from app.cv.adapters.guidecane_adapter import GuideCaneAdapter
from app.cv.adapters.citysurfaces_adapter import CitySurfacesAdapter
from app.cv.adapters.sidewalk_validator_adapter import SidewalkValidatorAdapter
from app.cv.adapters.hybrid_adapter import HybridVisionAdapter
from app.cv.fusion.spatial_association import calculate_iou, point_near_box, cluster_spatial_observations
from app.cv.fusion.evidence_fusion import EvidenceFusionEngine


def test_spatial_association_iou():
    boxA = (0.1, 0.1, 0.5, 0.5)
    boxB = (0.1, 0.1, 0.5, 0.5)
    assert calculate_iou(boxA, boxB) == 1.0

    boxC = (0.6, 0.6, 0.9, 0.9)
    assert calculate_iou(boxA, boxC) == 0.0

    boxD = (0.3, 0.3, 0.7, 0.7)
    iou = calculate_iou(boxA, boxD)
    assert 0.0 < iou < 1.0


def test_point_near_box_association():
    kp = PointKeypoint(x=0.3, y=0.3, confidence=0.9, label="curb_ramp_present")
    box = (0.2, 0.2, 0.4, 0.4)
    assert bool(point_near_box(kp, box, margin=0.04)) is True

    far_kp = PointKeypoint(x=0.8, y=0.8, confidence=0.9, label="curb_ramp_present")
    assert bool(point_near_box(far_kp, box, margin=0.04)) is False


def test_rampnet_adapter_lifecycle_and_seam():
    adapter = RampNetAdapter(enable_seam_suppression=True)
    assert adapter.status == AdapterStatus.NOT_LOADED

    loaded = adapter.load(device="cpu")
    assert loaded is True
    assert adapter.status in (AdapterStatus.READY, AdapterStatus.DEGRADED)

    # Test equirectangular seam suppression
    peaks = [
        (100, 10, 0.90),    # Left edge
        (105, 1015, 0.85),  # Right edge duplicate
        (250, 500, 0.92)    # Central ramp
    ]
    suppressed = adapter._suppress_seam_duplicates(peaks, heatmap_width=1024, wrap_margin=40)
    assert len(suppressed) == 2
    assert (105, 1015, 0.85) not in suppressed

    # One-to-one matching verification (ICCV'25 erratum)
    preds = [(0.50, 0.50), (0.51, 0.51)] # Two duplicate detections
    gt = [(0.50, 0.50)]                   # Single ground truth ramp
    metrics = RampNetAdapter.one_to_one_match(preds, gt, radius=0.05)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1 # Duplicate penalised!
    assert metrics["precision"] == 0.50

    unloaded = adapter.unload()
    assert unloaded is True
    assert adapter.status == AdapterStatus.NOT_LOADED


def test_guidecane_adapter_taxonomy_and_lifecycle():
    adapter = GuideCaneAdapter()
    assert adapter.status == AdapterStatus.NOT_LOADED

    # In strict mode without weights, load() must report UNAVAILABLE
    strict_loaded = adapter.load(device="cpu", allow_fallback=False)
    assert strict_loaded is False
    assert adapter.status == AdapterStatus.UNAVAILABLE

    # With allow_fallback=True, load() activates fallback
    loaded = adapter.load(device="cpu", allow_fallback=True)
    assert loaded is True
    assert adapter.status == AdapterStatus.DEGRADED

    # Test inference on blank test canvas
    canvas = np.full((480, 640, 3), 128, dtype=np.uint8)
    res = adapter.predict_v2(canvas, allow_fallback=True)
    assert res.is_successful is True
    assert res.adapter_name == "guidecane_adapter"
    assert res.execution_type == "heuristic_fallback"

    # Verify metadata and CC BY-NC 4.0 licensing attribution
    meta = adapter.metadata()
    assert "CC BY-NC 4.0" in meta.license
    assert OutputModality.BOUNDING_BOX in meta.modalities

    adapter.unload()
    assert adapter.status == AdapterStatus.NOT_LOADED


def test_citysurfaces_adapter_roughness():
    adapter = CitySurfacesAdapter()
    # In strict mode without weights, load() must report UNAVAILABLE
    assert adapter.load(allow_fallback=False) is False
    assert adapter.status == AdapterStatus.UNAVAILABLE

    # In fallback mode
    adapter.load(allow_fallback=True)
    assert adapter.status == AdapterStatus.DEGRADED

    # Synthetic smooth pavement
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    res = adapter.predict_v2(canvas, allow_fallback=True)
    assert res.is_successful is True
    assert res.mask is not None
    assert isinstance(res.mask.roughness_score, float)
    assert res.execution_type == "heuristic_fallback"

    adapter.unload()


def test_sidewalk_validator_calibration_lookup():
    validator = SidewalkValidatorAdapter()
    validator.load(allow_fallback=True)

    # Test calibrated accuracy lookup
    cal_acc = validator.lookup_calibrated_accuracy(
        label_type="curbramp",
        verdict="correct",
        raw_confidence=0.92
    )
    assert 0.50 <= cal_acc <= 1.0

    # Test inference verification
    canvas = np.full((480, 640, 3), 128, dtype=np.uint8)
    res = validator.predict_v2(canvas, candidate_label="curb_ramp_present", allow_fallback=True)
    assert res.is_successful is True
    assert res.verification is not None
    assert isinstance(res.verification.is_valid, bool)

    validator.unload()


def test_evidence_fusion_engine_conflict_and_consensus():
    engine = EvidenceFusionEngine(iou_threshold=0.40)

    # 1. Consensus Scenario: Two adapters agree on curb_ramp_present
    d1 = DetectedBarrier(
        barrier_type="curb_ramp_present",
        confidence=0.85,
        uncertainty_score=0.15,
        bbox=(0.7, 0.4, 0.8, 0.5)
    )
    kp = PointKeypoint(x=0.45, y=0.75, confidence=0.90, label="curb_ramp_present")

    r_yolo = AdapterInferenceResult(
        status=AdapterStatus.READY,
        adapter_name="yolo_adapter",
        detections=[d1]
    )
    r_ramp = AdapterInferenceResult(
        status=AdapterStatus.READY,
        adapter_name="rampnet_adapter",
        keypoints=[kp]
    )

    fused = engine.fuse([r_yolo, r_ramp])
    assert len(fused.fused_barriers) == 1
    barrier = fused.fused_barriers[0]
    assert barrier.consensus_status == "CONSENSUS"
    assert barrier.fused_confidence > 0.85 # Confidence boosted!
    assert barrier.verification_state == "VERIFIED"

    # 2. Contradiction Scenario: RampNet says ramp, but YOLO says stairs
    d_stairs = DetectedBarrier(
        barrier_type="stairs",
        confidence=0.80,
        uncertainty_score=0.20,
        bbox=(0.7, 0.4, 0.8, 0.5)
    )
    r_stairs = AdapterInferenceResult(
        status=AdapterStatus.READY,
        adapter_name="yolo_adapter",
        detections=[d_stairs]
    )

    fused_conflict = engine.fuse([r_stairs, r_ramp])
    assert len(fused_conflict.fused_barriers) == 1
    conflict_barrier = fused_conflict.fused_barriers[0]
    assert conflict_barrier.consensus_status == "DISAGREEMENT"
    assert conflict_barrier.uncertainty_score >= 0.60 # Uncertainty penalized!


def test_strict_failure_semantics_no_empty_masking():
    """Confirms that uninitialized or failing adapters never return empty list masked as success."""
    adapter = RampNetAdapter()
    # Explicitly do NOT load adapter
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # When load fails or is prevented
    adapter._status = AdapterStatus.NOT_LOADED
    # Mock failing load
    adapter.load = lambda *args, **kw: False

    res = adapter.predict_v2(canvas)
    assert res.is_successful is False
    assert res.status in (AdapterStatus.INFERENCE_FAILED, AdapterStatus.NOT_LOADED, AdapterStatus.UNAVAILABLE)
    assert res.error_message is not None
