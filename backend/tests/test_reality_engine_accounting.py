"""
Tests for Reality Engine Execution Integrity Accounting and Truthful Segmentation Status.
Verifies Defects C and D remediation:
1. scenes_processed_successfully vs scenes_processing_failed are accurately counted.
2. execution_integrity separates model attempts, successes, failures, unavailable, fallback, and zero detections.
3. segmentation_metrics.status is derived from actual segmentation execution records, never from IoU comparison.
4. Genuine segmentation model yielding zero masks reports READY with mean_iou=0.0.
5. Missing segmentation weights reports UNAVAILABLE regardless of ground-truth presence.
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from app.cv.evaluation.reality_engine import RealityEngine
from app.cv.adapters.base_v2 import ExecutionPolicy
from app.cv.orchestration.image_processing_service import (
    SegmentationResultRecord,
    ImageProcessingResult,
    ModelExecutionRecord
)


def test_reality_engine_accounting_fields_exist(tmp_path):
    """Verify execution_integrity contains separated, non-conflated metrics."""
    engine = RealityEngine(execution_policy=ExecutionPolicy.STRICT_GENUINE)
    out_file = tmp_path / "accounting_scorecard.json"

    # Mock process_image to return deterministic ImageProcessingResult
    mock_seg = SegmentationResultRecord(
        status="UNAVAILABLE",
        model_identifier="yolo_segmentation_adapter",
        model_version="1.1.0",
        execution_type="none",
        regions_count=0,
        regions=[]
    )
    mock_model_record = ModelExecutionRecord(
        adapter_name="yolo_adapter",
        model_family="YOLOv8n",
        upstream_repo="ultralytics",
        checkpoint_path="data/models/yolov8n.pt",
        weight_sha256="abc12345",
        model_architecture="YOLOv8",
        model_version="1.2.0",
        class_mapping_version="v1.0",
        device_used="cpu",
        execution_type="model_inference",
        execution_time_ms=12.5,
        input_dimensions=(640, 640),
        output_count=0,
        status="READY"
    )
    mock_res = ImageProcessingResult(
        run_id="run_test",
        image_id="img_test",
        source="benchmark",
        source_metadata={},
        original_sha256="sha1",
        sanitized_sha256="sha2",
        processing_timestamp="2026-11-19T00:00:00Z",
        sanitized_image_path=None,
        latitude=None,
        longitude=None,
        model_execution_records=[mock_model_record],
        detections=[],
        segmentation_results=mock_seg,
        fused_evidence=[],
        uncertainty_summary={},
        resource_telemetry={},
        warnings=[],
        errors=[],
        overall_status="SUCCESS"
    )

    with patch.object(engine.proc_service, "process_image", return_value=mock_res):
        scorecard = engine.run_reality_evaluation(
            dataset_dir="data/benchmarks/indian_streetscapes",
            output_scorecard_path=str(out_file)
        )

    integ = scorecard["execution_integrity"]
    assert integ["total_scenes"] == 25
    assert integ["scenes_processed_successfully"] == 25
    assert integ["scenes_processing_failed"] == 0
    assert integ["model_inference_attempts"] == 25
    assert integ["model_inference_successes"] == 25
    assert integ["model_inference_failures"] == 0
    assert integ["zero_detection_successes"] == 25
    assert integ["unavailable_models"] == 0
    assert integ["fallback_executions"] == 0

    # Segmentation status must be UNAVAILABLE based on mock_seg, not READY due to IoU logic
    seg = scorecard["segmentation_metrics"]
    assert seg["status"] == "UNAVAILABLE"
    assert seg["execution_mode"] == "none"
    assert seg["mean_iou"] == 0.0


def test_reality_engine_segmentation_status_ready_zero_masks(tmp_path):
    """When segmentation model is READY but outputs zero masks, status must remain READY."""
    engine = RealityEngine(execution_policy=ExecutionPolicy.STRICT_GENUINE)
    out_file = tmp_path / "ready_zero_masks_scorecard.json"

    mock_seg_ready = SegmentationResultRecord(
        status="READY",
        model_identifier="yolo_segmentation_adapter",
        model_version="1.1.0",
        execution_type="model_inference",
        regions_count=0,
        regions=[]
    )
    mock_res = ImageProcessingResult(
        run_id="run_test",
        image_id="img_test",
        source="benchmark",
        source_metadata={},
        original_sha256="sha1",
        sanitized_sha256="sha2",
        processing_timestamp="2026-11-19T00:00:00Z",
        sanitized_image_path=None,
        latitude=None,
        longitude=None,
        model_execution_records=[],
        detections=[],
        segmentation_results=mock_seg_ready,
        fused_evidence=[],
        uncertainty_summary={},
        resource_telemetry={},
        warnings=[],
        errors=[],
        overall_status="SUCCESS"
    )

    with patch.object(engine.proc_service, "process_image", return_value=mock_res):
        scorecard = engine.run_reality_evaluation(
            dataset_dir="data/benchmarks/indian_streetscapes",
            output_scorecard_path=str(out_file)
        )

    seg = scorecard["segmentation_metrics"]
    assert seg["status"] == "READY"
    assert seg["execution_mode"] == "genuine_neural_model"
    assert seg["mean_iou"] == 0.0
    assert seg["total_regions_segmented"] == 0


def test_reality_engine_corrupt_scene_accounting(tmp_path):
    """A scene that fails image processing must increment scenes_processing_failed."""
    engine = RealityEngine(execution_policy=ExecutionPolicy.STRICT_GENUINE)
    out_file = tmp_path / "failure_accounting_scorecard.json"

    # Simulate process_image returning FAILED for all scenes
    mock_res_failed = ImageProcessingResult(
        run_id="run_test",
        image_id="img_test",
        source="benchmark",
        source_metadata={},
        original_sha256="sha1",
        sanitized_sha256="sha2",
        processing_timestamp="2026-11-19T00:00:00Z",
        sanitized_image_path=None,
        latitude=None,
        longitude=None,
        model_execution_records=[],
        detections=[],
        segmentation_results=SegmentationResultRecord(
            status="UNAVAILABLE", model_identifier="seg", model_version="1.0", execution_type="none"
        ),
        fused_evidence=[],
        uncertainty_summary={},
        resource_telemetry={},
        warnings=[],
        errors=["Critical pipeline error"],
        overall_status="FAILED"
    )

    with patch.object(engine.proc_service, "process_image", return_value=mock_res_failed):
        scorecard = engine.run_reality_evaluation(
            dataset_dir="data/benchmarks/indian_streetscapes",
            output_scorecard_path=str(out_file)
        )

    integ = scorecard["execution_integrity"]
    assert integ["total_scenes"] == 25
    assert integ["scenes_processed_successfully"] == 0
    assert integ["scenes_processing_failed"] == 25
    assert scorecard["pipeline_execution_rate"] == 0.0
