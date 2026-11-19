"""
Deterministic Unit and Integration Tests for Inference Failure Integrity (Mission 006.5).

Covers Scenarios A through L:
- Scenario A: YOLO successful inference with detections
- Scenario B: YOLO successful inference with zero detections
- Scenario C: YOLO inference exception (INFERENCE_FAILED, failed, preserved error, not model_inference)
- Scenario D: ONNX inference exception (truthful failure, no silent substitution)
- Scenario E: Missing checkpoint (UNAVAILABLE, unavailable)
- Scenario F: Explicit development fallback (DEGRADED, heuristic_fallback)
- Scenario G: Resource exhaustion CUDA/CPU OOM (RESOURCE_EXHAUSTED, failed)
- Scenario H: Invalid image input (INFERENCE_FAILED, failed)
- Scenario I: Model execution record consistency and impossible combination rejection
- Scenario J: End-to-end service integrity (forced failure never claims model_inference)
- Scenario K: Zero-detection end-to-end behavior distinguishable from failed inference
- Scenario L: Full contract validation across adapters
"""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.cv.adapters.base_v2 import AdapterInferenceResult, AdapterStatus, ExecutionPolicy
from app.cv.adapters.yolo_adapter import YOLOAdapter
from app.cv.adapters.onnx_adapter import ONNXAdapter
from app.cv.adapters.sidewalk_validator_adapter import SidewalkValidatorAdapter
from app.cv.orchestration.image_processing_service import (
    ImageProcessingService,
    ImageProcessingRequest,
    ModelExecutionRecord
)
from app.cv.base import DetectedBarrier


@pytest.fixture
def dummy_valid_bgr():
    """Generates a valid 256x256 BGR image array."""
    return np.full((256, 256, 3), 128, dtype=np.uint8)


# =========================================================================
# Scenario A: YOLO successful inference with detections
# =========================================================================
def test_scenario_a_yolo_successful_inference_with_detections(dummy_valid_bgr):
    """Asserts that normal YOLO inference with detections reports READY and model_inference."""
    adapter = YOLOAdapter()
    adapter._status = AdapterStatus.READY
    adapter.is_initialized = True
    adapter.model_path = "data/models/yolov8n.onnx"
    adapter._weight_sha256 = "dummy_sha256_mock"

    mock_onnx = MagicMock()
    mock_barrier = DetectedBarrier(
        barrier_type="stairs",
        confidence=0.88,
        uncertainty_score=0.12,
        bbox=(0.1, 0.1, 0.5, 0.5),
        evidence={"source": "test"}
    )
    mock_onnx.predict.return_value = [mock_barrier]
    adapter.onnx_subadapter = mock_onnx

    res = adapter.predict_v2(dummy_valid_bgr, allow_fallback=False)

    assert res.status == AdapterStatus.READY
    assert res.execution_type == "model_inference"
    assert res.execution_succeeded is True
    assert res.execution_attempted is True
    assert len(res.detections) == 1
    assert res.detections[0].barrier_type == "stairs"
    assert res.error_message is None
    assert res.error_code is None


# =========================================================================
# Scenario B: YOLO successful inference with zero detections
# =========================================================================
def test_scenario_b_yolo_successful_inference_with_zero_detections(dummy_valid_bgr):
    """Asserts that normal YOLO inference with zero detections reports READY and model_inference with 0 outputs."""
    adapter = YOLOAdapter()
    adapter._status = AdapterStatus.READY
    adapter.is_initialized = True
    adapter.model_path = "data/models/yolov8n.onnx"
    adapter._weight_sha256 = "dummy_sha256_mock"

    mock_onnx = MagicMock()
    mock_onnx.predict.return_value = []
    adapter.onnx_subadapter = mock_onnx

    res = adapter.predict_v2(dummy_valid_bgr, allow_fallback=False)

    assert res.status == AdapterStatus.READY
    assert res.execution_type == "model_inference"
    assert res.execution_succeeded is True
    assert res.execution_attempted is True
    assert len(res.detections) == 0
    assert res.error_message is None
    assert res.error_code is None


# =========================================================================
# Scenario C: YOLO inference exception
# =========================================================================
def test_scenario_c_yolo_inference_exception(dummy_valid_bgr):
    """Asserts that an inference exception reports INFERENCE_FAILED and failed, preserving error details."""
    adapter = YOLOAdapter()
    adapter._status = AdapterStatus.READY
    adapter.is_initialized = True
    adapter.model = MagicMock()
    adapter.model.predict.side_effect = RuntimeError("CUDA runtime kernel execution failed")

    res = adapter.predict_v2(dummy_valid_bgr, allow_fallback=False)

    assert res.status == AdapterStatus.INFERENCE_FAILED
    assert res.execution_type == "failed"
    assert res.execution_succeeded is False
    assert res.execution_attempted is True
    assert "CUDA runtime kernel execution failed" in res.error_message
    assert res.error_code == "INFERENCE_FAILED"
    assert len(res.detections) == 0


# =========================================================================
# Scenario D: ONNX inference exception
# =========================================================================
def test_scenario_d_onnx_inference_exception(dummy_valid_bgr):
    """Asserts that an ONNX inference failure propagates truthfully without falling back or claiming success."""
    adapter = YOLOAdapter()
    adapter._status = AdapterStatus.READY
    adapter.is_initialized = True
    adapter.model_path = "data/models/yolov8n.onnx"
    adapter._weight_sha256 = "dummy_sha256_mock"

    mock_onnx = MagicMock()
    mock_onnx.predict.side_effect = RuntimeError("OrtSession::Run failed: Tensor dimension mismatch")
    adapter.onnx_subadapter = mock_onnx

    res = adapter.predict_v2(dummy_valid_bgr, allow_fallback=False)

    assert res.status == AdapterStatus.INFERENCE_FAILED
    assert res.execution_type == "failed"
    assert res.execution_succeeded is False
    assert res.execution_attempted is True
    assert "Tensor dimension mismatch" in res.error_message
    assert res.weight_path == "data/models/yolov8n.onnx"
    assert res.weight_sha256 == "dummy_sha256_mock"
    assert len(res.detections) == 0


# =========================================================================
# Scenario E: Missing checkpoint
# =========================================================================
def test_scenario_e_missing_checkpoint(dummy_valid_bgr):
    """Asserts that a missing checkpoint reports UNAVAILABLE and unavailable without model inference."""
    adapter = YOLOAdapter(model_path="data/models/nonexistent_yolo_weights.pt")
    res = adapter.predict_v2(dummy_valid_bgr, allow_fallback=False)

    assert res.status == AdapterStatus.UNAVAILABLE
    assert res.execution_type == "unavailable"
    assert res.execution_succeeded is False
    assert res.execution_attempted is False
    assert "MODEL_WEIGHTS_NOT_FOUND" in res.error_message
    assert len(res.detections) == 0


# =========================================================================
# Scenario F: Explicit development fallback
# =========================================================================
def test_scenario_f_explicit_development_fallback(dummy_valid_bgr):
    """Asserts that an explicit development fallback reports DEGRADED and heuristic_fallback."""
    adapter = YOLOAdapter(model_path="data/models/nonexistent_yolo_weights.pt")
    res = adapter.predict_v2(dummy_valid_bgr, allow_fallback=True)

    assert res.status == AdapterStatus.DEGRADED
    assert res.execution_type == "heuristic_fallback"
    assert res.execution_succeeded is True
    assert res.execution_attempted is True
    assert res.weight_path is None
    assert res.weight_sha256 is None


# =========================================================================
# Scenario G: Resource exhaustion (CUDA / CPU OOM)
# =========================================================================
def test_scenario_g_resource_exhaustion(dummy_valid_bgr):
    """Asserts that GPU or CPU memory exhaustion produces RESOURCE_EXHAUSTED and failed."""
    adapter = YOLOAdapter()
    adapter._status = AdapterStatus.READY
    adapter.is_initialized = True
    adapter.model = MagicMock()
    adapter.model.predict.side_effect = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    res = adapter.predict_v2(dummy_valid_bgr, allow_fallback=False)

    assert res.status == AdapterStatus.RESOURCE_EXHAUSTED
    assert res.execution_type == "failed"
    assert res.execution_succeeded is False
    assert res.execution_attempted is True
    assert res.error_code == "RESOURCE_EXHAUSTED"
    assert "CUDA out of memory" in res.error_message


# =========================================================================
# Scenario H: Invalid image input
# =========================================================================
@pytest.mark.parametrize("invalid_input", [
    None,
    np.array([], dtype=np.uint8),
    np.zeros((0, 0, 3), dtype=np.uint8),
    "not_an_array"
])
def test_scenario_h_invalid_image_input(invalid_input):
    """Asserts that malformed or non-array inputs report INFERENCE_FAILED and failed."""
    adapter = YOLOAdapter()
    adapter._status = AdapterStatus.READY
    adapter.is_initialized = True

    res = adapter.predict_v2(invalid_input, allow_fallback=False)

    assert res.status == AdapterStatus.INFERENCE_FAILED
    assert res.execution_type == "failed"
    assert res.execution_succeeded is False
    assert res.execution_attempted is True
    assert res.error_code == "INVALID_IMAGE_INPUT"
    assert "Invalid image input" in res.error_message


# =========================================================================
# Scenario I: Impossible adapter contract rejection
# =========================================================================
def test_scenario_i_impossible_contract_combinations_rejected():
    """Asserts that impossible combinations of status and execution_type raise ValueError."""
    # 1. UNAVAILABLE + model_inference
    with pytest.raises(ValueError, match="cannot have execution_type='model_inference'"):
        AdapterInferenceResult(
            status=AdapterStatus.UNAVAILABLE,
            adapter_name="test_adapter",
            execution_type="model_inference"
        )

    # 2. INFERENCE_FAILED + model_inference
    with pytest.raises(ValueError, match="cannot have execution_type='model_inference'"):
        AdapterInferenceResult(
            status=AdapterStatus.INFERENCE_FAILED,
            adapter_name="test_adapter",
            execution_type="model_inference"
        )

    # 3. RESOURCE_EXHAUSTED + model_inference
    with pytest.raises(ValueError, match="cannot have execution_type='model_inference'"):
        AdapterInferenceResult(
            status=AdapterStatus.RESOURCE_EXHAUSTED,
            adapter_name="test_adapter",
            execution_type="model_inference"
        )

    # 4. DEGRADED + model_inference when metadata notes fallback
    with pytest.raises(ValueError, match="cannot claim execution_type='model_inference'"):
        AdapterInferenceResult(
            status=AdapterStatus.DEGRADED,
            adapter_name="test_adapter",
            execution_type="model_inference",
            metadata={"is_fallback": True}
        )

    # 5. READY + unavailable with detections
    with pytest.raises(ValueError, match="cannot claim execution_type='unavailable'"):
        AdapterInferenceResult(
            status=AdapterStatus.READY,
            adapter_name="test_adapter",
            execution_type="unavailable",
            detections=[DetectedBarrier(barrier_type="stairs", confidence=0.9, uncertainty_score=0.1, bbox=(0,0,1,1))]
        )


# =========================================================================
# Scenario J: End-to-end service integrity with forced failure
# =========================================================================
def test_scenario_j_e2e_service_integrity_with_forced_failure(dummy_valid_bgr):
    """Asserts that an adapter inference failure in the unified service never falsely reports successful inference."""
    import cv2
    service = ImageProcessingService()

    # Force YOLO prediction to raise an exception
    mock_yolo = MagicMock()
    mock_yolo.name = "yolo_adapter"
    mock_yolo.version = "1.2.0"
    mock_yolo.predict_v2.return_value = AdapterInferenceResult.create_failed(
        adapter_name="yolo_adapter",
        error_message="Simulated hardware tensor core failure",
        status=AdapterStatus.INFERENCE_FAILED
    )

    with patch.object(service.hybrid_adapter, "yolo_adapter", mock_yolo):
        _, enc = cv2.imencode(".jpg", dummy_valid_bgr)
        req = ImageProcessingRequest(
            image_bytes=enc.tobytes(),
            filename="test_forced_failure.jpg",
            requested_adapter="yolo",
            execution_policy=ExecutionPolicy.STRICT_GENUINE,
            save_media=False
        )
        res = service.process_image(req)

        assert res.overall_status == "FAILED"
        assert len(res.model_execution_records) == 1
        record = res.model_execution_records[0]

        assert record.adapter_name == "yolo_adapter"
        assert record.status == "INFERENCE_FAILED"
        assert record.execution_type == "failed"
        assert record.output_count == 0
        assert len(record.errors) >= 1
        assert "Simulated hardware tensor core failure" in record.errors[0]
        assert len(res.detections) == 0


# =========================================================================
# Scenario K: Zero-detection end-to-end behavior
# =========================================================================
def test_scenario_k_zero_detection_distinguishable_from_failure(dummy_valid_bgr):
    """Asserts that a successful zero-detection run reports SUCCESS, READY, and model_inference."""
    import cv2
    service = ImageProcessingService()

    mock_yolo = MagicMock()
    mock_yolo.name = "yolo_adapter"
    mock_yolo.version = "1.2.0"
    mock_yolo.predict_v2.return_value = AdapterInferenceResult(
        status=AdapterStatus.READY,
        adapter_name="yolo_adapter",
        detections=[],
        execution_type="model_inference",
        execution_succeeded=True,
        execution_attempted=True
    )

    with patch.object(service.hybrid_adapter, "yolo_adapter", mock_yolo):
        _, enc = cv2.imencode(".jpg", dummy_valid_bgr)
        req = ImageProcessingRequest(
            image_bytes=enc.tobytes(),
            filename="test_zero_det.jpg",
            requested_adapter="yolo",
            execution_policy=ExecutionPolicy.STRICT_GENUINE,
            save_media=False
        )
        res = service.process_image(req)

        assert res.overall_status == "SUCCESS"
        assert len(res.model_execution_records) == 1
        record = res.model_execution_records[0]

        assert record.adapter_name == "yolo_adapter"
        assert record.status == "READY"
        assert record.execution_type == "model_inference"
        assert record.output_count == 0
        assert len(record.errors) == 0
        assert len(res.detections) == 0


# =========================================================================
# Scenario L: Full contract validation across adapters
# =========================================================================
def test_scenario_l_sidewalk_validator_failure_reporting():
    """Asserts that SidewalkValidatorAdapter properly reports failure without model_inference."""
    val = SidewalkValidatorAdapter(model_path="data/models/nonexistent_validator.pt")
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # 1. Unavailable when weights absent
    res_unavail = val.predict_v2(dummy_img, allow_fallback=False)
    assert res_unavail.status == AdapterStatus.UNAVAILABLE
    assert res_unavail.execution_type == "unavailable"
    assert res_unavail.execution_succeeded is False

    # 2. Inference exception during verification
    val._status = AdapterStatus.READY
    val.model = MagicMock()
    val.model.side_effect = RuntimeError("DINOv2 backbone forward pass crashed")

    with patch.dict("sys.modules", {"torch": MagicMock(), "torchvision": MagicMock(), "torchvision.transforms": MagicMock()}):
        res_fail = val.predict_v2(dummy_img, allow_fallback=False)
        assert res_fail.status == AdapterStatus.INFERENCE_FAILED
        assert res_fail.execution_type == "failed"
        assert res_fail.execution_succeeded is False
        assert "DINOv2 backbone forward pass crashed" in res_fail.error_message
