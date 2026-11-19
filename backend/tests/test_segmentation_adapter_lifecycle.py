"""
Deterministic Tests for Segmentation Adapter Lifecycle and Error Isolation.
Verifies Defect E remediation:
1. self.seg_adapter.load() is invoked before model availability check.
2. Initialization exceptions are captured and recorded as INITIALIZATION_ERROR.
3. Inference exceptions during predict_masks are caught and recorded as INFERENCE_FAILED or RESOURCE_EXHAUSTED.
4. Fallback vs strict unavailable routing is strictly preserved.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from app.cv.orchestration.image_processing_service import ImageProcessingService
from app.cv.segmentation.base import SegmentationAdapter, SegmentedRegion


class DummySegAdapter(SegmentationAdapter):
    """Test stub for segmentation lifecycle tests."""
    def __init__(self, load_raises=False, predict_raises=None, has_weights=True):
        super().__init__(name="dummy_seg_adapter", version="1.0.0")
        self.load_raises = load_raises
        self.predict_raises = predict_raises
        self.has_weights = has_weights
        self.load_called = False
        self.model = None

    def load(self) -> bool:
        self.load_called = True
        if self.load_raises:
            raise RuntimeError("Corrupt segmentation model weights header")
        if self.has_weights:
            self.model = "mock_model_handle"
            self.is_loaded = True
            return True
        self.is_loaded = False
        return False

    def predict_masks(self, image_bgr: np.ndarray):
        if self.predict_raises:
            raise self.predict_raises
        return [
            SegmentedRegion(
                class_id=1,
                class_name="stairs",
                polygon=[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
                confidence=0.88,
                bbox=(0.1, 0.1, 0.5, 0.5),
                area_pixels=1600,
                area_ratio=0.16,
                model_identifier="dummy_seg_adapter",
                model_version="1.0.0"
            )
        ]


def test_segmentation_load_called_and_success():
    """_execute_segmentation must call load() and return READY on genuine inference."""
    adapter = DummySegAdapter(has_weights=True)
    service = ImageProcessingService(segmentation_adapter=adapter)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    rec = service._execute_segmentation(img, allow_fallback=False)
    assert adapter.load_called is True
    assert rec.status == "READY"
    assert rec.execution_type == "model_inference"
    assert rec.regions_count == 1


def test_segmentation_initialization_exception():
    """Initialization exceptions in load() must be caught and reported as INITIALIZATION_ERROR."""
    adapter = DummySegAdapter(load_raises=True)
    service = ImageProcessingService(segmentation_adapter=adapter)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    rec = service._execute_segmentation(img, allow_fallback=False)
    assert adapter.load_called is True
    assert rec.status == "UNAVAILABLE"
    assert rec.execution_type == "none"
    assert "INITIALIZATION_ERROR" in rec.error_message
    assert "Corrupt segmentation model weights" in rec.error_message


def test_segmentation_inference_oom_exception():
    """CUDA OOM during predict_masks must be caught and reported as RESOURCE_EXHAUSTED."""
    adapter = DummySegAdapter(predict_raises=RuntimeError("CUDA out of memory in segmentation decoder"))
    service = ImageProcessingService(segmentation_adapter=adapter)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    rec = service._execute_segmentation(img, allow_fallback=False)
    assert rec.status == "RESOURCE_EXHAUSTED"
    assert rec.execution_type == "failed"
    assert "Segmentation inference exception" in rec.error_message


def test_segmentation_inference_general_exception():
    """General runtime errors during predict_masks must be reported as INFERENCE_FAILED."""
    adapter = DummySegAdapter(predict_raises=ValueError("Invalid tensor shape in mask head"))
    service = ImageProcessingService(segmentation_adapter=adapter)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    rec = service._execute_segmentation(img, allow_fallback=False)
    assert rec.status == "INFERENCE_FAILED"
    assert rec.execution_type == "failed"
    assert "Invalid tensor shape" in rec.error_message


def test_segmentation_missing_weights_fallback():
    """Missing weights with allow_fallback=True must return DEGRADED."""
    adapter = DummySegAdapter(has_weights=False)
    service = ImageProcessingService(segmentation_adapter=adapter)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    rec = service._execute_segmentation(img, allow_fallback=True)
    assert rec.status == "DEGRADED"
    assert rec.execution_type == "heuristic_overlay"
