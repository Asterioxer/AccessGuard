import numpy as np

from app.cv.adapters.base_v2 import AdapterStatus
from app.cv.adapters.yolo_adapter import YOLOAdapter
from app.cv.segmentation.yolo_seg_adapter import YOLOSegmentationAdapter


def test_empty_yolo_checkpoint_is_unavailable(tmp_path):
    checkpoint = tmp_path / "empty-yolo.pt"
    checkpoint.write_bytes(b"")

    adapter = YOLOAdapter(model_path=str(checkpoint))
    result = adapter.predict_v2(np.zeros((64, 64, 3), dtype=np.uint8), allow_fallback=False)

    assert result.status == AdapterStatus.UNAVAILABLE
    assert result.execution_type == "unavailable"
    assert result.execution_succeeded is False


def test_empty_yolo_segmentation_checkpoint_is_unavailable(tmp_path):
    checkpoint = tmp_path / "empty-yolo-seg.pt"
    checkpoint.write_bytes(b"")

    adapter = YOLOSegmentationAdapter(model_path=str(checkpoint))

    assert adapter.load() is False
    assert adapter.is_loaded is False
    # Missing weights must not trigger heuristic contour generation.
    from unittest.mock import Mock
    adapter._fallback.predict_masks = Mock(side_effect=AssertionError("heuristic fallback must not run"))
    assert adapter.predict_masks(np.zeros((64, 64, 3), dtype=np.uint8)) == []
    adapter._fallback.predict_masks.assert_not_called()
