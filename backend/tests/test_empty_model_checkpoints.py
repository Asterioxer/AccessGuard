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



def test_video_with_no_decodable_sampled_frames_is_failed(tmp_path, monkeypatch):
    from app.cv.orchestration import image_processing_service as service_module

    video_path = tmp_path / "empty-video.mp4"
    video_path.write_bytes(b"not-empty-but-no-frames")

    class EmptyCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            if prop == service_module.cv2.CAP_PROP_FPS:
                return 30.0
            if prop == service_module.cv2.CAP_PROP_FRAME_COUNT:
                return 0
            return 0.0

        def read(self):
            return False, None

        def release(self):
            return None

    monkeypatch.setattr(service_module.cv2, "VideoCapture", lambda *_args: EmptyCapture())
    result = service_module.ImageProcessingService().process_video(
        video_path=video_path,
        save_media=False
    )

    assert result.overall_status == "FAILED"
    assert result.sampled_frames == 0
    assert any("No decodable video frames" in error for error in result.errors)
