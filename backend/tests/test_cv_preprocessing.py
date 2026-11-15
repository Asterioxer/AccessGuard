import pytest
import numpy as np
import cv2
from app.cv.pipeline import CVPipeline

def test_clahe_contrast_enhancement():
    pipeline = CVPipeline(adapter_name="dummy")
    
    # Generate low contrast / foggy image (std < 22)
    fog_img = np.full((480, 640, 3), 140, dtype=np.uint8)
    for y in [240, 280, 320, 360, 400]:
        cv2.line(fog_img, (100, y), (540, y), (130, 130, 130), 3)

    enhanced, meta = pipeline.preprocess_image(fog_img)
    assert meta["is_low_contrast"] is True
    assert meta["clahe_applied"] is True
    # The enhanced image should have higher standard deviation than the original
    assert np.std(enhanced) > np.std(fog_img)

def test_blur_detection_penalty():
    pipeline = CVPipeline(adapter_name="dummy")
    
    # Create smooth image with almost zero high-frequency edges (Laplacian var < 80)
    blurred_img = np.full((480, 640, 3), 150, dtype=np.uint8)
    # Apply heavy blur
    blurred_img = cv2.GaussianBlur(blurred_img, (51, 51), 20)

    _, meta = pipeline.preprocess_image(blurred_img)
    assert meta["is_blurred"] is True

    # Test that pipeline process_image applies uncertainty penalty when blurred
    _, enc = cv2.imencode(".jpg", blurred_img)
    detections, _ = pipeline.process_image(enc.tobytes())
    assert len(detections) > 0
    # Every detection should carry blur warning in evidence
    assert any("blur" in d.evidence.get("blur_warning", "").lower() for d in detections)
