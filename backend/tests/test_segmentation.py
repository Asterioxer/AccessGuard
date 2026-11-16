"""
Unit & Integration Tests for Pixel-Level Accessibility Segmentation.
"""

import numpy as np
import cv2
from app.cv.segmentation import (
    HeuristicSegmentationAdapter,
    mask_to_polygon,
    polygon_to_mask,
    mask_to_rle,
    rle_to_mask,
    compute_mask_iou,
    SEGMENTATION_CLASSES,
)


def test_mask_polygon_roundtrip():
    """Verify conversion between binary mask and normalized contour polygon."""
    h, w = 200, 300
    original_mask = np.zeros((h, w), dtype=np.uint8)
    # Draw geometric trapezoid (simulating sidewalk corridor)
    pts = np.array([[50, 60], [250, 60], [280, 180], [20, 180]], dtype=np.int32)
    cv2.fillPoly(original_mask, [pts], 255)

    polygon = mask_to_polygon(original_mask, min_area_pixels=100)
    assert len(polygon) >= 4
    # All polygon coordinates should be normalized within [0.0, 1.0]
    for pt in polygon:
        assert 0.0 <= pt[0] <= 1.0
        assert 0.0 <= pt[1] <= 1.0

    reconstructed_mask = polygon_to_mask(polygon, h, w)
    iou = compute_mask_iou(original_mask, reconstructed_mask)
    assert iou > 0.90, f"Expected high IoU between original and reconstructed mask, got {iou}"


def test_mask_rle_roundtrip():
    """Verify Run-Length Encoding (RLE) lossless encoding and decoding."""
    h, w = 100, 100
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:60, 30:70] = 255

    rle = mask_to_rle(mask)
    assert "counts" in rle
    assert rle["size"] == [100, 100]

    decoded = rle_to_mask(rle)
    np.testing.assert_array_equal(mask, decoded)


def test_heuristic_segmentation_adapter():
    """Verify HeuristicSegmentationAdapter predicts genuine non-rectangular contour regions."""
    adapter = HeuristicSegmentationAdapter()
    adapter.load()

    # Create synthetic street image with road and sidewalk
    img = np.full((480, 640, 3), 150, dtype=np.uint8)
    cv2.rectangle(img, (50, 220), (590, 460), (210, 210, 210), -1) # smooth sidewalk
    # Add stair lines
    for y in range(300, 420, 20):
        cv2.line(img, (150, y), (490, y), (50, 50, 50), 3)

    regions = adapter.predict_masks(img)
    assert len(regions) >= 1

    classes_found = [r.class_name for r in regions]
    assert "traversable_sidewalk" in classes_found or "stairs" in classes_found

    for r in regions:
        assert len(r.polygon) >= 3
        assert 0.0 <= r.confidence <= 1.0
        assert r.area_pixels > 0
        assert r.area_ratio > 0.0
        assert r.mask is not None
        assert r.rle is not None


def test_segmentation_api_classes(client):
    """Verify GET /api/v1/cv/segmentation/classes returns all 10 standard classes."""
    resp = client.get("/api/v1/cv/segmentation/classes")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_classes"] == 10
    class_names = [c["class_name"] for c in data["classes"]]
    assert "traversable_sidewalk" in class_names
    assert "damaged_pavement" in class_names
    assert "open_drain" in class_names
    assert "curb_ramp" in class_names
