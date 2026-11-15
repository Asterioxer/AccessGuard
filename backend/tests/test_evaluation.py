import pytest
from app.cv.evaluation.metrics import (
    compute_box_iou, evaluate_detections, compute_expected_calibration_error
)

def test_box_iou_calculation():
    # Box: (ymin, xmin, ymax, xmax)
    box1 = (0.0, 0.0, 1.0, 1.0) # area = 1.0
    box2 = (0.0, 0.0, 1.0, 1.0) # exact match
    assert compute_box_iou(box1, box2) == 1.0

    box3 = (0.0, 0.0, 0.5, 0.5) # area = 0.25, inside box1
    # intersection = 0.25, union = 1.0 -> IoU = 0.25
    assert compute_box_iou(box1, box3) == 0.25

    box4 = (2.0, 2.0, 3.0, 3.0) # non-overlapping
    assert compute_box_iou(box1, box4) == 0.0

def test_evaluation_metrics_ap_map():
    gt = [
        {"class": "stairs", "bbox": (0.2, 0.2, 0.8, 0.8)},
        {"class": "curb_ramp_missing", "bbox": (0.6, 0.1, 0.9, 0.4)}
    ]
    preds = [
        {"class": "stairs", "confidence": 0.95, "bbox": (0.2, 0.2, 0.8, 0.8)}, # TP
        {"class": "curb_ramp_missing", "confidence": 0.90, "bbox": (0.6, 0.1, 0.9, 0.4)}, # TP
        {"class": "stairs", "confidence": 0.50, "bbox": (0.0, 0.0, 0.1, 0.1)} # FP
    ]
    res = evaluate_detections(gt, preds, iou_threshold=0.5)
    assert res["mean_average_precision_0.5"] > 0.8
    assert res["classes"]["stairs"]["true_positives"] == 1
    assert res["classes"]["stairs"]["false_positives"] == 1
    assert res["classes"]["curb_ramp_missing"]["true_positives"] == 1

def test_expected_calibration_error():
    confidences = [0.95, 0.90, 0.85, 0.80]
    correctness = [True, True, True, True]
    ece = compute_expected_calibration_error(confidences, correctness, num_bins=5)
    assert ece >= 0.0
    assert ece <= 1.0
