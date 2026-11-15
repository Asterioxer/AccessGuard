import pytest
import numpy as np
import cv2
from app.cv.base import DetectedBarrier
from app.cv.adapters.dummy_adapter import DummyAdapter
from app.cv.adapters.heuristic_adapter import HeuristicCVAdapter
from app.cv.pipeline import CVPipeline

def test_dummy_adapter_modes():
    adapter = DummyAdapter(mode="standard")
    empty_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = adapter.predict(empty_img)
    assert len(dets) == 2
    assert dets[0].barrier_type == "stairs"
    assert dets[0].confidence >= 0.85

    adapter_empty = DummyAdapter(mode="empty")
    assert adapter_empty.predict(empty_img) == []

    adapter_unc = DummyAdapter(mode="uncertain")
    unc_dets = adapter_unc.predict(empty_img)
    assert len(unc_dets) == 1
    assert unc_dets[0].is_uncertain is True

def test_heuristic_stair_detection():
    # Synthesize stairs pattern
    w, h = 640, 480
    img = np.full((h, w, 3), 180, dtype=np.uint8)
    for y in [240, 280, 320, 360, 400]:
        cv2.line(img, (100, y), (540, y), (30, 30, 30), 4)

    adapter = HeuristicCVAdapter()
    dets = adapter.predict(img)
    stair_dets = [d for d in dets if d.barrier_type == "stairs"]
    assert len(stair_dets) >= 1
    stair = stair_dets[0]
    assert stair.confidence > 0.65
    assert "rungs" in stair.evidence.get("explanation", "").lower()

def test_heuristic_surface_damage_detection():
    # Synthesize high-frequency crack pattern
    w, h = 640, 480
    img = np.full((h, w, 3), 150, dtype=np.uint8)
    np.random.seed(42)
    for _ in range(50):
        pt1 = (np.random.randint(100, 540), np.random.randint(280, 460))
        pt2 = (pt1[0] + np.random.randint(-30, 30), pt1[1] + np.random.randint(-30, 30))
        cv2.line(img, pt1, pt2, (10, 10, 10), 2)

    adapter = HeuristicCVAdapter()
    dets = adapter.predict(img)
    dmg_dets = [d for d in dets if d.barrier_type == "surface_damage"]
    assert len(dmg_dets) >= 1
    assert "fracture" in dmg_dets[0].evidence.get("explanation", "").lower()

def test_pipeline_uncertainty_calibration():
    pipeline = CVPipeline(adapter_name="dummy")
    # Using dummy adapter uncertain mode
    pipeline.adapter = DummyAdapter(mode="uncertain")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", img)
    calibrated, _ = pipeline.process_image(enc.tobytes())
    assert len(calibrated) == 1
    assert calibrated[0].evidence["calibrated_is_uncertain"] is True
    assert calibrated[0].evidence["calibrated_is_confirmed"] is False
