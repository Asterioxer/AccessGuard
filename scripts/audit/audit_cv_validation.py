import sys
from pathlib import Path
import json
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from app.cv.pipeline import CVPipeline
from app.cv.adapters.heuristic_adapter import HeuristicCVAdapter
from app.cv.adapters.torchvision_adapter import TorchvisionAdapter
from app.cv.adapters.yolo_adapter import YOLOAdapter
from app.cv.adapters.dummy_adapter import DummyAdapter
from app.cv.evaluation.metrics import evaluate_detections, compute_expected_calibration_error, compute_box_iou

def audit_cv_pipeline():
    results = {}
    w, h = 640, 480

    print("=== Phase 2: Computer Vision Audit ===")

    # 1. Adapter Implementations Audit
    print("\n1. Auditing Model Adapter Implementations...")
    dummy = DummyAdapter()
    heuristic = HeuristicCVAdapter()
    torch_adapter = TorchvisionAdapter()
    yolo_adapter = YOLOAdapter()

    results["adapters"] = {
        "dummy": {
            "type": "Mocked / Deterministic test stub",
            "is_model_driven": False,
            "has_fallback": False
        },
        "heuristic": {
            "type": "Classical Heuristic Computer Vision (OpenCV)",
            "is_model_driven": False,
            "has_fallback": False
        },
        "torchvision": {
            "type": "PyTorch deep learning wrapper with Heuristic fallback",
            "is_model_driven": torch_adapter.model is not None,
            "fallback_active": torch_adapter.model is None,
            "notes": "Falls back to HeuristicCVAdapter if no GPU/weights supplied."
        },
        "yolo": {
            "type": "YOLO / ONNX stub with Heuristic fallback",
            "is_model_driven": False,
            "fallback_active": True,
            "notes": "Falls back to HeuristicCVAdapter unconditionally when model_path is None."
        }
    }

    # 2. Test Crosswalk Striping False Positive on Stair Detector
    print("\n2. Testing Crosswalk vs Stairs False Positive...")
    # A standard zebra crosswalk has wide parallel horizontal bands
    img_crosswalk = np.full((h, w, 3), 120, dtype=np.uint8)
    for y in [240, 290, 340, 390, 440]:
        cv2.rectangle(img_crosswalk, (80, y), (560, y + 25), (240, 240, 240), -1)

    crosswalk_dets = heuristic.predict(img_crosswalk)
    stair_fp = [d for d in crosswalk_dets if d.barrier_type == "stairs"]
    results["crosswalk_test"] = {
        "false_positive_stairs_detected": len(stair_fp) > 0,
        "details": [(d.barrier_type, d.confidence, d.uncertainty_score) for d in crosswalk_dets]
    }
    print(f"   Crosswalk detected as stairs? {len(stair_fp) > 0} (Detections: {len(crosswalk_dets)})")

    # 3. Test Difficult Lighting Conditions
    print("\n3. Testing Difficult Lighting Conditions...")
    # Base stairs image
    img_stairs_base = np.full((h, w, 3), 180, dtype=np.uint8)
    for y in [240, 280, 320, 360, 400]:
        cv2.line(img_stairs_base, (100, y), (540, y), (40, 40, 40), 4)

    # A: Severe Underexposure / Night (mean brightness < 20)
    img_dark = (img_stairs_base * 0.10).astype(np.uint8)
    dark_dets = heuristic.predict(img_dark)
    dark_stairs = [d for d in dark_dets if d.barrier_type == "stairs"]

    # B: Severe Overexposure / Direct Sun Glare (mean brightness > 230)
    img_glare = np.clip(img_stairs_base.astype(np.float32) + 80, 0, 255).astype(np.uint8)
    glare_dets = heuristic.predict(img_glare)
    glare_stairs = [d for d in glare_dets if d.barrier_type == "stairs"]

    # C: Low Contrast / Fog (std < 10)
    img_fog = np.full((h, w, 3), 140, dtype=np.uint8)
    for y in [240, 280, 320, 360, 400]:
        cv2.line(img_fog, (100, y), (540, y), (132, 132, 132), 3)
    fog_dets = heuristic.predict(img_fog)
    fog_stairs = [d for d in fog_dets if d.barrier_type == "stairs"]

    results["lighting_stress_test"] = {
        "underexposure_night": {
            "mean_brightness": float(np.mean(img_dark)),
            "detected": len(dark_stairs) > 0,
            "uncertainty_score": dark_stairs[0].uncertainty_score if dark_stairs else None
        },
        "glare_overexposure": {
            "mean_brightness": float(np.mean(img_glare)),
            "detected": len(glare_stairs) > 0,
            "uncertainty_score": glare_stairs[0].uncertainty_score if glare_stairs else None
        },
        "low_contrast_fog": {
            "std_contrast": float(np.std(img_fog)),
            "detected": len(fog_stairs) > 0,
            "uncertainty_score": fog_stairs[0].uncertainty_score if fog_stairs else None
        }
    }
    print(f"   Underexposure (night): detected={len(dark_stairs) > 0}")
    print(f"   Overexposure (glare): detected={len(glare_stairs) > 0}")
    print(f"   Low contrast (fog): detected={len(fog_stairs) > 0}")

    # 4. Preprocessing: RGB vs BGR Channel Inversion
    print("\n4. Testing Preprocessing Channel Consistency (RGB vs BGR)...")
    img_bgr = np.full((h, w, 3), (200, 50, 50), dtype=np.uint8) # Blue dominant in BGR
    for y in [240, 280, 320, 360, 400]:
        cv2.line(img_bgr, (100, y), (540, y), (20, 20, 20), 4)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Red dominant
    dets_bgr = heuristic.predict(img_bgr)
    dets_rgb = heuristic.predict(img_rgb)
    
    results["channel_consistency"] = {
        "bgr_detection_count": len(dets_bgr),
        "rgb_detection_count": len(dets_rgb),
        "bgr_confidences": [d.confidence for d in dets_bgr],
        "rgb_confidences": [d.confidence for d in dets_rgb],
        "sensitive_to_channel_order": [d.confidence for d in dets_bgr] != [d.confidence for d in dets_rgb]
    }
    print(f"   BGR vs RGB identical? {[d.confidence for d in dets_bgr] == [d.confidence for d in dets_rgb]}")

    # 5. Dataset Split & Leakage Audit
    print("\n5. Auditing Evaluation Split & Data Leakage...")
    results["dataset_leakage_audit"] = {
        "has_train_test_split": False,
        "sample_count": 5,
        "data_leakage_identified": True,
        "leakage_mechanism": "Heuristic thresholds (edge density > 0.025, step count clustering) were tuned directly against the 5 synthetic evaluation scenes."
    }

    # 6. Indian Pedestrian Infrastructure Representation Audit
    results["indian_infrastructure_audit"] = {
        "represented_in_evaluation": False,
        "critical_unhandled_barriers": [
            "Open/broken storm-water drain slabs (kaccha drains)",
            "Dislodged, uneven interlocking paver blocks",
            "Sidewalk encroachment by parked two-wheelers and street vendor stalls",
            "High curb transitions (30cm - 50cm without curb cuts vs Western ADA 15cm)",
            "Monsoon waterlogging / puddles obscuring sidewalk depth",
            "Utility poles and telecom transformer boxes obstructing the middle of footpaths",
            "Stray cattle / animal obstacles"
        ]
    }

    out_path = Path(__file__).parent / "cv_audit_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCV Audit complete. Results saved to {out_path}")
    return results

if __name__ == "__main__":
    audit_cv_pipeline()
