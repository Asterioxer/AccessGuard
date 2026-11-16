"""
Indian Street-Scene Benchmark Evaluator.

Evaluates AccessGuard model adapters (Heuristic, Torchvision, YOLO, ONNX)
specifically against the curated Indian pedestrian infrastructure benchmark collection.
"""

import os
import json
import cv2
import numpy as np
from app.cv.evaluation.benchmark_suite import BenchmarkSuite
from app.cv.adapters.heuristic_adapter import HeuristicCVAdapter
from app.cv.adapters.torchvision_adapter import TorchvisionAdapter
from app.cv.adapters.yolo_adapter import YOLOAdapter
from app.cv.adapters.onnx_adapter import ONNXAdapter

def run_indian_benchmark():
    dataset_dir = "data/benchmarks/indian_streetscapes"
    manifest_path = os.path.join(dataset_dir, "benchmark_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    scenes = []
    for sc in manifest["scenes"]:
        img_path = os.path.join(dataset_dir, sc["image_file"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        scenes.append({
            "image": img,
            "ground_truth": [{
                "class": sc["accessibility_hazard"],
                "bbox": sc["ground_truth_bbox"]
            }],
            "title": sc["title"]
        })

    suite = BenchmarkSuite()
    results = suite.run_full_suite(scenes)

    out_path = "scripts/audit/indian_streetscape_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print(" ACCESSGUARD INDIAN STREETSCAPE BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Model Adapter':<28} | {'mAP@0.5':<10} | {'mAP@0.5:0.95':<12} | {'ECE':<8} | {'Latency (ms)':<12} | {'FPS':<6}")
    print("-"*80)
    for model_name, res in results.items():
        if "error" in res:
            print(f"{model_name:<28} | ERROR: {res['error']}")
        else:
            print(f"{model_name:<28} | {res['mAP_50']:<10.4f} | {res['mAP_50_95']:<12.4f} | {res['expected_calibration_error']:<8.4f} | {res['latency']['mean_ms']:<12.2f} | {res['latency']['fps']:<6.1f}")
    print("="*80)
    print(f"Detailed audit results saved to {out_path}\n")

if __name__ == "__main__":
    run_indian_benchmark()
