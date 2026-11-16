"""
AccessGuard Multi-Model Benchmark Suite.

Executes reproducible, independent benchmarks across Heuristic, Torchvision, YOLO,
and ONNX adapters, evaluating precision, recall, mAP@0.5, mAP@0.5:0.95, ECE,
confusion matrices, inference latency percentiles, and memory profiles.
"""

import time
import os
import psutil
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

from app.cv.base import ModelAdapter
from app.cv.adapters.heuristic_adapter import HeuristicCVAdapter
from app.cv.adapters.torchvision_adapter import TorchvisionAdapter
from app.cv.adapters.yolo_adapter import YOLOAdapter
from app.cv.adapters.onnx_adapter import ONNXAdapter
from app.cv.evaluation.metrics import (
    evaluate_detections,
    compute_map_50_95,
    compute_confusion_matrix,
    compute_expected_calibration_error
)
from app.core.logging import logger

def measure_memory_mb() -> Tuple[float, float]:
    """Measures current process RSS RAM (MB) and PyTorch CUDA allocated VRAM (MB)."""
    process = psutil.Process()
    ram_mb = process.memory_info().rss / (1024 * 1024)
    vram_mb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return round(ram_mb, 2), round(vram_mb, 2)

class BenchmarkSuite:
    """Orchestrates comprehensive model evaluation across standardized test datasets."""

    def __init__(self, target_classes: Optional[List[str]] = None):
        self.target_classes = target_classes or [
            "stairs",
            "curb_ramp_present",
            "surface_damage",
            "raised_curb",
            "pole_or_bollard",
            "construction_obstacle"
        ]

    def run_adapter_benchmark(
        self,
        adapter: ModelAdapter,
        test_scenes: List[Dict[str, Any]],
        num_warmup: int = 2
    ) -> Dict[str, Any]:
        """
        Runs comprehensive benchmark for a single model adapter on a test dataset.
        """
        logger.info(f"Starting benchmark for {adapter.name} across {len(test_scenes)} scenes...")
        adapter.load()

        # Warmup
        if test_scenes and num_warmup > 0:
            for _ in range(num_warmup):
                adapter.predict(test_scenes[0]["image"])

        latencies_ms: List[float] = []
        all_gt: List[Dict[str, Any]] = []
        all_preds: List[Dict[str, Any]] = []

        ram_before, vram_before = measure_memory_mb()

        for scene in test_scenes:
            img = scene["image"]
            gt_boxes = scene["ground_truth"]

            # Ground truth collection
            for gt in gt_boxes:
                all_gt.append({
                    "class": gt["class"],
                    "bbox": gt["bbox"]
                })

            # Timed prediction
            t0 = time.perf_counter()
            preds = adapter.predict(img)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            # Predictions collection
            for p in preds:
                all_preds.append({
                    "class": p.barrier_type,
                    "confidence": p.confidence,
                    "bbox": p.bbox
                })

        ram_after, vram_after = measure_memory_mb()

        # Compute metrics
        eval_50 = evaluate_detections(all_gt, all_preds, iou_threshold=0.50)
        map_50 = eval_50["mean_average_precision"]
        map_50_95 = compute_map_50_95(all_gt, all_preds)

        ece = compute_expected_calibration_error(eval_50["confidences"], eval_50["correctness"])
        confusion = compute_confusion_matrix(all_gt, all_preds, self.target_classes, iou_threshold=0.50)

        # Aggregate macro precision, recall, F1
        classes_dict = eval_50["classes"]
        precisions = [v["precision"] for v in classes_dict.values() if v["ground_truth_count"] > 0]
        recalls = [v["recall"] for v in classes_dict.values() if v["ground_truth_count"] > 0]
        f1s = [v["f1_score"] for v in classes_dict.values() if v["ground_truth_count"] > 0]

        macro_p = float(np.mean(precisions)) if precisions else 0.0
        macro_r = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

        # Latency percentiles
        lat_arr = np.array(latencies_ms)
        latency_profile = {
            "mean_ms": round(float(np.mean(lat_arr)), 2) if len(lat_arr) > 0 else 0.0,
            "p50_ms": round(float(np.median(lat_arr)), 2) if len(lat_arr) > 0 else 0.0,
            "p95_ms": round(float(np.percentile(lat_arr, 95)), 2) if len(lat_arr) > 0 else 0.0,
            "p99_ms": round(float(np.percentile(lat_arr, 99)), 2) if len(lat_arr) > 0 else 0.0,
            "fps": round(1000.0 / float(np.mean(lat_arr)), 1) if len(lat_arr) > 0 and np.mean(lat_arr) > 0 else 0.0
        }

        memory_profile = {
            "ram_rss_mb": ram_after,
            "vram_allocated_mb": vram_after,
            "ram_delta_mb": round(ram_after - ram_before, 2),
            "vram_delta_mb": round(vram_after - vram_before, 2)
        }

        return {
            "adapter_name": adapter.name,
            "num_test_scenes": len(test_scenes),
            "ground_truth_count": len(all_gt),
            "prediction_count": len(all_preds),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "mAP_50": map_50,
            "mAP_50_95": map_50_95,
            "expected_calibration_error": ece,
            "per_class_metrics": classes_dict,
            "confusion_matrix": confusion,
            "latency": latency_profile,
            "memory": memory_profile
        }

    def run_full_suite(self, test_scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs benchmarks across all 4 adapters:
        1. HeuristicCVAdapter
        2. TorchvisionAdapter
        3. YOLOAdapter
        4. ONNXAdapter
        """
        adapters = [
            HeuristicCVAdapter(),
            TorchvisionAdapter(),
            YOLOAdapter(),
            ONNXAdapter()
        ]

        results = {}
        for adp in adapters:
            try:
                results[adp.name] = self.run_adapter_benchmark(adp, test_scenes)
            except Exception as e:
                logger.error(f"Failed benchmark for {adp.name}: {e}")
                results[adp.name] = {"error": str(e)}

        return results
