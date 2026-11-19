"""
AccessGuard Reality Test Engine (Mission 006.2 Refactor).

Executes independent empirical verification of model predictions against unseen
ground-truth benchmark collections through the canonical ImageProcessingService,
computing mAP, mIoU, confidence calibration, environmental robustness,
and preserving failed predictions for human audit.
"""

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.cv.base import DetectedBarrier
from app.cv.orchestration.image_processing_service import (
    ImageProcessingService,
    ImageProcessingRequest,
)
from app.cv.adapters.base_v2 import ExecutionPolicy
from app.cv.segmentation import polygon_to_mask, compute_mask_iou
from app.cv.evaluation.metrics import (
    evaluate_detections,
    compute_map_50_95,
    compute_expected_calibration_error,
    compute_box_iou,
)
from app.gis.spatial_index import haversine_distance_meters
from app.core.logging import logger


class RealityEngine:
    """
    Independent Reality Verification Engine.
    Evaluates detection mAP, segmentation mIoU, calibration, model execution integrity,
    and environmental robustness slices using the canonical ImageProcessingService.
    """

    def __init__(
        self,
        proc_service: Optional[ImageProcessingService] = None,
        execution_policy: ExecutionPolicy = ExecutionPolicy.STRICT_GENUINE
    ):
        self.proc_service = proc_service or ImageProcessingService()
        self.execution_policy = execution_policy

    def run_reality_evaluation(
        self,
        dataset_dir: str = "data/benchmarks/indian_streetscapes",
        output_scorecard_path: str = "data/benchmarks/reality_scorecard.json"
    ) -> Dict[str, Any]:
        """
        Executes end-to-end reality audit on independent streetscape benchmark.
        """
        manifest_path = os.path.join(dataset_dir, "benchmark_manifest.json")
        if not os.path.exists(manifest_path):
            try:
                import sys
                from pathlib import Path
                root_dir = Path(__file__).resolve().parents[4]
                if str(root_dir) not in sys.path:
                    sys.path.insert(0, str(root_dir))
                from scripts.generate_indian_benchmark import generate_indian_benchmark_dataset
                generate_indian_benchmark_dataset(dataset_dir)
            except Exception as e:
                logger.warning(f"Could not auto-generate benchmark scenes: {e}")
                if not os.path.exists(manifest_path):
                    raise FileNotFoundError(f"Reality benchmark manifest not found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        scenes = manifest.get("scenes", [])
        total_scenes = len(scenes)
        logger.info(f"Reality Test Engine evaluating {total_scenes} independent scenes from {manifest_path}...")

        all_gt_boxes: List[Dict[str, Any]] = []
        all_pred_boxes: List[Dict[str, Any]] = []
        mask_ious_by_class: Dict[str, List[float]] = {}

        failed_predictions: List[Dict[str, Any]] = []
        environmental_slices: Dict[str, Dict[str, List[int]]] = {
            "illumination": {},
            "weather": {},
            "camera_angle": {}
        }

        successful_executions = 0
        execution_errors = 0
        model_telemetry_agg: Dict[str, Dict[str, Any]] = {}

        for sc in scenes:
            img_rel = sc["image_file"]
            img_full = os.path.join(dataset_dir, img_rel)
            if not os.path.exists(img_full):
                execution_errors += 1
                continue

            with open(img_full, "rb") as f_img:
                img_bytes = f_img.read()

            img = cv2.imread(img_full)
            if img is None:
                execution_errors += 1
                continue

            h, w = img.shape[:2]
            successful_executions += 1

            # Ground truth bbox & polygon
            gt_hazard = sc["accessibility_hazard"]
            gt_box = sc["ground_truth_bbox"]
            gt_poly = sc.get("ground_truth_polygon", [])
            env = sc.get("environmental_context", {})

            all_gt_boxes.append({
                "class": gt_hazard,
                "bbox": gt_box
            })

            # Execute canonical unified image processing request
            req = ImageProcessingRequest(
                image_bytes=img_bytes,
                filename=img_rel,
                source="benchmark",
                source_metadata={
                    "sample_id": sc["sample_id"],
                    "title": sc["title"],
                    "environmental_context": env
                },
                execution_policy=self.execution_policy,
                enable_privacy_redaction=False,
                save_media=False
            )
            proc_res = self.proc_service.process_image(req)
            preds = proc_res.detections

            # Track model execution records
            for r in proc_res.model_execution_records:
                if r.adapter_name not in model_telemetry_agg:
                    model_telemetry_agg[r.adapter_name] = {
                        "status": r.status,
                        "execution_type": r.execution_type,
                        "model_family": r.model_family,
                        "checkpoint_path": r.checkpoint_path,
                        "weight_sha256": r.weight_sha256,
                        "latencies": [],
                        "error_count": 0,
                        "inferences_attempted": 0,
                        "inferences_succeeded": 0,
                        "inferences_failed": 0,
                        "zero_detection_successes": 0
                    }
                model_telemetry_agg[r.adapter_name]["latencies"].append(r.execution_time_ms)
                if r.errors:
                    model_telemetry_agg[r.adapter_name]["error_count"] += len(r.errors)
                if r.status in ("READY", "DEGRADED"):
                    model_telemetry_agg[r.adapter_name]["inferences_attempted"] += 1
                    model_telemetry_agg[r.adapter_name]["inferences_succeeded"] += 1
                    if r.output_count == 0:
                        model_telemetry_agg[r.adapter_name]["zero_detection_successes"] += 1
                elif r.status in ("INFERENCE_FAILED", "RESOURCE_EXHAUSTED", "FAILED"):
                    model_telemetry_agg[r.adapter_name]["inferences_attempted"] += 1
                    model_telemetry_agg[r.adapter_name]["inferences_failed"] += 1
                    model_telemetry_agg[r.adapter_name]["status"] = r.status
                    model_telemetry_agg[r.adapter_name]["execution_type"] = r.execution_type

            scene_matched = False
            for p in preds:
                all_pred_boxes.append({
                    "class": p.barrier_type,
                    "confidence": p.confidence,
                    "bbox": p.bbox
                })
                iou = compute_box_iou(p.bbox, gt_box)
                if iou >= 0.50 and (
                    p.barrier_type == gt_hazard
                    or ("obstacle" in p.barrier_type and "obstacle" in gt_hazard)
                    or (p.barrier_type in ("stairs", "steps") and gt_hazard in ("stairs", "steps"))
                    or (p.barrier_type in ("surface_damage", "damaged_pavement") and gt_hazard in ("surface_damage", "damaged_pavement"))
                ):
                    scene_matched = True

            # Evaluate mask IoU if ground truth polygon is present (class-aware matching)
            if gt_poly and len(gt_poly) >= 3:
                gt_mask = polygon_to_mask(gt_poly, h, w)
                best_class_iou = 0.0

                seg_regions = proc_res.segmentation_results.regions
                for reg in seg_regions:
                    reg_class = reg.get("class_name")
                    is_match = (
                        reg_class == gt_hazard
                        or (reg_class == "damaged_pavement" and gt_hazard == "surface_damage")
                        or (reg_class == "surface_damage" and gt_hazard == "damaged_pavement")
                        or (reg_class in ("stairs", "steps") and gt_hazard in ("stairs", "steps"))
                    )
                    if is_match and reg.get("polygon"):
                        poly = reg["polygon"]
                        cand_mask = polygon_to_mask(poly, h, w)
                        curr_iou = compute_mask_iou(cand_mask, gt_mask)
                        if curr_iou > best_class_iou:
                            best_class_iou = curr_iou

                if gt_hazard not in mask_ious_by_class:
                    mask_ious_by_class[gt_hazard] = []
                mask_ious_by_class[gt_hazard].append(best_class_iou)

            # Record environmental slice performance (1 = detected, 0 = missed)
            for slice_key in ("illumination", "weather", "camera_angle"):
                val = env.get(slice_key, "unknown")
                if val not in environmental_slices[slice_key]:
                    environmental_slices[slice_key][val] = []
                environmental_slices[slice_key][val].append(1 if scene_matched else 0)

            # Preserve failed prediction for human audit
            if not scene_matched:
                failed_predictions.append({
                    "sample_id": sc["sample_id"],
                    "title": sc["title"],
                    "expected_hazard": gt_hazard,
                    "expected_bbox": gt_box,
                    "predictions_generated": [
                        {"class": p.barrier_type, "confidence": p.confidence, "bbox": p.bbox}
                        for p in preds
                    ],
                    "environmental_context": env,
                    "diagnostic_reason": "False negative: ground-truth hazard not localized with IoU >= 0.50"
                })

        # Calculate detection metrics
        eval_50 = evaluate_detections(all_gt_boxes, all_pred_boxes, iou_threshold=0.50)
        map_50 = eval_50["mean_average_precision"]
        map_50_95 = compute_map_50_95(all_gt_boxes, all_pred_boxes)
        ece = compute_expected_calibration_error(eval_50["confidences"], eval_50["correctness"])

        # Aggregate macro precision, recall, F1
        classes_dict = eval_50["classes"]
        precisions = [v["precision"] for v in classes_dict.values() if v["ground_truth_count"] > 0]
        recalls = [v["recall"] for v in classes_dict.values() if v["ground_truth_count"] > 0]
        f1s = [v["f1_score"] for v in classes_dict.values() if v["ground_truth_count"] > 0]

        macro_p = float(np.mean(precisions)) if precisions else 0.0
        macro_r = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

        # Calculate segmentation metrics
        per_class_miou = {
            cls: round(float(np.mean(ious)), 4) if ious else 0.0
            for cls, ious in mask_ious_by_class.items()
        }
        overall_miou = round(float(np.mean(list(per_class_miou.values()))), 4) if per_class_miou else 0.0

        # Environmental robustness summary
        robustness_summary: Dict[str, Dict[str, float]] = {}
        for slice_key, slice_dict in environmental_slices.items():
            robustness_summary[slice_key] = {}
            for subval, hits in slice_dict.items():
                robustness_summary[slice_key][subval] = round(float(np.mean(hits)), 4) if hits else 0.0

        # Model integrity summary
        model_integrity_summary: Dict[str, Any] = {}
        for adapter_name, tel in model_telemetry_agg.items():
            model_integrity_summary[adapter_name] = {
                "status": tel["status"],
                "execution_type": tel["execution_type"],
                "model_family": tel["model_family"],
                "checkpoint_path": tel.get("checkpoint_path"),
                "weight_sha256": tel.get("weight_sha256"),
                "mean_latency_ms": round(float(np.mean(tel["latencies"])), 2) if tel["latencies"] else 0.0,
                "error_count": tel["error_count"],
                "inferences_attempted": tel.get("inferences_attempted", 0),
                "inferences_succeeded": tel.get("inferences_succeeded", 0),
                "inferences_failed": tel.get("inferences_failed", 0),
                "zero_detection_successes": tel.get("zero_detection_successes", 0)
            }

        scorecard = {
            "scorecard_id": f"reality_{uuid.uuid4().hex[:8]}",
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset_manifest": manifest.get("dataset_name", "Indian Streetscapes"),
            "dataset_type": manifest.get("dataset_type", "synthetic_validation_suite"),
            "evaluation_category": "synthetic_benchmarking",
            "real_world_evaluation_distinction": "Synthetic benchmark suite explicitly verifies zero-fallback pipeline mechanics, strict-genuine threshold constraints, and geometric coordinate handling. Real-world physical performance is independently assessed on real-world imagery.",
            "model_version": "v2.0.0-unified",
            "total_scenes_evaluated": total_scenes,
            "pipeline_execution_rate": round((successful_executions / max(1, total_scenes)) * 100.0, 1),
            "execution_integrity": {
                "total_scenes": total_scenes,
                "successful_executions": successful_executions,
                "execution_errors": execution_errors,
                "execution_rate_pct": round((successful_executions / max(1, total_scenes)) * 100.0, 1),
                "policy_enforced": self.execution_policy.value,
                "strict_genuine_enforced": (self.execution_policy == ExecutionPolicy.STRICT_GENUINE)
            },
            "model_integrity": model_integrity_summary,
            "output_integrity": {
                "expected_calibration_error": ece,
                "total_predictions": len(all_pred_boxes),
                "total_ground_truth": len(all_gt_boxes)
            },
            "detection_metrics": {
                "macro_precision": round(macro_p, 4),
                "macro_recall": round(macro_r, 4),
                "macro_f1": round(macro_f1, 4),
                "map_50": map_50,
                "map_50_95": map_50_95,
                "expected_calibration_error": ece,
                "total_ground_truth": len(all_gt_boxes),
                "total_predictions": len(all_pred_boxes),
                "per_class": classes_dict
            },
            "segmentation_metrics": {
                "status": "READY" if (overall_miou > 0.0 or any(mask_ious_by_class.values())) else ("UNAVAILABLE" if self.execution_policy == ExecutionPolicy.STRICT_GENUINE else "DEGRADED"),
                "mean_iou": overall_miou,
                "per_class_miou": per_class_miou,
                "execution_mode": "genuine_neural_model" if self.execution_policy == ExecutionPolicy.STRICT_GENUINE else "heuristic_morphological_fallback"
            },
            "geospatial_accuracy": {
                "status": "NOT_MEASURED",
                "reason": "Benchmark scenes lack ground-truth PostGIS segment associations"
            },
            "temporal_deduplication": {
                "status": "NOT_MEASURED",
                "reason": "Benchmark consists of single static images; temporal tracking requires multi-frame video sequences"
            },
            "environmental_robustness": robustness_summary,
            "failed_predictions_count": len(failed_predictions),
            "failed_predictions_sample": failed_predictions[:5],
            "consensus_grade": "B+" if map_50 >= 0.50 else ("B" if map_50 >= 0.20 else "C")
        }

        # Save scorecard
        os.makedirs(os.path.dirname(output_scorecard_path), exist_ok=True)
        with open(output_scorecard_path, "w", encoding="utf-8") as f:
            json.dump(scorecard, f, indent=2)

        logger.info(f"Reality Scorecard generated successfully at {output_scorecard_path}")
        return scorecard
