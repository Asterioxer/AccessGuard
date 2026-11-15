import numpy as np
from typing import List, Dict, Tuple, Any

def compute_box_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Computes IoU between two (ymin, xmin, ymax, xmax) boxes."""
    y1_max = max(box1[0], box2[0])
    x1_max = max(box1[1], box2[1])
    y2_min = min(box1[2], box2[2])
    x2_min = min(box1[3], box2[3])

    if y2_min <= y1_max or x2_min <= x1_max:
        return 0.0

    intersection = (y2_min - y1_max) * (x2_min - x1_max)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def evaluate_detections(
    ground_truth: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    iou_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Evaluates predictions against ground truth using PASCAL VOC / COCO style evaluation.
    Computes per-class True Positives (TP), False Positives (FP), False Negatives (FN),
    Precision, Recall, F1-Score, and AP.
    """
    classes = list(set([item["class"] for item in ground_truth] + [item["class"] for item in predictions]))
    results = {}

    for cls in classes:
        cls_gt = [gt for gt in ground_truth if gt["class"] == cls]
        cls_preds = [p for p in predictions if p["class"] == cls]
        
        # Sort predictions by confidence descending
        cls_preds.sort(key=lambda x: x["confidence"], reverse=True)
        
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        gt_matched = [False] * len(cls_gt)
        
        for i, pred in enumerate(cls_preds):
            best_iou = 0.0
            best_gt_idx = -1
            
            for j, gt in enumerate(cls_gt):
                if gt_matched[j]:
                    continue
                iou = compute_box_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
                    
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp[i] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1
                
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        total_gt = len(cls_gt)
        if total_gt == 0:
            recall = np.zeros_like(cum_tp)
            precision = np.zeros_like(cum_tp)
            ap = 0.0
        else:
            recall = cum_tp / total_gt
            precision = cum_tp / (cum_tp + cum_fp + 1e-10)
            
            # 11-point interpolation for Average Precision
            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                p_interp = np.max(precision[recall >= t]) if np.sum(recall >= t) > 0 else 0.0
                ap += p_interp / 11.0

        final_tp = int(np.sum(tp))
        final_fp = int(np.sum(fp))
        final_fn = total_gt - final_tp
        
        final_prec = final_tp / (final_tp + final_fp) if (final_tp + final_fp) > 0 else 0.0
        final_rec = final_tp / total_gt if total_gt > 0 else 0.0
        f1 = (2 * final_prec * final_rec) / (final_prec + final_rec) if (final_prec + final_rec) > 0 else 0.0
        
        results[cls] = {
            "ground_truth_count": total_gt,
            "prediction_count": len(cls_preds),
            "true_positives": final_tp,
            "false_positives": final_fp,
            "false_negatives": final_fn,
            "precision": round(final_prec, 4),
            "recall": round(final_rec, 4),
            "f1_score": round(f1, 4),
            "average_precision_at_0.5": round(ap, 4)
        }

    # Mean Average Precision (mAP)
    ap_values = [v["average_precision_at_0.5"] for v in results.values() if v["ground_truth_count"] > 0]
    map_score = float(np.mean(ap_values)) if ap_values else 0.0

    return {
        "classes": results,
        "mean_average_precision_0.5": round(map_score, 4)
    }

def compute_expected_calibration_error(
    confidences: List[float],
    correctness: List[bool],
    num_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) for assessing confidence calibration.
    ECE measures the difference in expectation between confidence and accuracy.
    """
    if len(confidences) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(num_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        
        in_bin = [
            (c, corr) for c, corr in zip(confidences, correctness)
            if (bin_lower <= c < bin_upper if i < num_bins - 1 else bin_lower <= c <= bin_upper)
        ]
        
        if len(in_bin) > 0:
            bin_conf = np.mean([x[0] for x in in_bin])
            bin_acc = np.mean([1.0 if x[1] else 0.0 for x in in_bin])
            bin_weight = len(in_bin) / total_samples
            ece += bin_weight * abs(bin_acc - bin_conf)
            
    return round(float(ece), 4)
