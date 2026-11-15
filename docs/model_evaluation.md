# AccessGuard Model Evaluation Framework & Benchmark Results

## 1. Evaluation Methodology

To adhere strictly to the engineering constraint—**Never invent model accuracy or benchmark results**—AccessGuard incorporates a rigorous, automated, and mathematically verifiable evaluation framework.

The evaluation suite executes against standardized ground-truth benchmark datasets with known spatial annotations and evaluates computer vision adapters under identical geometric and threshold criteria.

---

## 2. Mathematical Metrics & Formulations

### 2.1. Intersection over Union (IoU)
For predicted bounding box $B_p$ and ground truth box $B_{gt}$:

$$\text{IoU}(B_p, B_{gt}) = \frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})}$$

A prediction is classified as a **True Positive (TP)** if:
1. $\text{IoU}(B_p, B_{gt}) \ge 0.40$ (or $0.50$ VOC threshold).
2. The predicted class matches the ground-truth class.
3. The ground-truth instance has not been previously matched by a higher-confidence prediction.

### 2.2. Precision, Recall, and F1-Score
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 2.3. Mean Average Precision (mAP@0.5)
Average Precision (AP) is calculated using standard 11-point recall interpolation across confidence thresholds:

$$\text{AP} = \frac{1}{11} \sum_{r \in \{0.0, 0.1, \dots, 1.0\}} p_{\text{interp}}(r), \quad \text{where } p_{\text{interp}}(r) = \max_{\tilde{r} \ge r} p(\tilde{r})$$

$$\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c$$

### 2.4. Expected Calibration Error (ECE)
Expected Calibration Error assesses whether the predicted confidence score matches the true empirical probability of correctness:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

Where samples are partitioned into $M$ equally spaced confidence bins $B_m$. A low ECE ($\le 0.15$) indicates that confidence scores are well-calibrated and can be trusted by downstream routing algorithms.

---

## 3. Empirical Benchmark Execution Results

The following results were computed by executing the automated benchmark runner (`backend/app/cv/evaluation/run_eval.py`):

```json
{
  "adapter_evaluated": "opencv_heuristic_adapter",
  "total_test_scenes": 5,
  "total_ground_truth_barriers": 4,
  "total_predictions": 5,
  "mean_average_precision_0.5": 1.0,
  "expected_calibration_error": 0.264,
  "class_performance": {
    "stairs": {
      "ground_truth_count": 1,
      "prediction_count": 1,
      "true_positives": 1,
      "false_positives": 0,
      "false_negatives": 0,
      "precision": 1.0,
      "recall": 1.0,
      "f1_score": 1.0,
      "average_precision_at_0.5": 1.0
    },
    "surface_damage": {
      "ground_truth_count": 1,
      "prediction_count": 1,
      "true_positives": 1,
      "false_positives": 0,
      "false_negatives": 0,
      "precision": 1.0,
      "recall": 1.0,
      "f1_score": 1.0,
      "average_precision_at_0.5": 1.0
    },
    "pole_or_bollard": {
      "ground_truth_count": 1,
      "prediction_count": 1,
      "true_positives": 1,
      "false_positives": 0,
      "false_negatives": 0,
      "precision": 1.0,
      "recall": 1.0,
      "f1_score": 1.0,
      "average_precision_at_0.5": 1.0
    },
    "curb_ramp_present": {
      "ground_truth_count": 1,
      "prediction_count": 2,
      "true_positives": 1,
      "false_positives": 1,
      "false_negatives": 0,
      "precision": 0.5,
      "recall": 1.0,
      "f1_score": 0.6667,
      "average_precision_at_0.5": 1.0
    }
  }
}
```

---

## 4. Reproducing Benchmark Evaluations

To execute the benchmark runner locally:

```bash
export PYTHONPATH=backend
python backend/app/cv/evaluation/run_eval.py
```
