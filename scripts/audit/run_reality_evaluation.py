"""
Reality Test Engine Execution Script (Mission 005 Addendum).

Executes independent blind evaluation across 25 diverse Indian streetscapes,
generating an evidence-backed Reality Scorecard with detection mAP, segmentation mIoU,
calibration ECE, environmental robustness slices, and failed prediction logs.
"""

from app.cv.evaluation.reality_engine import RealityEngine
from app.core.logging import logger

def main():
    engine = RealityEngine()
    scorecard = engine.run_reality_evaluation(
        dataset_dir="data/benchmarks/indian_streetscapes",
        output_scorecard_path="data/benchmarks/reality_scorecard.json"
    )

    print("\n" + "="*80)
    print(" ACCESSGUARD REALITY TEST ENGINE SCORECARD (MISSION 005 ADDENDUM)")
    print("="*80)
    print(f"Scorecard ID:               {scorecard['scorecard_id']}")
    print(f"Timestamp:                  {scorecard['evaluation_timestamp']}")
    print(f"Dataset Manifest:           {scorecard['dataset_manifest']}")
    print(f"Total Scenes Evaluated:     {scorecard['total_scenes_evaluated']}")
    print(f"Pipeline Execution Rate:    {scorecard['pipeline_execution_rate']}%")
    print(f"Consensus Grade:            {scorecard['consensus_grade']}")
    print("-"*80)
    det = scorecard["detection_metrics"]
    print(f"Detection Macro Precision:  {det['macro_precision']:.4f}")
    print(f"Detection Macro Recall:     {det['macro_recall']:.4f}")
    print(f"Detection Macro F1:         {det['macro_f1']:.4f}")
    print(f"Detection mAP@0.5:          {det['map_50']:.4f}")
    print(f"Detection mAP@0.5:0.95:     {det['map_50_95']:.4f}")
    print(f"Calibration ECE:            {det['expected_calibration_error']:.4f}")
    print("-"*80)
    seg = scorecard["segmentation_metrics"]
    print(f"Segmentation Mean IoU:      {seg['mean_iou']:.4f}")
    for cls, val in seg["per_class_miou"].items():
        print(f"  - {cls:<25}: IoU {val:.4f}")
    print("-"*80)
    geo = scorecard.get("geospatial_accuracy", {})
    if geo.get("status") == "NOT_MEASURED":
        print(f"Geospatial Association:     NOT_MEASURED ({geo.get('reason', '')})")
    else:
        print(f"Geospatial Association Error: {geo.get('mean_association_error_meters', 'N/A')}m (Correctly Snapped: {geo.get('percent_correctly_snapped', 'N/A')}%)")
    dedup = scorecard.get("temporal_deduplication", {})
    if dedup.get("status") == "NOT_MEASURED":
        print(f"Temporal Deduplication:       NOT_MEASURED ({dedup.get('reason', '')})")
    else:
        print(f"Temporal Dedup Efficiency:    {dedup.get('suppression_efficiency_percent', 'N/A')}% (Reduction: {dedup.get('frame_to_obstacle_reduction_ratio', 'N/A')})")
    print("-"*80)
    print("Environmental Robustness:")
    for cat, vals in scorecard["environmental_robustness"].items():
        print(f"  [{cat}]")
        for sub, score in vals.items():
            print(f"    - {sub:<24}: {score * 100:.1f}%")
    print("-"*80)
    print(f"Failed Predictions Preserved: {scorecard['failed_predictions_count']}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
