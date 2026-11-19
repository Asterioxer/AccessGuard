"""
Unit & Integration Tests for Reality Test Engine (Mission 005 Addendum).
"""

from app.cv.evaluation.reality_engine import RealityEngine


def test_reality_engine_scorecard_generation(tmp_path):
    """Verify RealityEngine executes blind evaluation and generates complete Reality Scorecard."""
    engine = RealityEngine()
    out_file = tmp_path / "test_scorecard.json"

    scorecard = engine.run_reality_evaluation(
        dataset_dir="data/benchmarks/indian_streetscapes",
        output_scorecard_path=str(out_file)
    )

    assert scorecard["total_scenes_evaluated"] == 25
    assert scorecard["pipeline_execution_rate"] == 100.0
    assert "detection_metrics" in scorecard
    assert "segmentation_metrics" in scorecard
    assert "environmental_robustness" in scorecard

    # Ensure segmentation mIoU metric structure exists
    seg = scorecard["segmentation_metrics"]
    assert seg["mean_iou"] >= 0.0
    assert "stairs" in seg["per_class_miou"]
    assert "surface_damage" in seg["per_class_miou"]

    # Ensure environmental robustness slices exist
    env = scorecard["environmental_robustness"]
    assert "illumination" in env
    assert "weather" in env
    assert "camera_angle" in env

    # Ensure failed predictions are preserved
    assert scorecard["failed_predictions_count"] > 0
    assert len(scorecard["failed_predictions_sample"]) > 0


def test_reality_api_endpoints(client):
    """Verify GET /api/v1/reality/scorecard and /failures endpoints."""
    resp = client.get("/api/v1/reality/scorecard")
    assert resp.status_code == 200
    data = resp.json()
    assert "scorecard_id" in data
    assert data["total_scenes_evaluated"] == 25

    resp_fail = client.get("/api/v1/reality/failures")
    assert resp_fail.status_code == 200
    data_fail = resp_fail.json()
    assert "failed_count" in data_fail
    assert len(data_fail["samples"]) > 0
