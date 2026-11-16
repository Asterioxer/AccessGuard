"""
Reality Test Engine & Scorecard API (Mission 005 Addendum).
"""

import json
import os
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import RealityTestRun
from app.cv.evaluation.reality_engine import RealityEngine
from app.core.security import require_role
from app.core.logging import logger

router = APIRouter(prefix="/reality", tags=["Reality Engine"])
reality_engine = RealityEngine()
SCORECARD_PATH = "data/benchmarks/reality_scorecard.json"


@router.get("/scorecard")
def get_latest_scorecard():
    """Retrieves the latest empirical Reality Scorecard."""
    if os.path.exists(SCORECARD_PATH):
        try:
            with open(SCORECARD_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read scorecard: {e}")

    # Generate if not present
    return reality_engine.run_reality_evaluation(output_scorecard_path=SCORECARD_PATH)


@router.post("/evaluate")
def trigger_reality_evaluation(
    db: Session = Depends(get_db),
    user_payload: Dict[str, Any] = Depends(require_role(["verifier", "admin"]))
):
    """Executes the complete Reality Test Engine verification suite."""
    try:
        scorecard = reality_engine.run_reality_evaluation(output_scorecard_path=SCORECARD_PATH)

        # Persist test run record in DB
        run_record = RealityTestRun(
            test_suite_name=scorecard.get("dataset_manifest", "Indian Streetscapes"),
            scorecard_json=json.dumps(scorecard),
            passed=scorecard.get("detection_metrics", {}).get("map_50", 0.0) >= 0.20,
            summary_markdown=f"mAP@0.5: {scorecard['detection_metrics']['map_50']:.4f}, mIoU: {scorecard['segmentation_metrics']['mean_iou']:.4f}"
        )
        db.add(run_record)
        db.commit()

        return scorecard
    except Exception as e:
        logger.error(f"Reality evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/failures")
def list_failed_predictions():
    """Retrieves preserved failed predictions for universal accessibility auditor review."""
    if not os.path.exists(SCORECARD_PATH):
        reality_engine.run_reality_evaluation(output_scorecard_path=SCORECARD_PATH)

    try:
        with open(SCORECARD_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "failed_count": data.get("failed_predictions_count", 0),
            "samples": data.get("failed_predictions_sample", []),
            "dataset_manifest": data.get("dataset_manifest")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
