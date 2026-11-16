"""
Unit & Integration Tests for Ingestion Engine & Scheduler.
"""

import json
from datetime import datetime, timezone, timedelta
from app.ingestion.engine import AcquisitionEngine
from app.ingestion.scheduler import IngestionScheduler
from app.db.models import IngestionJob, ImageProvenance, Observation, PedestrianSegment


def test_ingestion_engine_duplicate_detection(db_session):
    """Verify AcquisitionEngine correctly detects external ID and spatial-temporal duplicates."""
    engine = AcquisitionEngine()

    prov = ImageProvenance(
        provider="mapillary",
        external_image_id="mapillary_dup_test_123",
        ingestion_timestamp="1700000000",
        latitude=28.6139,
        longitude=77.2090
    )
    db_session.add(prov)
    db_session.commit()

    # Exact duplicate check
    assert engine.is_duplicate(
        db=db_session,
        provider="mapillary",
        external_image_id="mapillary_dup_test_123",
        coordinates=(77.2090, 28.6139),
        capture_timestamp="2026-05-10T14:30:00Z"
    ) is True

    # Non-duplicate check
    assert engine.is_duplicate(
        db=db_session,
        provider="mapillary",
        external_image_id="mapillary_unique_456",
        coordinates=(77.2500, 28.6500),
        capture_timestamp="2026-05-10T14:30:00Z"
    ) is False


def test_ingestion_engine_job_execution(db_session):
    """Verify execution of an automated acquisition job produces observations and updates scores."""
    engine = AcquisitionEngine()

    # Create dummy pedestrian segment
    seg = PedestrianSegment(
        name="Connaught Place Test Sidewalk",
        start_lat=28.6130,
        start_lon=77.2080,
        end_lat=28.6150,
        end_lon=77.2100,
        geometry_geojson=json.dumps([[77.2080, 28.6130], [77.2100, 28.6150]]),
        length_meters=100.0,
        accessibility_score=50.0,
        score_grade="U"
    )
    db_session.add(seg)
    db_session.commit()

    job = IngestionJob(
        provider="mapillary",
        bbox_min_lon=77.2080,
        bbox_min_lat=28.6130,
        bbox_max_lon=77.2100,
        bbox_max_lat=28.6150,
        interval_minutes=60,
        priority="high"
    )
    db_session.add(job)
    db_session.commit()

    res = engine.execute_job(job.id, db_session, max_items=2)

    assert res["status"] == "completed"
    assert res["items_processed"] >= 1
    assert job.status == "completed"
    assert job.next_run_at is not None

    # Verify provenance and observation were created
    prov_count = db_session.query(ImageProvenance).filter(ImageProvenance.provider == "mapillary").count()
    assert prov_count >= 1


def test_ingestion_scheduler_priority_dispatch(db_session):
    """Verify IngestionScheduler dispatches high-priority jobs before normal/low jobs."""
    scheduler = IngestionScheduler()

    job_low = IngestionJob(
        provider="mapillary",
        bbox_min_lon=77.20,
        bbox_min_lat=28.60,
        bbox_max_lon=77.21,
        bbox_max_lat=28.61,
        priority="low"
    )
    job_high = IngestionJob(
        provider="kartaview",
        bbox_min_lon=77.20,
        bbox_min_lat=28.60,
        bbox_max_lon=77.21,
        bbox_max_lat=28.61,
        priority="high"
    )
    db_session.add_all([job_low, job_high])
    db_session.commit()

    executed = scheduler.trigger_pending_jobs(db_session, max_jobs=1)
    assert executed == 1

    # High priority job should be completed first
    db_session.refresh(job_high)
    db_session.refresh(job_low)
    assert job_high.status == "completed"
    assert job_low.status == "pending"
