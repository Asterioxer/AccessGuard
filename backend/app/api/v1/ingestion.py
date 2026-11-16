"""
Automated Ingestion & Multi-Source Imagery Acquisition API.
"""

from typing import Any, Dict, List, Optional, Tuple
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import IngestionJob, ImageProvenance, User
from app.schemas.ingestion import (
    IngestionJobCreate,
    IngestionJobResponse,
    AdHocImageryQuery,
    ProvenanceRecordSchema,
    IngestionProviderInfo,
)
from app.ingestion.engine import AcquisitionEngine
from app.core.security import require_role, get_optional_user_payload
from app.core.logging import logger

router = APIRouter(prefix="/ingestion", tags=["Ingestion"])
engine_instance = AcquisitionEngine()


@router.get("/providers", response_model=List[IngestionProviderInfo])
def list_providers():
    """Returns status, authentication, rate limits, and quota info for all imagery providers."""
    return engine_instance.get_providers_status()


@router.post("/query")
def query_provider_imagery(query: AdHocImageryQuery):
    """Performs an ad-hoc geographic search for street-level or satellite imagery."""
    try:
        search_res = engine_instance.query_provider(
            provider=query.provider,
            bbox=query.bbox,
            start_date=query.start_date,
            end_date=query.end_date,
            limit=query.limit
        )
        return {
            "provider": search_res.provider,
            "total_count": search_res.total_count,
            "query_bbox": search_res.query_bbox,
            "next_page_token": search_res.next_page_token,
            "items": [
                {
                    "item_id": item.item_id,
                    "provider": item.provider,
                    "coordinates": item.coordinates,
                    "capture_timestamp": item.capture_timestamp,
                    "url": item.url,
                    "heading": item.heading,
                    "sequence_id": item.sequence_id,
                    "license": item.license,
                    "attribution": item.attribution,
                    "is_live": item.is_live,
                    "raw_metadata": item.raw_metadata
                }
                for item in search_res.items
            ]
        }
    except Exception as e:
        logger.error(f"Error querying provider {query.provider}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/jobs", response_model=IngestionJobResponse, status_code=status.HTTP_201_CREATED)
def create_ingestion_job(
    job_in: IngestionJobCreate,
    db: Session = Depends(get_db),
    user_payload: Dict[str, Any] = Depends(require_role(["contributor", "verifier", "admin"]))
):
    """Registers an automated ingestion job for periodic polling."""
    import json
    min_lon, min_lat, max_lon, max_lat = job_in.bbox
    job = IngestionJob(
        provider=job_in.provider,
        bbox_min_lon=min_lon,
        bbox_min_lat=min_lat,
        bbox_max_lon=max_lon,
        bbox_max_lat=max_lat,
        interval_minutes=job_in.interval_minutes,
        priority=job_in.priority,
        config_json=json.dumps(job_in.config or {})
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    return IngestionJobResponse(
        id=job.id,
        provider=job.provider,
        status=job.status,
        priority=job.priority,
        bbox=(job.bbox_min_lon, job.bbox_min_lat, job.bbox_max_lon, job.bbox_max_lat),
        interval_minutes=job.interval_minutes,
        items_fetched=job.items_fetched,
        items_processed=job.items_processed,
        items_failed=job.items_failed,
        error_message=job.error_message,
        last_run_at=job.last_run_at,
        next_run_at=job.next_run_at,
        created_at=job.created_at
    )


@router.get("/jobs", response_model=List[IngestionJobResponse])
def list_ingestion_jobs(db: Session = Depends(get_db)):
    """Lists all automated ingestion jobs."""
    jobs = db.query(IngestionJob).order_by(IngestionJob.created_at.desc()).all()
    res = []
    for job in jobs:
        res.append(
            IngestionJobResponse(
                id=job.id,
                provider=job.provider,
                status=job.status,
                priority=job.priority,
                bbox=(job.bbox_min_lon, job.bbox_min_lat, job.bbox_max_lon, job.bbox_max_lat),
                interval_minutes=job.interval_minutes,
                items_fetched=job.items_fetched,
                items_processed=job.items_processed,
                items_failed=job.items_failed,
                error_message=job.error_message,
                last_run_at=job.last_run_at,
                next_run_at=job.next_run_at,
                created_at=job.created_at
            )
        )
    return res


@router.get("/jobs/{job_id}", response_model=IngestionJobResponse)
def get_ingestion_job(job_id: str, db: Session = Depends(get_db)):
    """Retrieves ingestion job status and telemetry."""
    job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return IngestionJobResponse(
        id=job.id,
        provider=job.provider,
        status=job.status,
        priority=job.priority,
        bbox=(job.bbox_min_lon, job.bbox_min_lat, job.bbox_max_lon, job.bbox_max_lat),
        interval_minutes=job.interval_minutes,
        items_fetched=job.items_fetched,
        items_processed=job.items_processed,
        items_failed=job.items_failed,
        error_message=job.error_message,
        last_run_at=job.last_run_at,
        next_run_at=job.next_run_at,
        created_at=job.created_at
    )


@router.post("/jobs/{job_id}/run")
def trigger_job_run(
    job_id: str,
    db: Session = Depends(get_db),
    user_payload: Dict[str, Any] = Depends(require_role(["verifier", "admin"]))
):
    """Manually triggers immediate execution of an ingestion job."""
    try:
        outcome = engine_instance.execute_job(job_id, db)
        return outcome
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to execute job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/provenance/{observation_id}")
def get_observation_provenance(observation_id: str, db: Session = Depends(get_db)):
    """Retrieves external data provenance and attribution for an observation."""
    import json
    prov = db.query(ImageProvenance).filter(ImageProvenance.observation_id == observation_id).first()
    if not prov:
        raise HTTPException(status_code=404, detail="No provenance record found for this observation.")

    raw_meta = {}
    if prov.raw_metadata_json:
        try:
            raw_meta = json.loads(prov.raw_metadata_json)
        except Exception:
            pass

    return {
        "id": prov.id,
        "observation_id": prov.observation_id,
        "provider": prov.provider,
        "external_image_id": prov.external_image_id,
        "sequence_id": prov.sequence_id,
        "capture_timestamp": prov.capture_timestamp,
        "ingestion_timestamp": prov.ingestion_timestamp,
        "latitude": prov.latitude,
        "longitude": prov.longitude,
        "heading": prov.heading,
        "source_attribution": prov.source_attribution,
        "license_metadata": prov.license_metadata,
        "model_version": prov.model_version,
        "raw_metadata": raw_meta
    }
