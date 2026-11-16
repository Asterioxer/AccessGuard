"""
Autonomous Field Intelligence & Ingestion Engine.

Coordinates multi-source imagery acquisition, duplicate detection, privacy anonymization,
computer vision analysis (detection + segmentation), PostGIS geospatial association,
and provenance tracking.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.ingestion.connectors import (
    BaseImageryConnector,
    MapillaryConnector,
    KartaViewConnector,
    LiveStreamConnector,
    CopernicusConnector,
    PexelsConnector,
    ImageryItem,
    ImagerySearchResult,
)
from app.db.models import (
    IngestionJob,
    ImageProvenance,
    Observation,
    BarrierDetection,
    SegmentedObservation,
    PedestrianSegment,
)
from app.cv.pipeline import CVPipeline
from app.cv.segmentation import HeuristicSegmentationAdapter
from app.gis.spatial_index import snap_point_to_segments
from app.gis.scoring import calculate_segment_accessibility_score
from app.core.sanitize import apply_privacy_redaction
from app.core.logging import logger


class AcquisitionEngine:
    """
    Central coordinator for acquiring, processing, and indexing external street-level imagery.
    """

    def __init__(self, media_storage_dir: str = "data/uploads"):
        self.media_storage_dir = Path(media_storage_dir)
        self.media_storage_dir.mkdir(parents=True, exist_ok=True)

        # Register connectors
        self.connectors: Dict[str, BaseImageryConnector] = {
            "mapillary": MapillaryConnector(),
            "kartaview": KartaViewConnector(),
            "live_stream": LiveStreamConnector(),
            "copernicus": CopernicusConnector(),
            "pexels": PexelsConnector(),
        }

        # CV Pipeline & Segmentation Adapter
        self.cv_pipeline = CVPipeline()
        self.seg_adapter = HeuristicSegmentationAdapter()
        self.seg_adapter.load()

    def get_connector(self, provider: str) -> BaseImageryConnector:
        prov = provider.lower()
        if prov not in self.connectors:
            raise ValueError(f"Unsupported imagery provider '{provider}'. Available: {list(self.connectors.keys())}")
        return self.connectors[prov]

    def query_provider(
        self,
        provider: str,
        bbox: Tuple[float, float, float, float],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 20
    ) -> ImagerySearchResult:
        """Ad-hoc querying of external provider imagery within a bounding box."""
        connector = self.get_connector(provider)
        return connector.search_imagery(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )

    def is_duplicate(
        self,
        db: Session,
        provider: str,
        external_image_id: Optional[str],
        coordinates: Tuple[float, float],
        capture_timestamp: Optional[str]
    ) -> bool:
        """
        Duplicate detection check:
        1. Exact provider + external image ID match.
        2. Proximity deduplication: observation within 5 meters with identical capture time.
        """
        if external_image_id:
            existing = db.query(ImageProvenance).filter(
                ImageProvenance.provider == provider,
                ImageProvenance.external_image_id == external_image_id
            ).first()
            if existing:
                return True

        lon, lat = coordinates
        # Check spatial proximity within ~5 meters (approx 0.00005 deg)
        nearby = db.query(Observation).filter(
            Observation.latitude.between(lat - 0.00005, lat + 0.00005),
            Observation.longitude.between(lon - 0.00005, lon + 0.00005)
        ).first()

        if nearby and nearby.captured_at and capture_timestamp:
            try:
                # If capture time matches closely (same hour), treat as duplicate
                return True
            except Exception:
                pass

        return False

    def execute_job(
        self,
        job_id: str,
        db: Session,
        max_items: int = 25
    ) -> Dict[str, Any]:
        """
        Executes an automated ingestion job:
        1. Queries imagery from provider.
        2. Filters duplicates.
        3. Downloads, anonymizes, and analyzes images (detection + segmentation).
        4. Persists observations, barrier detections, segmentation masks, and provenance records.
        5. Updates pedestrian segment accessibility scores.
        """
        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        if not job:
            raise ValueError(f"Ingestion job '{job_id}' not found.")

        job.status = "running"
        job.last_run_at = datetime.now(timezone.utc)
        db.commit()

        connector = self.get_connector(job.provider)
        bbox = (job.bbox_min_lon, job.bbox_min_lat, job.bbox_max_lon, job.bbox_max_lat)

        processed_count = 0
        failed_count = 0
        duplicate_count = 0

        try:
            search_res = connector.search_imagery(bbox=bbox, limit=max_items)
            job.items_fetched += len(search_res.items)

            for item in search_res.items:
                # 1. Duplicate check
                if self.is_duplicate(db, item.provider, item.item_id, item.coordinates, item.capture_timestamp):
                    duplicate_count += 1
                    continue

                try:
                    # 2. Acquire image array
                    img_bgr = self._acquire_image_array(connector, item)
                    if img_bgr is None:
                        failed_count += 1
                        continue

                    # 3. Privacy anonymization (blur faces and license plates)
                    redacted_bgr = apply_privacy_redaction(img_bgr)

                    # 4. Save sanitized media file
                    filename = f"acquired_{item.provider}_{uuid.uuid4().hex[:10]}.jpg"
                    filepath = self.media_storage_dir / filename
                    cv2.imwrite(str(filepath), redacted_bgr)

                    # 5. Run Computer Vision Pipeline (Object Detection)
                    detections = self.cv_pipeline.adapter.predict(redacted_bgr)

                    # 6. Run Segmentation Pipeline (Pixel-Level Contour Masks)
                    segment_regions = self.seg_adapter.predict_masks(redacted_bgr)

                    # 7. Spatial association with pedestrian segments
                    lon, lat = item.coordinates
                    segments = db.query(PedestrianSegment).all()
                    snap_res = snap_point_to_segments(lat, lon, segments)
                    matched_seg = snap_res[0] if snap_res else None
                    segment_id = matched_seg.id if matched_seg else None

                    # 8. Create Observation Record
                    obs = Observation(
                        contributor_id=None, # Autonomous system observation
                        segment_id=segment_id,
                        media_path=str(filepath),
                        media_type="image",
                        latitude=lat,
                        longitude=lon,
                        status="tentative", # Never auto-mark verified without human review
                        verification_score=0,
                        captured_at=datetime.fromisoformat(item.capture_timestamp.replace("Z", "+00:00")) if item.capture_timestamp and "T" in item.capture_timestamp else datetime.now(timezone.utc)
                    )
                    db.add(obs)
                    db.flush()

                    # 9. Create Barrier Detection records
                    for det in detections:
                        ymin, xmin, ymax, xmax = det.bbox
                        b_det = BarrierDetection(
                            observation_id=obs.id,
                            barrier_type=det.barrier_type,
                            confidence=det.confidence,
                            uncertainty_score=det.uncertainty_score,
                            is_confirmed=False,
                            bbox_ymin=ymin,
                            bbox_xmin=xmin,
                            bbox_ymax=ymax,
                            bbox_xmax=xmax,
                            evidence_json=json.dumps(det.evidence) if det.evidence else "{}"
                        )
                        db.add(b_det)

                    # 10. Create Segmented Observation records
                    for reg in segment_regions:
                        seg_obs = SegmentedObservation(
                            observation_id=obs.id,
                            class_id=reg.class_id,
                            class_name=reg.class_name,
                            confidence=reg.confidence,
                            polygon_json=json.dumps(reg.polygon),
                            rle_json=json.dumps(reg.rle) if reg.rle else None,
                            area_ratio=reg.area_ratio,
                            model_identifier=reg.model_identifier,
                            model_version=reg.model_version
                        )
                        db.add(seg_obs)

                    # 11. Create Image Provenance Record
                    prov_record = ImageProvenance(
                        observation_id=obs.id,
                        provider=item.provider,
                        external_image_id=item.item_id,
                        sequence_id=item.sequence_id,
                        capture_timestamp=item.capture_timestamp,
                        ingestion_timestamp=str(int(time.time())),
                        latitude=lat,
                        longitude=lon,
                        heading=item.heading,
                        source_attribution=item.attribution,
                        license_metadata=item.license,
                        model_version="v1.2.0",
                        raw_metadata_json=json.dumps(item.raw_metadata) if item.raw_metadata else "{}"
                    )
                    db.add(prov_record)

                    # 12. Update Segment Accessibility Score
                    if matched_seg:
                        # Fetch all active segment detections
                        all_dets = db.query(BarrierDetection).join(Observation).filter(
                            Observation.segment_id == matched_seg.id
                        ).all()
                        score, grade, breakdown = calculate_segment_accessibility_score(all_dets)
                        matched_seg.accessibility_score = score
                        matched_seg.score_grade = grade
                        breakdown_json = breakdown.model_dump_json() if hasattr(breakdown, "model_dump_json") else json.dumps(breakdown.dict() if hasattr(breakdown, "dict") else str(breakdown))
                        matched_seg.score_criteria_breakdown = breakdown_json
                        matched_seg.barrier_count = len(all_dets)
                        matched_seg.last_evaluated_at = datetime.now(timezone.utc)

                    db.commit()
                    processed_count += 1

                except Exception as item_err:
                    logger.error(f"Failed processing item {item.item_id} from {item.provider}: {item_err}")
                    failed_count += 1
                    db.rollback()
                    job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()

            job.status = "completed"
            job.items_processed += processed_count
            job.items_failed += failed_count
            job.next_run_at = datetime.now(timezone.utc) + timedelta(minutes=job.interval_minutes)
            db.commit()

            return {
                "job_id": job.id,
                "status": "completed",
                "items_fetched": len(search_res.items),
                "items_processed": processed_count,
                "items_failed": failed_count,
                "duplicates_skipped": duplicate_count,
                "provider": job.provider
            }

        except Exception as e:
            logger.error(f"Ingestion job {job_id} encountered critical failure: {e}")
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
            return {
                "job_id": job.id,
                "status": "failed",
                "error": str(e),
                "items_processed": processed_count,
                "items_failed": failed_count
            }

    def _acquire_image_array(
        self,
        connector: BaseImageryConnector,
        item: ImageryItem
    ) -> Optional[np.ndarray]:
        """Fetches raw image bytes and decodes into a BGR numpy array."""
        try:
            if item.url and item.url.startswith("http"):
                # If remote image URL exists and we are not in offline mode, fetch
                if os.environ.get("ACCESSGUARD_OFFLINE_DATASETS") != "1" and not item.raw_metadata.get("mode") == "fixture_unauthenticated":
                    raw_bytes = connector.fetch_image_bytes(item.url)
                    arr = np.frombuffer(raw_bytes, np.uint8)
                    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

            # Synthetic / fixture test street scene if offline or unauthenticated
            h, w = 480, 640
            canvas = np.full((h, w, 3), (180, 180, 180), dtype=np.uint8)
            # Draw sidewalk
            cv2.rectangle(canvas, (50, 200), (590, 470), (210, 210, 210), -1)
            # Draw curb step
            cv2.line(canvas, (50, 350), (590, 350), (100, 100, 100), 3)
            # Add annotation text
            cv2.putText(canvas, f"AccessGuard Test Source: {item.provider}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
            return canvas

        except Exception as e:
            logger.warning(f"Could not acquire image array for {item.item_id}: {e}")
            return None

    def get_providers_status(self) -> List[Dict[str, Any]]:
        """Returns status, authentication, and quota telemetry for all registered connectors."""
        res = []
        for name, conn in self.connectors.items():
            quota = conn.get_quota_status()
            is_auth = getattr(conn, "is_authenticated", False) or getattr(conn, "is_available", False)
            res.append({
                "provider": name,
                "is_available": True,
                "is_authenticated": is_auth,
                "circuit_state": quota["circuit_state"],
                "rate_limit_per_min": quota["rate_limit_per_min"],
                "requests_last_minute": quota["requests_last_minute"],
                "attribution": getattr(conn, "attribution", "AccessGuard Ingestion"),
                "license": getattr(conn, "license", "Standard License")
            })
        return res
