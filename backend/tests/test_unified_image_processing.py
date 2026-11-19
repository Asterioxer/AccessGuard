"""
Unit & Integration Tests for Unified Inference & Evidence Integrity (Mission 006.2).

Validates:
1. Canonical 15-stage ImageProcessingService lifecycle.
2. Dual SHA-256 calculation (original & sanitized).
3. Strict Genuine vs Development Fallback execution policy enforcement.
4. Semantic segmentation decoupling and honest UNAVAILABLE reporting.
5. Class-aware spatial clustering preventing conflicting class collapse.
6. Elimination of synthetic (0.60, 0.20, 0.95, 0.80) surface damage bounding boxes.
7. Ingress integration (user upload & autonomous ingestion) using unified service.
8. Preservation of missing coordinates without false GPS defaulting.
"""

import os
import json
import uuid
import hashlib
import numpy as np
import cv2
import pytest

from app.cv.orchestration.image_processing_service import (
    ImageProcessingService,
    ImageProcessingRequest,
    ImageProcessingResult,
    ModelExecutionRecord,
)
from app.cv.adapters.base_v2 import ExecutionPolicy, AdapterStatus
from app.cv.base import DetectedBarrier
from app.cv.adapters.base_v2 import PointKeypoint
from app.cv.fusion.spatial_association import (
    are_classes_compatible,
    cluster_spatial_observations,
    calculate_iou,
)
from app.cv.fusion.evidence_fusion import EvidenceFusionEngine
from app.db.models import Observation, BarrierDetection, ImageProvenance, PedestrianSegment
from app.ingestion.engine import AcquisitionEngine
from app.ingestion.connectors.base import ImageryItem, ImagerySearchResult


def get_real_or_synthetic_image_bytes():
    """Returns real demo image bytes if present, or synthetic image bytes."""
    for cand in ("data/processed/demo_stairs.jpg", "data/benchmarks/indian_streetscapes/scenes/scene_001.jpg"):
        if os.path.exists(cand):
            with open(cand, "rb") as f:
                return f.read(), os.path.basename(cand)
    img = np.full((320, 320, 3), 160, dtype=np.uint8)
    for y in [80, 140, 200, 260]:
        cv2.line(img, (20, y), (300, y), (40, 40, 40), 3)
    _, enc = cv2.imencode(".jpg", img)
    return enc.tobytes(), "synthetic_test.jpg"


def test_unified_service_lifecycle_and_provenance(db_session):
    """Verifies complete 15-stage pipeline, dual SHA-256 calculation, and database persistence."""
    img_bytes, filename = get_real_or_synthetic_image_bytes()
    orig_hash = hashlib.sha256(img_bytes).hexdigest()

    service = ImageProcessingService()
    req = ImageProcessingRequest(
        image_bytes=img_bytes,
        filename=filename,
        source="test_runner",
        source_metadata={"test": "lifecycle_test"},
        latitude=28.6139,
        longitude=77.2090,
        execution_policy=ExecutionPolicy.DEVELOPMENT_FALLBACK,
        enable_privacy_redaction=True,
        save_media=True
    )

    result = service.process_image(req, db=db_session)

    # 1. Pipeline containers
    assert result.run_id.startswith("proc_")
    assert result.original_sha256 == orig_hash
    assert len(result.sanitized_sha256) == 64
    assert result.overall_status in ("SUCCESS", "PARTIAL")

    # 2. Specialist execution records
    adapters_recorded = [r.adapter_name for r in result.model_execution_records]
    assert "yolo_adapter" in adapters_recorded
    assert "rampnet_adapter" in adapters_recorded
    assert "guidecane_adapter" in adapters_recorded
    assert "citysurfaces_adapter" in adapters_recorded

    # 3. Telemetry metrics
    assert result.resource_telemetry["total_wall_time_ms"] > 0
    assert "ram_used_mb" in result.resource_telemetry

    # 4. Database persistence
    assert result.observation_id is not None
    obs = db_session.query(Observation).filter(Observation.id == result.observation_id).first()
    assert obs is not None
    assert obs.latitude == pytest.approx(28.6139, 0.0001)

    prov = db_session.query(ImageProvenance).filter(ImageProvenance.observation_id == obs.id).first()
    assert prov is not None
    meta = json.loads(prov.raw_metadata_json)
    assert meta["original_sha256"] == orig_hash
    assert meta["run_id"] == result.run_id


def test_strict_genuine_vs_development_fallback_policies():
    """Verifies strict policy forbids heuristic fallbacks while fallback policy marks them DEGRADED."""
    img_bytes, filename = get_real_or_synthetic_image_bytes()
    service = ImageProcessingService()

    # 1. STRICT_GENUINE mode
    strict_req = ImageProcessingRequest(
        image_bytes=img_bytes,
        filename=filename,
        execution_policy=ExecutionPolicy.STRICT_GENUINE,
        enable_privacy_redaction=False,
        save_media=False
    )
    strict_res = service.process_image(strict_req)

    # No model record in STRICT mode may claim heuristic_fallback as successful model inference
    for record in strict_res.model_execution_records:
        if record.execution_type == "heuristic_fallback":
            pytest.fail(f"Adapter {record.adapter_name} executed heuristic_fallback in STRICT_GENUINE mode!")
        if record.status == "READY":
            assert record.execution_type == "model_inference"

    # Segmentation in strict mode without weights must report UNAVAILABLE
    if service.seg_adapter.model is None:
        assert strict_res.segmentation_results.status == "UNAVAILABLE"
        assert strict_res.segmentation_results.execution_type == "none"
        assert len(strict_res.segmentation_results.regions) == 0

    # 2. DEVELOPMENT_FALLBACK mode
    dev_req = ImageProcessingRequest(
        image_bytes=img_bytes,
        filename=filename,
        execution_policy=ExecutionPolicy.DEVELOPMENT_FALLBACK,
        enable_privacy_redaction=False,
        save_media=False
    )
    dev_res = service.process_image(dev_req)
    assert dev_res.segmentation_results.status in ("READY", "DEGRADED")
    if dev_res.segmentation_results.status == "DEGRADED":
        assert dev_res.segmentation_results.execution_type == "heuristic_overlay"


def test_spatial_clustering_class_compatibility():
    """Verifies class-aware clustering prevents merging contradictory or independent physical objects."""
    # 1. Compatibility rules
    assert are_classes_compatible("stairs", "steps") is True
    assert are_classes_compatible("curb_ramp_present", "curb_ramp_missing") is True
    assert are_classes_compatible("stairs", "pole_or_bollard") is False
    assert are_classes_compatible("surface_damage", "trash_can") is False

    # 2. Spatial clustering with identical bbox but incompatible classes
    bbox = (0.50, 0.40, 0.70, 0.60)
    det_stairs = ("yolo", DetectedBarrier(
        barrier_type="stairs",
        confidence=0.85,
        uncertainty_score=0.15,
        bbox=bbox
    ))
    det_pole = ("guidecane", DetectedBarrier(
        barrier_type="pole_or_bollard",
        confidence=0.80,
        uncertainty_score=0.20,
        bbox=bbox
    ))

    clusters = cluster_spatial_observations(
        detections_by_adapter=[det_stairs, det_pole],
        keypoints_by_adapter=[],
        iou_threshold=0.40
    )

    # Incompatible classes must NOT be merged into a single cluster!
    assert len(clusters) == 2, f"Expected 2 separate clusters for incompatible classes, got {len(clusters)}"


def test_surface_roughness_no_synthetic_bbox():
    """Verifies elimination of hardcoded (0.60, 0.20, 0.95, 0.80) bounding box."""
    from app.cv.adapters.base_v2 import AdapterInferenceResult, AdapterStatus
    engine = EvidenceFusionEngine()

    # Create dummy CitySurfaces result with high roughness in metadata but NO spatial mask
    dummy_city = AdapterInferenceResult(
        status=AdapterStatus.READY,
        adapter_name="citysurfaces_adapter",
        metadata={"roughness_score": 0.85},
        mask=None # No spatial mask
    )

    fused = engine.fuse([dummy_city])
    assert fused.surface_roughness == 0.85

    # Verify NO synthetic bounding box (0.60, 0.20, 0.95, 0.80) was injected
    for b in fused.fused_barriers:
        if b.canonical_type == "surface_damage":
            assert b.bbox != (0.60, 0.20, 0.95, 0.80), "Found forbidden synthetic bbox (0.60, 0.20, 0.95, 0.80)!"


def test_upload_endpoint_preserves_missing_coordinates(client, contributor_token):
    """Verifies user uploads without GPS never default to San Francisco (37.7765, -122.4194)."""
    img_bytes, filename = get_real_or_synthetic_image_bytes()

    response = client.post(
        "/api/v1/observations/upload",
        files={"file": (filename, img_bytes, "image/jpeg")},
        data={"adapter": "heuristic"},
        headers={"Authorization": f"Bearer {contributor_token}"}
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    # Missing coordinates must remain None honestly!
    assert data["latitude"] is None
    assert data["longitude"] is None


def test_ingestion_engine_uses_unified_service(db_session, monkeypatch):
    """Verifies external acquisition engine executes through canonical ImageProcessingService."""
    engine = AcquisitionEngine()
    test_img = np.full((256, 256, 3), 140, dtype=np.uint8)
    monkeypatch.setattr(engine, "_acquire_image_array", lambda conn, it: test_img)

    mock_item = ImageryItem(
        item_id="unified_ingest_test_001",
        provider="kartaview",
        coordinates=(77.2100, 28.6140),
        capture_timestamp="2026-11-01T10:00:00Z",
        url="https://kartaview.org/test.jpg",
        license="CC-BY-SA-4.0",
        attribution="KartaView Contributor",
        is_live=False,
        raw_metadata={"mode": "test"}
    )
    monkeypatch.setattr(
        engine.get_connector("kartaview"),
        "search_imagery",
        lambda bbox, limit=25: ImagerySearchResult(
            items=[mock_item],
            total_count=1,
            provider="kartaview",
            query_bbox=bbox,
            status="success"
        )
    )

    from app.db.models import IngestionJob
    job = IngestionJob(
        provider="kartaview",
        bbox_min_lon=77.20,
        bbox_min_lat=28.60,
        bbox_max_lon=77.22,
        bbox_max_lat=28.62,
        interval_minutes=60,
        priority="high"
    )
    db_session.add(job)
    db_session.commit()

    res = engine.execute_job(job.id, db_session, max_items=1)
    assert res["status"] == "completed"
    assert res["items_processed"] == 1

    # Verify provenance was saved by unified service with run_id and models_executed
    prov = db_session.query(ImageProvenance).filter(
        ImageProvenance.external_image_id == f"acquired_kartaview_{mock_item.item_id}.jpg"
    ).first()
    assert prov is not None
    meta = json.loads(prov.raw_metadata_json)
    assert "run_id" in meta
    assert "models_executed" in meta
    assert "original_sha256" in meta


def test_model_checkpoints_integrity_and_hashes():
    """Validates existence, exact SHA-256 hashes, and clean initialization of real models."""
    from app.cv.adapters.rampnet_adapter import RampNetAdapter
    from app.cv.adapters.yolo_adapter import YOLOAdapter
    from app.cv.segmentation.yolo_seg_adapter import YOLOSegmentationAdapter

    # 1. RampNet Checkpoint (360MB safetensors present locally; gitignored in CI checkout)
    ramp_path = "data/models/rampnet.safetensors"
    if os.path.exists(ramp_path):
        ramp_hasher = hashlib.sha256()
        with open(ramp_path, "rb") as f:
            while chunk := f.read(65536):
                ramp_hasher.update(chunk)
        ramp_sha = ramp_hasher.hexdigest()
        assert ramp_sha == "f2119e3becb0b551fa1470f7b7ba85b82122a3f73a6ed2a85609dd57617866b5"

        ramp_adapter = RampNetAdapter(model_path=ramp_path)
        assert ramp_adapter.load() is True
        assert ramp_adapter._status == AdapterStatus.READY
        assert ramp_adapter._weight_sha256 == ramp_sha
    else:
        ramp_adapter = RampNetAdapter(model_path=ramp_path)
        assert ramp_adapter.load(allow_fallback=False) is False
        assert ramp_adapter._status == AdapterStatus.UNAVAILABLE

    # 2. YOLO Detection Checkpoint
    yolo_path = "data/models/yolov8n.onnx"
    assert os.path.exists(yolo_path), f"YOLO ONNX checkpoint missing at {yolo_path}"
    yolo_hasher = hashlib.sha256()
    with open(yolo_path, "rb") as f:
        while chunk := f.read(65536):
            yolo_hasher.update(chunk)
    yolo_sha = yolo_hasher.hexdigest()
    assert yolo_sha == "fbb11393b8823050ff79f2ed837eeeab4c51e9d9220da80fd773d94ed7531d76"

    yolo_adapter = YOLOAdapter(model_path=yolo_path)
    assert yolo_adapter.load() is True
    assert yolo_adapter._status == AdapterStatus.READY
    assert yolo_adapter._weight_sha256 == yolo_sha

    # 3. YOLO Segmentation Checkpoint (tracked in git)
    seg_path = "data/models/yolov8n-seg.pt"
    assert os.path.exists(seg_path), f"YOLO segmentation checkpoint missing at {seg_path}"
    seg_hasher = hashlib.sha256()
    with open(seg_path, "rb") as f:
        while chunk := f.read(65536):
            seg_hasher.update(chunk)
    seg_sha = seg_hasher.hexdigest()
    assert seg_sha == "a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c"

    seg_adapter = YOLOSegmentationAdapter(model_path=seg_path)
    loaded = seg_adapter.load()
    if loaded:
        assert seg_adapter.model is not None
        assert seg_adapter._weight_sha256 == seg_sha
    else:
        assert seg_adapter.status == SegmentationStatus.UNAVAILABLE


def test_real_image_strict_genuine_model_execution():
    """Executes canonical service on real streetscape image under STRICT_GENUINE policy without synthetic fallback."""
    real_img_path = "data/processed/demo_stairs.jpg"
    assert os.path.exists(real_img_path), f"Real demo image missing at {real_img_path}"
    with open(real_img_path, "rb") as f:
        img_bytes = f.read()

    service = ImageProcessingService()
    req = ImageProcessingRequest(
        image_bytes=img_bytes,
        filename="demo_stairs.jpg",
        source="reality_audit",
        execution_policy=ExecutionPolicy.STRICT_GENUINE,
        enable_privacy_redaction=True,
        save_media=False
    )
    result = service.process_image(req)

    assert result.overall_status in ("SUCCESS", "PARTIAL")
    records_by_name = {r.adapter_name: r for r in result.model_execution_records}

    # YOLO executed genuine model inference with non-zero latency
    yolo_rec = records_by_name["yolo_adapter"]
    assert yolo_rec.status == "READY"
    assert yolo_rec.execution_type == "model_inference"
    assert yolo_rec.execution_time_ms > 0
    assert yolo_rec.weight_sha256 is not None

    # RampNet executed genuine model inference with non-zero latency when present; honest UNAVAILABLE when absent
    ramp_rec = records_by_name["rampnet_adapter"]
    if os.path.exists("data/models/rampnet.safetensors"):
        assert ramp_rec.status == "READY"
        assert ramp_rec.execution_type == "model_inference"
        assert ramp_rec.execution_time_ms > 0
        assert ramp_rec.weight_sha256 is not None
    else:
        assert ramp_rec.status == "UNAVAILABLE"
        assert ramp_rec.execution_type == "unavailable"

    # Missing models honestly report UNAVAILABLE and never execute heuristic fallback
    guide_rec = records_by_name["guidecane_adapter"]
    assert guide_rec.status == "UNAVAILABLE"
    assert guide_rec.execution_type == "unavailable"

    city_rec = records_by_name["citysurfaces_adapter"]
    assert city_rec.status == "UNAVAILABLE"
    assert city_rec.execution_type == "unavailable"

    # Segmentation adapter
    if result.segmentation_results.status == "READY":
        assert result.segmentation_results.execution_type == "model_inference"
    else:
        assert result.segmentation_results.status == "UNAVAILABLE"
        assert result.segmentation_results.execution_type == "unavailable"


def test_strict_genuine_mode_forbids_heuristic_fallback():
    """Asserts that in STRICT_GENUINE mode, missing weights produce UNAVAILABLE rather than heuristics."""
    from app.cv.adapters.rampnet_adapter import RampNetAdapter
    # Missing weights with allow_fallback=False
    adapter = RampNetAdapter(model_path="data/models/nonexistent_model.safetensors")
    dummy_img = np.full((128, 128, 3), 128, dtype=np.uint8)
    res = adapter.predict_v2(dummy_img, allow_fallback=False)

    assert res.status == AdapterStatus.UNAVAILABLE
    assert res.execution_type == "unavailable"
    assert len(res.keypoints) == 0


def test_video_ingestion_unification_and_temporal_tracking(tmp_path, db_session):
    """Verifies video frames are routed individually through ImageProcessingService with temporal tracking."""
    video_file = tmp_path / "test_street_walk.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_file), fourcc, 30.0, (320, 240))
    # Generate 60 frames (2 seconds @ 30 fps)
    for i in range(60):
        frame = np.full((240, 320, 3), 140, dtype=np.uint8)
        # Draw horizontal stair lines
        for y in (80, 120, 160):
            cv2.line(frame, (40, y), (280, y), (20, 20, 20), 3)
        writer.write(frame)
    writer.release()

    service = ImageProcessingService()
    vid_res = service.process_video(
        video_path=video_file,
        source="test_video_source",
        filename="test_street_walk.mp4",
        latitude=28.6139,
        longitude=77.2090,
        sample_fps=1.0,
        max_duration_seconds=10,
        execution_policy=ExecutionPolicy.STRICT_GENUINE,
        db=db_session,
        save_media=True
    )

    assert vid_res.overall_status in ("SUCCESS", "PARTIAL")
    assert vid_res.total_frames == 60
    assert vid_res.sampled_frames == 2
    assert len(vid_res.frame_results) == 2

    # Every sampled frame followed the canonical ImageProcessingResult contract
    for fr in vid_res.frame_results:
        assert isinstance(fr, ImageProcessingResult)
        assert fr.run_id.startswith("proc_")
        assert len(fr.original_sha256) == 64
        assert len(fr.sanitized_sha256) == 64

    # DB persistence check
    assert vid_res.observation_id is not None
    obs = db_session.query(Observation).filter(Observation.id == vid_res.observation_id).first()
    assert obs is not None
    assert obs.media_type == "video"
    assert obs.latitude == pytest.approx(28.6139, 0.0001)

    prov = db_session.query(ImageProvenance).filter(ImageProvenance.observation_id == obs.id).first()
    assert prov is not None
    p_meta = json.loads(prov.raw_metadata_json)
    assert p_meta["original_sha256"] == vid_res.original_sha256
    assert p_meta["total_frames"] == 60
    assert p_meta["sampled_frames"] == 2


def test_video_upload_endpoint_unification(client, contributor_token, tmp_path):
    """Verifies upload endpoint handles video uploads via canonical unified service."""
    video_file = tmp_path / "upload_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_file), fourcc, 30.0, (160, 120))
    for i in range(30):
        frame = np.full((120, 160, 3), 100, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    with open(video_file, "rb") as f:
        vid_bytes = f.read()

    response = client.post(
        "/api/v1/observations/upload",
        files={"file": ("upload_test.mp4", vid_bytes, "video/mp4")},
        data={"latitude": 28.6129, "longitude": 77.2295},
        headers={"Authorization": f"Bearer {contributor_token}"}
    )
    assert response.status_code == 201, f"Upload failed: {response.text}"
    data = response.json()
    assert "id" in data
    assert data["media_type"] == "video"
    assert data["latitude"] == pytest.approx(28.6129, 0.0001)
    assert data["longitude"] == pytest.approx(77.2295, 0.0001)
