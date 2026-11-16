import uuid
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text
)
from sqlalchemy.orm import relationship
from app.db.session import Base

def generate_uuid() -> str:
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="contributor", nullable=False) # admin, verifier, contributor, public
    reputation_score = Column(Integer, default=10)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    observations = relationship("Observation", back_populates="contributor", cascade="all, delete-orphan")
    reviews = relationship("VerificationReview", back_populates="reviewer", cascade="all, delete-orphan")

class PedestrianSegment(Base):
    __tablename__ = "pedestrian_segments"
    
    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    osm_way_id = Column(String, nullable=True, index=True)
    name = Column(String, nullable=False, default="Pedestrian Sidewalk")
    
    # Coordinates in WGS84
    start_lat = Column(Float, nullable=False)
    start_lon = Column(Float, nullable=False)
    end_lat = Column(Float, nullable=False)
    end_lon = Column(Float, nullable=False)
    
    # GeoJSON coordinates list [[lon, lat], [lon, lat], ...]
    geometry_geojson = Column(Text, nullable=False)
    length_meters = Column(Float, default=50.0)
    
    # Transparent Accessibility Scoring (0.0 to 100.0)
    accessibility_score = Column(Float, default=50.0)
    score_grade = Column(String, default="U") # A, B, C, D, F, U (Unassessed)
    score_criteria_breakdown = Column(Text, default="{}") # JSON of penalty/bonus components
    barrier_count = Column(Integer, default=0)
    last_evaluated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    observations = relationship("Observation", back_populates="segment")

class Observation(Base):
    __tablename__ = "observations"
    
    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    contributor_id = Column(String, ForeignKey("users.id"), nullable=True)
    segment_id = Column(String, ForeignKey("pedestrian_segments.id"), nullable=True)
    
    media_path = Column(String, nullable=False)
    media_type = Column(String, nullable=False, default="image") # image, video
    
    latitude = Column(Float, nullable=True, index=True)
    longitude = Column(Float, nullable=True, index=True)
    
    # Status lifecycle: tentative -> verified | disputed | rejected
    status = Column(String, default="tentative", index=True)
    verification_score = Column(Integer, default=0)
    
    captured_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    contributor = relationship("User", back_populates="observations")
    segment = relationship("PedestrianSegment", back_populates="observations")
    detections = relationship("BarrierDetection", back_populates="observation", cascade="all, delete-orphan")
    reviews = relationship("VerificationReview", back_populates="observation", cascade="all, delete-orphan")

class BarrierDetection(Base):
    __tablename__ = "barrier_detections"
    
    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    observation_id = Column(String, ForeignKey("observations.id"), nullable=False)
    
    # Types: stairs, curb_ramp_missing, curb_ramp_present, surface_damage,
    # construction_obstacle, narrow_sidewalk, steep_incline, pole_or_bollard
    barrier_type = Column(String, nullable=False, index=True)
    
    # Mathematical confidence and uncertainty metrics
    confidence = Column(Float, nullable=False) # 0.0 to 1.0
    uncertainty_score = Column(Float, default=0.0) # 0.0 to 1.0
    is_confirmed = Column(Boolean, default=False)
    
    # Normalized bounding box coordinates [ymin, xmin, ymax, xmax] (0.0 to 1.0)
    bbox_ymin = Column(Float, nullable=False)
    bbox_xmin = Column(Float, nullable=False)
    bbox_ymax = Column(Float, nullable=False)
    bbox_xmax = Column(Float, nullable=False)
    
    # Polygon mask coordinates (JSON string of [[x, y], ...])
    polygon_json = Column(Text, nullable=True)
    
    # Concrete evidence explaining prediction (step count, roughness, clearance width, etc.)
    evidence_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    observation = relationship("Observation", back_populates="detections")

class VerificationReview(Base):
    __tablename__ = "verification_reviews"
    
    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    observation_id = Column(String, ForeignKey("observations.id"), nullable=False)
    reviewer_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    vote = Column(String, nullable=False) # confirm, dispute, reject
    suggested_barrier_type = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    observation = relationship("Observation", back_populates="reviews")
    reviewer = relationship("User", back_populates="reviews")

class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    provider = Column(String, nullable=False, index=True) # mapillary, kartaview, copernicus, live_stream, pexels
    status = Column(String, default="pending", index=True) # pending, running, completed, failed, cancelled
    priority = Column(String, default="normal") # high, normal, low

    bbox_min_lon = Column(Float, nullable=False)
    bbox_min_lat = Column(Float, nullable=False)
    bbox_max_lon = Column(Float, nullable=False)
    bbox_max_lat = Column(Float, nullable=False)

    interval_minutes = Column(Integer, default=60)
    items_fetched = Column(Integer, default=0)
    items_processed = Column(Integer, default=0)
    items_failed = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)

    last_run_at = Column(DateTime, nullable=True)
    next_run_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    config_json = Column(Text, default="{}")

class ImageProvenance(Base):
    __tablename__ = "image_provenance"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    observation_id = Column(String, ForeignKey("observations.id"), nullable=True, index=True)
    provider = Column(String, nullable=False, index=True)
    external_image_id = Column(String, nullable=True, index=True)
    sequence_id = Column(String, nullable=True)

    capture_timestamp = Column(String, nullable=True)
    ingestion_timestamp = Column(String, nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    heading = Column(Float, nullable=True)

    source_attribution = Column(String, default="")
    license_metadata = Column(String, default="")
    model_version = Column(String, default="v1.2.0")
    raw_metadata_json = Column(Text, default="{}")

class SegmentedObservation(Base):
    __tablename__ = "segmented_observations"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    observation_id = Column(String, ForeignKey("observations.id"), nullable=False, index=True)

    class_id = Column(Integer, nullable=False)
    class_name = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)

    polygon_json = Column(Text, nullable=False) # JSON array of [[x, y], ...]
    rle_json = Column(Text, nullable=True)      # RLE mask representation if dense
    area_ratio = Column(Float, default=0.0)

    model_identifier = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class RealityTestRun(Base):
    __tablename__ = "reality_test_runs"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    test_suite_name = Column(String, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    scorecard_json = Column(Text, nullable=False)
    passed = Column(Boolean, default=False)
    summary_markdown = Column(Text, default="")

