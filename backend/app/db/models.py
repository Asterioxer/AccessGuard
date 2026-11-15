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
    accessibility_score = Column(Float, default=100.0)
    score_grade = Column(String, default="A") # A, B, C, D, F
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
