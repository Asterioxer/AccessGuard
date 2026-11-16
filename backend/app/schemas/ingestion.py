"""
Pydantic Schemas for External Imagery Ingestion & Provenance.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict


class IngestionJobCreate(BaseModel):
    provider: str = Field(..., description="Imagery provider name (mapillary, kartaview, copernicus, live_stream, pexels)")
    bbox: Tuple[float, float, float, float] = Field(..., description="(min_lon, min_lat, max_lon, max_lat)")
    interval_minutes: int = Field(60, ge=5, le=1440, description="Periodic ingestion interval in minutes")
    priority: str = Field("normal", description="Job priority: high, normal, low")
    start_date: Optional[str] = Field(None, description="ISO timestamp start filter")
    end_date: Optional[str] = Field(None, description="ISO timestamp end filter")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Provider-specific parameters")


class IngestionJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    provider: str
    status: str
    priority: str
    bbox: Tuple[float, float, float, float]
    interval_minutes: int
    items_fetched: int
    items_processed: int
    items_failed: int
    error_message: Optional[str] = None
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    created_at: datetime


class AdHocImageryQuery(BaseModel):
    provider: str
    bbox: Tuple[float, float, float, float]
    limit: int = Field(20, ge=1, le=100)
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class ProvenanceRecordSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    observation_id: Optional[str] = None
    provider: str
    external_image_id: Optional[str] = None
    sequence_id: Optional[str] = None
    capture_timestamp: Optional[str] = None
    ingestion_timestamp: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    heading: Optional[float] = None
    source_attribution: str
    license_metadata: str
    model_version: str = "v1.2.0"
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionProviderInfo(BaseModel):
    provider: str
    is_available: bool
    is_authenticated: bool
    circuit_state: str
    rate_limit_per_min: int
    requests_last_minute: int
    attribution: str
    license: str
