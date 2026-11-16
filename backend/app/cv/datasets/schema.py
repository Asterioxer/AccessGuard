"""
AccessGuard Unified Annotation Schema for Accessibility Datasets.

Defines standardized data models for bounding boxes, annotations, sample metadata,
and the AccessGuard category taxonomy.
"""

from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field

class AccessGuardCategory(str, Enum):
    """
    Standardized AccessGuard accessibility categories.
    Only classes with compatible semantics are mapped into primary barrier/feature categories.
    """
    CURB_RAMP = "curb_ramp"
    STAIRS = "stairs"
    SURFACE_DAMAGE = "surface_damage"
    OBSTACLE = "obstacle"
    CROSSWALK = "crosswalk"
    UNKNOWN_INFRASTRUCTURE = "unknown_infrastructure"

class BoundingBoxFormat(str, Enum):
    """Supported bounding box coordinate formats."""
    # Normalized [ymin, xmin, ymax, xmax] in [0.0, 1.0]
    VOC_NORMALIZED = "voc_normalized"
    # Normalized [x_center, y_center, width, height] in [0.0, 1.0] (YOLO standard)
    YOLO_NORMALIZED = "yolo_normalized"
    # Absolute pixel [xmin, ymin, width, height] (COCO standard)
    COCO_PIXELS = "coco_pixels"
    # Absolute pixel [xmin, ymin, xmax, ymax] (Pascal VOC pixels)
    VOC_PIXELS = "voc_pixels"

class NormalizedAnnotation(BaseModel):
    """Unified annotation representation across all external datasets."""
    category: AccessGuardCategory = Field(..., description="AccessGuard standardized category")
    source_dataset: str = Field(..., description="Original dataset identifier / HF repo")
    source_label: str = Field(..., description="Original raw label from the source dataset")
    # Normalized (ymin, xmin, ymax, xmax) in [0.0, 1.0]
    bbox: Tuple[float, float, float, float] = Field(..., description="(ymin, xmin, ymax, xmax) in [0.0, 1.0]")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Annotation confidence score")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Metadata preserving source domain tags")
    is_verified: bool = Field(default=False, description="Whether human consensus verified this label")

    def to_yolo_normalized(self) -> Tuple[float, float, float, float]:
        """Convert [ymin, xmin, ymax, xmax] to YOLO [x_center, y_center, width, height]."""
        ymin, xmin, ymax, xmax = self.bbox
        width = max(0.0, xmax - xmin)
        height = max(0.0, ymax - ymin)
        x_center = xmin + width / 2.0
        y_center = ymin + height / 2.0
        return (x_center, y_center, width, height)

class NormalizedDatasetSample(BaseModel):
    """A single normalized image sample with unified annotations."""
    sample_id: str
    dataset_name: str
    split: str = "train"
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    annotations: List[NormalizedAnnotation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
