"""
AccessGuard CV Datasets Package.
Provides dataset schemas, storage safety checks, Hugging Face ingestion,
and semantic normalization pipelines.
"""

from app.cv.datasets.schema import (
    AccessGuardCategory,
    BoundingBoxFormat,
    NormalizedAnnotation,
    NormalizedDatasetSample
)
from app.cv.datasets.storage_safety import check_storage_safety
from app.cv.datasets.normalizer import TaxonomyNormalizer
from app.cv.datasets.manifest import DatasetManifest
from app.cv.datasets.hf_loader import HuggingFaceDatasetLoader

__all__ = [
    "AccessGuardCategory",
    "BoundingBoxFormat",
    "NormalizedAnnotation",
    "NormalizedDatasetSample",
    "check_storage_safety",
    "TaxonomyNormalizer",
    "DatasetManifest",
    "HuggingFaceDatasetLoader"
]
