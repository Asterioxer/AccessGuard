"""
Dataset Manifest Management for Hugging Face and Local Accessibility Datasets.

Maintains formal records of dataset provenance, licensing, source attribution,
split distributions, and integrity checksums.
"""

import json
import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from app.core.logging import logger

class ClassDistribution(BaseModel):
    """Distribution of classes within a split."""
    class_counts: Dict[str, int] = Field(default_factory=dict)
    total_annotations: int = 0

class DatasetSplitInfo(BaseModel):
    """Metadata for a single dataset split."""
    num_samples: int = 0
    class_distribution: ClassDistribution = Field(default_factory=ClassDistribution)
    storage_bytes: int = 0

class DatasetManifest(BaseModel):
    """Full dataset manifest with provenance, licensing, and schema verification."""
    dataset_name: str
    hf_repo_id: Optional[str] = None
    license: str = "Unknown"
    source_attribution: str = ""
    paper_citation: Optional[str] = None
    created_at: str
    version: str = "1.0.0"
    splits: Dict[str, DatasetSplitInfo] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def save(self, output_path: str) -> None:
        """Persists the manifest JSON to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)
        logger.info(f"Dataset manifest written to {output_path}")

    @classmethod
    def load(cls, manifest_path: str) -> "DatasetManifest":
        """Loads a manifest JSON from disk."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
