"""
Isolated Synthetic Development Fixtures for Offline Testing & CI.

CRITICAL POLICY:
These fixtures are strictly for hermetic unit testing and CI development.
They must carry data_origin = 'synthetic_fixture' and must NEVER enter
the real-world benchmark or be represented as externally acquired field data.
"""

from typing import List, Optional
from app.cv.datasets.schema import (
    AccessGuardCategory,
    BoundingBoxFormat,
    NormalizedAnnotation,
    NormalizedDatasetSample
)
from app.cv.datasets.normalizer import TaxonomyNormalizer


class SyntheticBenchmarkFixtureGenerator:
    """
    Generates isolated, reproducible synthetic test fixtures.
    Strictly isolated from external real-data pipelines.
    """

    @staticmethod
    def generate_fixture_sample(
        dataset_name: str,
        split: str,
        index: int,
        width: int = 640,
        height: int = 480
    ) -> NormalizedDatasetSample:
        """
        Produces a synthetic sample with explicit synthetic metadata.
        """
        sample_id = f"synthetic_fixture_{dataset_name.split('/')[-1]}_{split}_{index:03d}"
        annotations: List[NormalizedAnnotation] = []
        clean_name = dataset_name.lower()

        if "obstacle" in clean_name:
            label = "pole" if index % 2 == 0 else "stairs"
            box = (0.35, 0.40, 0.75, 0.60)
            ann = TaxonomyNormalizer.create_normalized_annotation(
                dataset_name="synthetic_fixture",
                source_label=label,
                raw_bbox=box,
                bbox_format=BoundingBoxFormat.VOC_NORMALIZED,
                img_width=width,
                img_height=height
            )
            annotations.append(ann)
        elif "ramp" in clean_name or "sidewalk" in clean_name:
            label = "curb_ramp" if index % 2 == 0 else "missing curb ramp"
            box = (0.50, 0.20, 0.85, 0.55)
            ann = TaxonomyNormalizer.create_normalized_annotation(
                dataset_name="synthetic_fixture",
                source_label=label,
                raw_bbox=box,
                bbox_format=BoundingBoxFormat.VOC_NORMALIZED,
                img_width=width,
                img_height=height
            )
            annotations.append(ann)
        else:
            ann = TaxonomyNormalizer.create_normalized_annotation(
                dataset_name="synthetic_fixture",
                source_label="stairs",
                raw_bbox=(0.40, 0.30, 0.80, 0.70),
                bbox_format=BoundingBoxFormat.VOC_NORMALIZED,
                img_width=width,
                img_height=height
            )
            annotations.append(ann)

        return NormalizedDatasetSample(
            sample_id=sample_id,
            dataset_name=f"fixture::{dataset_name}",
            split=split,
            data_origin="synthetic_fixture",
            license="MIT / Synthetic Test Fixture",
            attribution="AccessGuard Synthetic Test Generator (Hermetic Testing Only)",
            image_width=width,
            image_height=height,
            annotations=annotations,
            metadata={
                "data_origin": "synthetic_fixture",
                "is_synthetic": True,
                "reproducible_seed": index,
                "warning": "NOT REAL-WORLD DATA; STRICTLY FOR OFFLINE UNIT TESTS"
            }
        )
