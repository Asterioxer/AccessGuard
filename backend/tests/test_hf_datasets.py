"""
Tests for Hugging Face dataset ingestion, streaming, storage safety, and manifests.
Verifies strict elimination of silent synthetic fallback and placeholder generation.
"""

import os
import tempfile
import pytest
from app.cv.datasets.hf_loader import HuggingFaceDatasetLoader, DatasetAcquisitionError
from app.cv.datasets.storage_safety import check_storage_safety
from app.cv.datasets.manifest import DatasetManifest, DatasetSplitInfo, ClassDistribution
from app.cv.datasets.schema import BoundingBoxFormat

def test_storage_safety_checks():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Requesting a normal small amount should pass
        is_safe, msg = check_storage_safety(tmp_dir, requested_mb=50.0, min_free_gb=0.1)
        assert is_safe is True
        assert "Storage safe" in msg

        # Requesting impossibly huge amount should fail safety check
        is_safe, msg = check_storage_safety(tmp_dir, requested_mb=100_000_000.0, min_free_gb=1.0)
        assert is_safe is False
        assert "Storage safety check failed" in msg

def test_inspect_dataset_schema():
    loader = HuggingFaceDatasetLoader()
    schema = loader.inspect_dataset_schema("Abtinzandi/Obstacle-Detection-Dataset-YOLO", force_offline=True)
    assert schema["dataset_name"] == "Abtinzandi/Obstacle-Detection-Dataset-YOLO"
    assert "metadata" in schema
    assert schema["metadata"]["license"] == "CC-BY-4.0"
    assert schema["streaming_supported"] is True

def test_stream_offline_no_silent_synthetic_substitution():
    """Verify that offline streaming NEVER silently substitutes synthetic samples into real data paths."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        loader = HuggingFaceDatasetLoader(cache_dir=tmp_dir)
        samples = list(loader.stream_samples(
            "Abtinzandi/Obstacle-Detection-Dataset-YOLO",
            split="train",
            max_samples=5,
            force_offline=True,
            allow_fixtures=False
        ))
        assert len(samples) == 0

def test_stream_fixtures_explicit_tagging():
    """Verify that development fixtures are explicitly marked with data_origin='synthetic_fixture'."""
    loader = HuggingFaceDatasetLoader()
    samples = list(loader.stream_samples(
        "Abtinzandi/Obstacle-Detection-Dataset-YOLO",
        split="train",
        max_samples=3,
        force_offline=True,
        allow_fixtures=True
    ))
    assert len(samples) == 3
    for s in samples:
        assert s.data_origin == "synthetic_fixture"
        assert s.metadata.get("is_synthetic") is True
        assert s.image_width > 0
        assert s.image_height > 0
        assert len(s.annotations) > 0
        for ann in s.annotations:
            ymin, xmin, ymax, xmax = ann.bbox
            assert 0.0 <= ymin <= ymax <= 1.0
            assert 0.0 <= xmin <= xmax <= 1.0

def test_manifest_creation_and_loading():
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest = DatasetManifest(
            dataset_name="Test-Accessibility-Dataset",
            hf_repo_id="test/accessibility-dataset",
            license="CC-BY-4.0",
            source_attribution="Test Attribution",
            created_at="2026-11-16T00:00:00Z",
            splits={
                "train": DatasetSplitInfo(
                    num_samples=10,
                    class_distribution=ClassDistribution(
                        class_counts={"stairs": 6, "curb_ramp": 4},
                        total_annotations=10
                    )
                )
            }
        )
        out_path = os.path.join(tmp_dir, "manifest.json")
        manifest.save(out_path)
        assert os.path.exists(out_path)

        loaded = DatasetManifest.load(out_path)
        assert loaded.dataset_name == "Test-Accessibility-Dataset"
        assert loaded.splits["train"].num_samples == 10
        assert loaded.splits["train"].class_distribution.class_counts["stairs"] == 6

def test_materialize_refuses_fake_placeholders():
    """Verify that download_and_materialize fails rather than writing fake grey placeholders."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        loader = HuggingFaceDatasetLoader(cache_dir=tmp_dir)
        # Attempting to materialize when offline without fixtures must fail, not write grey images
        with pytest.raises(RuntimeError) as exc_info:
            loader.download_and_materialize(
                "Abtinzandi/Obstacle-Detection-Dataset-YOLO",
                split="train",
                max_samples=2,
                force_offline=True,
                allow_fixtures=False
            )
        assert "Stream yielded 0 samples" in str(exc_info.value)
