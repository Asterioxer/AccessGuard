"""
Tests for TaxonomyNormalizer, category mapping, and coordinate normalization.
"""

import pytest
from app.cv.datasets.normalizer import TaxonomyNormalizer
from app.cv.datasets.schema import AccessGuardCategory, BoundingBoxFormat

def test_project_sidewalk_normalization():
    # Curb ramp
    cat, attrs = TaxonomyNormalizer.map_category("ProjectSidewalk", "CurbRamp")
    assert cat == AccessGuardCategory.CURB_RAMP
    assert attrs.get("wheelchair_friendly") is True

    # Missing curb ramp -> obstacle
    cat, attrs = TaxonomyNormalizer.map_category("ProjectSidewalk", "missing curb ramp")
    assert cat == AccessGuardCategory.OBSTACLE
    assert attrs.get("curb_height_issue") is True

    # Surface problem -> surface damage
    cat, attrs = TaxonomyNormalizer.map_category("ProjectSidewalk", "surface problem")
    assert cat == AccessGuardCategory.SURFACE_DAMAGE
    assert attrs.get("sidewalk_verified") is True

def test_road_damage_semantic_separation():
    """
    CRITICAL CONSTRAINT: Do not assume road cracks are equivalent to sidewalk accessibility defects.
    """
    # General roadway crack without sidewalk verification -> unknown_infrastructure
    cat, attrs = TaxonomyNormalizer.map_category("ShuoZheLi/SVRDD_YOLO", "D20", additional_context={"is_sidewalk": False})
    assert cat == AccessGuardCategory.UNKNOWN_INFRASTRUCTURE
    assert attrs.get("sidewalk_verified") is False
    assert attrs.get("roadway_only") is True

    # Road crack with explicit sidewalk verification -> surface_damage
    cat_sw, attrs_sw = TaxonomyNormalizer.map_category("ShuoZheLi/SVRDD_YOLO", "D20", additional_context={"is_sidewalk": True})
    assert cat_sw == AccessGuardCategory.SURFACE_DAMAGE
    assert attrs_sw.get("sidewalk_verified") is True

def test_coordinate_normalization():
    # YOLO format: (x_center=0.5, y_center=0.5, width=0.4, height=0.6)
    box = (0.5, 0.5, 0.4, 0.6)
    ymin, xmin, ymax, xmax = TaxonomyNormalizer.normalize_bbox(box, BoundingBoxFormat.YOLO_NORMALIZED, 640, 480)
    assert pytest.approx(xmin, 0.01) == 0.30
    assert pytest.approx(xmax, 0.01) == 0.70
    assert pytest.approx(ymin, 0.01) == 0.20
    assert pytest.approx(ymax, 0.01) == 0.80

    # COCO pixel format: (xmin=64, ymin=48, width=320, height=240) in 640x480
    coco_box = (64, 48, 320, 240)
    ymin, xmin, ymax, xmax = TaxonomyNormalizer.normalize_bbox(coco_box, BoundingBoxFormat.COCO_PIXELS, 640, 480)
    assert pytest.approx(xmin, 0.01) == 0.10
    assert pytest.approx(ymin, 0.01) == 0.10
    assert pytest.approx(xmax, 0.01) == 0.60
    assert pytest.approx(ymax, 0.01) == 0.60

def test_coordinate_clamping():
    # Out of bounds coordinates should be clamped to [0.0, 1.0]
    out_box = (-0.5, -0.2, 1.5, 1.8)
    ymin, xmin, ymax, xmax = TaxonomyNormalizer.normalize_bbox(out_box, BoundingBoxFormat.VOC_NORMALIZED, 640, 480)
    assert ymin == 0.0
    assert xmin == 0.0
    assert ymax == 1.0
    assert xmax == 1.0
