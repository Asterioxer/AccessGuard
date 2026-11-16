"""
Taxonomy Normalizer for Accessibility Datasets.

Maps heterogeneous class taxonomies from Hugging Face and academic datasets into
standardized AccessGuard categories with semantic verification and metadata preservation.
"""

from typing import Dict, Any, Tuple, Optional
from app.cv.datasets.schema import (
    AccessGuardCategory,
    NormalizedAnnotation,
    BoundingBoxFormat
)
from app.core.logging import logger

class TaxonomyNormalizer:
    """
    Normalizes dataset-specific class labels and bounding box formats
    into AccessGuard standardized annotations.
    """

    # ProjectSidewalk category mapping
    PROJECT_SIDEWALK_MAP: Dict[str, Tuple[AccessGuardCategory, Dict[str, Any]]] = {
        "curb ramp": (AccessGuardCategory.CURB_RAMP, {"pedestrian_facility": "ramp", "wheelchair_friendly": True}),
        "curb_ramp": (AccessGuardCategory.CURB_RAMP, {"pedestrian_facility": "ramp", "wheelchair_friendly": True}),
        "curbramp": (AccessGuardCategory.CURB_RAMP, {"pedestrian_facility": "ramp", "wheelchair_friendly": True}),
        "ramp": (AccessGuardCategory.CURB_RAMP, {"pedestrian_facility": "ramp", "wheelchair_friendly": True}),
        "missing curb ramp": (AccessGuardCategory.OBSTACLE, {"barrier_nature": "missing_curb_ramp", "curb_height_issue": True}),
        "no curb ramp": (AccessGuardCategory.OBSTACLE, {"barrier_nature": "missing_curb_ramp", "curb_height_issue": True}),
        "nocurbramp": (AccessGuardCategory.OBSTACLE, {"barrier_nature": "missing_curb_ramp", "curb_height_issue": True}),
        "obstacle": (AccessGuardCategory.OBSTACLE, {"barrier_nature": "sidewalk_obstruction", "wheelchair_friendly": False}),
        "surface problem": (AccessGuardCategory.SURFACE_DAMAGE, {"sidewalk_verified": True, "wheelchair_vibration_hazard": True}),
        "surface_problem": (AccessGuardCategory.SURFACE_DAMAGE, {"sidewalk_verified": True, "wheelchair_vibration_hazard": True}),
        "crosswalk": (AccessGuardCategory.CROSSWALK, {"pedestrian_crossing": True}),
        "occlusion": (AccessGuardCategory.UNKNOWN_INFRASTRUCTURE, {"view_blocked": True, "evaluable": False}),
        "stairs": (AccessGuardCategory.STAIRS, {"barrier_nature": "staircase", "step_barrier": True, "wheelchair_friendly": False}),
        "stair": (AccessGuardCategory.STAIRS, {"barrier_nature": "staircase", "step_barrier": True, "wheelchair_friendly": False})
    }

    # Abtinzandi/Obstacle-Detection-Dataset-YOLO mapping
    OBSTACLE_DATASET_MAP: Dict[str, Tuple[AccessGuardCategory, Dict[str, Any]]] = {
        "obstacle": (AccessGuardCategory.OBSTACLE, {"barrier_nature": "physical_blockage"}),
        "pole": (AccessGuardCategory.OBSTACLE, {"barrier_nature": "utility_pole_obstruction"}),
        "bollard": (AccessGuardCategory.OBSTACLE, {"barrier_nature": "bollard", "clearance_check_needed": True}),
        "stairs": (AccessGuardCategory.STAIRS, {"step_barrier": True, "wheelchair_friendly": False}),
        "steps": (AccessGuardCategory.STAIRS, {"step_barrier": True, "wheelchair_friendly": False}),
        "barrier": (AccessGuardCategory.OBSTACLE, {"barrier_nature": "construction_or_security_barrier"}),
        "pothole": (AccessGuardCategory.SURFACE_DAMAGE, {"sidewalk_verified": False, "trip_hazard": True}),
        "person": (AccessGuardCategory.UNKNOWN_INFRASTRUCTURE, {"dynamic_entity": True, "evaluable": False}),
        "vehicle": (AccessGuardCategory.UNKNOWN_INFRASTRUCTURE, {"dynamic_entity": True, "evaluable": False})
    }

    # SVRDD (Road Damage Detection) mapping
    # CRITICAL: Roadway asphalt cracks are NOT assumed to be equivalent to sidewalk defects!
    SVRDD_MAP: Dict[str, Tuple[AccessGuardCategory, Dict[str, Any]]] = {
        "d00": (AccessGuardCategory.UNKNOWN_INFRASTRUCTURE, {"road_damage_type": "longitudinal_crack", "roadway_only": True, "sidewalk_verified": False}),
        "d10": (AccessGuardCategory.UNKNOWN_INFRASTRUCTURE, {"road_damage_type": "transverse_crack", "roadway_only": True, "sidewalk_verified": False}),
        "d20": (AccessGuardCategory.UNKNOWN_INFRASTRUCTURE, {"road_damage_type": "alligator_crack", "roadway_only": True, "sidewalk_verified": False}),
        "d40": (AccessGuardCategory.UNKNOWN_INFRASTRUCTURE, {"road_damage_type": "pothole", "roadway_only": True, "sidewalk_verified": False}),
        "d44": (AccessGuardCategory.CROSSWALK, {"road_feature": "crosswalk_blur", "roadway_only": True, "sidewalk_verified": False}),
        "repair": (AccessGuardCategory.UNKNOWN_INFRASTRUCTURE, {"road_feature": "surface_patch", "roadway_only": True, "sidewalk_verified": False})
    }

    @classmethod
    def normalize_bbox(
        cls,
        raw_bbox: Tuple[float, float, float, float],
        format: BoundingBoxFormat,
        img_width: int,
        img_height: int
    ) -> Tuple[float, float, float, float]:
        """
        Converts any bounding box format into AccessGuard normalized [ymin, xmin, ymax, xmax] in [0.0, 1.0].
        Clamps values to [0.0, 1.0].
        """
        if img_width <= 0 or img_height <= 0:
            raise ValueError(f"Invalid image dimensions: {img_width}x{img_height}")

        if format == BoundingBoxFormat.VOC_NORMALIZED:
            ymin, xmin, ymax, xmax = raw_bbox
        elif format == BoundingBoxFormat.YOLO_NORMALIZED:
            # YOLO format: x_center, y_center, width, height (all normalized)
            xc, yc, w, h = raw_bbox
            xmin = xc - w / 2.0
            xmax = xc + w / 2.0
            ymin = yc - h / 2.0
            ymax = yc + h / 2.0
        elif format == BoundingBoxFormat.COCO_PIXELS:
            # COCO format: xmin, ymin, width, height (in pixels)
            x, y, w, h = raw_bbox
            xmin = x / img_width
            ymin = y / img_height
            xmax = (x + w) / img_width
            ymax = (y + h) / img_height
        elif format == BoundingBoxFormat.VOC_PIXELS:
            # Pascal VOC pixels: xmin, ymin, xmax, ymax
            x1, y1, x2, y2 = raw_bbox
            xmin = x1 / img_width
            ymin = y1 / img_height
            xmax = x2 / img_width
            ymax = y2 / img_height
        else:
            raise ValueError(f"Unsupported bounding box format: {format}")

        # Clamp to [0.0, 1.0] and ensure ymin <= ymax, xmin <= xmax
        ymin_c = max(0.0, min(1.0, float(min(ymin, ymax))))
        ymax_c = max(0.0, min(1.0, float(max(ymin, ymax))))
        xmin_c = max(0.0, min(1.0, float(min(xmin, xmax))))
        xmax_c = max(0.0, min(1.0, float(max(xmin, xmax))))

        return (round(ymin_c, 5), round(xmin_c, 5), round(ymax_c, 5), round(xmax_c, 5))

    @classmethod
    def map_category(
        cls,
        dataset_name: str,
        source_label: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[AccessGuardCategory, Dict[str, Any]]:
        """
        Maps a source label from a given dataset into an AccessGuard category.
        Preserves original raw labels and attaches context attributes.
        """
        clean_name = dataset_name.lower()
        clean_label = str(source_label).strip().lower()
        context = additional_context or {}

        # 1. ProjectSidewalk or RampNet datasets
        if "projectsidewalk" in clean_name or "rampnet" in clean_name:
            if clean_label in cls.PROJECT_SIDEWALK_MAP:
                cat, attrs = cls.PROJECT_SIDEWALK_MAP[clean_label]
                merged = {**attrs, **context, "original_source_label": source_label}
                return cat, merged
            # Partial substring matching for variations
            if "ramp" in clean_label and "no" not in clean_label and "missing" not in clean_label:
                return AccessGuardCategory.CURB_RAMP, {"pedestrian_facility": "ramp", "original_source_label": source_label}
            if "stair" in clean_label or "step" in clean_label:
                return AccessGuardCategory.STAIRS, {"step_barrier": True, "original_source_label": source_label}

        # 2. Obstacle Detection YOLO dataset
        elif "obstacle" in clean_name:
            if clean_label in cls.OBSTACLE_DATASET_MAP:
                cat, attrs = cls.OBSTACLE_DATASET_MAP[clean_label]
                merged = {**attrs, **context, "original_source_label": source_label}
                return cat, merged

        # 3. SVRDD (Road Damage Detection)
        elif "svrdd" in clean_name or "rdd" in clean_name or "damage" in clean_name:
            # Check if this crack is specifically on a pedestrian walkway/sidewalk
            is_sidewalk = context.get("is_sidewalk", False) or context.get("sidewalk_verified", False)
            if clean_label in cls.SVRDD_MAP:
                cat, attrs = cls.SVRDD_MAP[clean_label]
                # Only elevate to surface_damage if explicitly on a sidewalk!
                if is_sidewalk:
                    return AccessGuardCategory.SURFACE_DAMAGE, {
                        **attrs,
                        "sidewalk_verified": True,
                        "original_source_label": source_label
                    }
                else:
                    return cat, {
                        **attrs,
                        "sidewalk_verified": False,
                        "original_source_label": source_label
                    }

        # Fallback heuristic mapping for generic terms
        if clean_label in ["curb_ramp", "curbramp", "ramp"]:
            return AccessGuardCategory.CURB_RAMP, {"inferred": True, "original_source_label": source_label}
        elif clean_label in ["stairs", "staircase", "steps"]:
            return AccessGuardCategory.STAIRS, {"inferred": True, "original_source_label": source_label}
        elif clean_label in ["surface_damage", "broken_sidewalk", "cracked_footpath"]:
            return AccessGuardCategory.SURFACE_DAMAGE, {"inferred": True, "original_source_label": source_label}
        elif clean_label in ["obstacle", "pole", "bollard", "barrier", "obstruction"]:
            return AccessGuardCategory.OBSTACLE, {"inferred": True, "original_source_label": source_label}
        elif clean_label in ["crosswalk", "pedestrian_crossing", "zebra_crossing"]:
            return AccessGuardCategory.CROSSWALK, {"inferred": True, "original_source_label": source_label}

        # Unknown or incompatible
        return AccessGuardCategory.UNKNOWN_INFRASTRUCTURE, {
            "unmapped_source_label": source_label,
            "compatible_semantics": False,
            "original_source_label": source_label
        }

    @classmethod
    def create_normalized_annotation(
        cls,
        dataset_name: str,
        source_label: str,
        raw_bbox: Tuple[float, float, float, float],
        bbox_format: BoundingBoxFormat,
        img_width: int,
        img_height: int,
        confidence: float = 1.0,
        additional_context: Optional[Dict[str, Any]] = None,
        is_verified: bool = False
    ) -> NormalizedAnnotation:
        """Helper to create a fully validated NormalizedAnnotation."""
        norm_bbox = cls.normalize_bbox(raw_bbox, bbox_format, img_width, img_height)
        category, attributes = cls.map_category(dataset_name, source_label, additional_context)
        return NormalizedAnnotation(
            category=category,
            source_dataset=dataset_name,
            source_label=source_label,
            bbox=norm_bbox,
            confidence=confidence,
            attributes=attributes,
            is_verified=is_verified
        )
