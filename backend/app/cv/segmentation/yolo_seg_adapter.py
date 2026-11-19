"""
Ultralytics YOLO Segmentation Model Adapter.

Runs native YOLOv8/v11 segmentation inference with instance polygon extraction,
mask confidence estimation, and class taxonomy normalization.
"""

import os
import hashlib
from typing import List, Optional
import cv2
import numpy as np

from app.cv.segmentation.base import (
    SegmentationAdapter,
    SegmentedRegion,
    mask_to_polygon,
    mask_to_rle,
    CLASS_NAME_TO_ID,
)
from app.cv.adapters.base_v2 import AdapterStatus
from app.cv.segmentation.heuristic_seg_adapter import HeuristicSegmentationAdapter
from app.core.logging import logger


class YOLOSegmentationAdapter(SegmentationAdapter):
    """
    Ultralytics YOLO segmentation model adapter.
    Produces pixel-level masks and boundary polygons for accessibility barriers.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.35
    ):
        super().__init__(name="yolo_segmentation_adapter", version="1.1.0")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._weight_sha256: Optional[str] = None
        self._device: str = "cpu"
        self._fallback = HeuristicSegmentationAdapter()

    def load(self) -> bool:
        try:
            from ultralytics import YOLO
            seg_default = "data/models/yolov8n-seg.pt"
            weights = self.model_path if self.model_path and os.path.exists(self.model_path) else (seg_default if os.path.exists(seg_default) else None)

            if weights:
                self.model = YOLO(weights)
                hasher = hashlib.sha256()
                with open(weights, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                self._weight_sha256 = hasher.hexdigest()
                try:
                    self._device = str(next(self.model.model.parameters()).device)
                except Exception:
                    self._device = "cpu"
                logger.info(f"Loaded YOLO segmentation model from {weights} (sha256: {self._weight_sha256[:12]}...)")
                self.is_loaded = True
                return True
            else:
                self.model = None
                self._weight_sha256 = None
                self._device = "cpu"
                self.is_loaded = False
                logger.info("YOLOSegmentationAdapter weights not found; model marked unavailable.")
                return False
        except Exception as e:
            logger.warning(f"Could not load YOLO segmentation model: {e}")
            self.model = None
            self.is_loaded = False
            return False

    @property
    def status(self) -> AdapterStatus:
        if self.is_loaded and self.model is not None:
            return AdapterStatus.READY
        return AdapterStatus.UNAVAILABLE

    def predict_masks(self, image_bgr: np.ndarray) -> List[SegmentedRegion]:
        if not self.is_loaded:
            self.load()

        if self.model is None:
            return self._fallback.predict_masks(image_bgr)

        h, w = image_bgr.shape[:2]
        total_pixels = h * w

        try:
            results = self.model.predict(
                source=image_bgr,
                conf=self.confidence_threshold,
                device="cpu",
                verbose=False
            )

            regions: List[SegmentedRegion] = []
            if results and len(results) > 0:
                res = results[0]
                if res.masks is not None and res.boxes is not None:
                    boxes = res.boxes
                    masks_xy = res.masks.xy # List of polygon arrays in pixel coordinates

                    names = self.model.names if hasattr(self.model, "names") else {}
                    for i, poly_pts in enumerate(masks_xy):
                        if len(poly_pts) < 3:
                            continue

                        cls_id = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        raw_name = names.get(cls_id, f"class_{cls_id}").lower()

                        # Taxonomy mapping
                        if "stair" in raw_name or "step" in raw_name:
                            mapped_class = "stairs"
                        elif "ramp" in raw_name:
                            mapped_class = "curb_ramp"
                        elif "pothole" in raw_name or "crack" in raw_name or "damage" in raw_name:
                            mapped_class = "damaged_pavement"
                        elif "sidewalk" in raw_name or "walkway" in raw_name:
                            mapped_class = "traversable_sidewalk"
                        elif "road" in raw_name or "street" in raw_name:
                            mapped_class = "roadway"
                        else:
                            mapped_class = "pedestrian_obstruction"

                        # Normalize polygon points
                        norm_poly = [[round(float(pt[0]) / w, 4), round(float(pt[1]) / h, 4)] for pt in poly_pts]
                        x_coords = [p[0] for p in norm_poly]
                        y_coords = [p[1] for p in norm_poly]
                        bbox = (min(y_coords), min(x_coords), max(y_coords), max(x_coords))

                        # Create binary mask for RLE computation
                        bin_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(bin_mask, [np.array(poly_pts, dtype=np.int32)], 255)
                        area = int(cv2.countNonZero(bin_mask))

                        regions.append(
                            SegmentedRegion(
                                class_id=CLASS_NAME_TO_ID.get(mapped_class, 9),
                                class_name=mapped_class,
                                polygon=norm_poly,
                                confidence=round(conf, 4),
                                bbox=bbox,
                                area_pixels=area,
                                area_ratio=round(float(area) / total_pixels, 4),
                                model_identifier=self.name,
                                model_version=self.version,
                                mask=bin_mask,
                                rle=mask_to_rle(bin_mask)
                            )
                        )

            if not regions:
                return self._fallback.predict_masks(image_bgr)
            return regions
        except Exception as e:
            logger.warning(f"Error in YOLO segmentation prediction: {e}")
            return self._fallback.predict_masks(image_bgr)
