"""
Pixel-Level Accessibility Segmentation REST API.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, status
import cv2
import numpy as np

from app.cv.segmentation import (
    HeuristicSegmentationAdapter,
    SEGMENTATION_CLASSES,
    CLASS_COLOR_PALETTE,
    CLASS_NAME_TO_ID,
)
from app.core.logging import logger

router = APIRouter(prefix="/cv/segmentation", tags=["Segmentation"])
seg_adapter = HeuristicSegmentationAdapter()
seg_adapter.load()


@router.get("/classes")
def list_segmentation_classes():
    """Returns the 10 standard accessibility segmentation classes with color palette."""
    classes_info = []
    for cls in SEGMENTATION_CLASSES:
        rgb = CLASS_COLOR_PALETTE.get(cls, (128, 128, 128))
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        classes_info.append({
            "class_id": CLASS_NAME_TO_ID[cls],
            "class_name": cls,
            "color_rgb": rgb,
            "color_hex": hex_color
        })
    return {"classes": classes_info, "total_classes": len(classes_info)}


@router.post("/predict")
async def predict_segmentation_masks(
    file: UploadFile = File(...)
):
    """
    Computes genuine pixel-level segmentation masks and contour polygons
    for an uploaded streetscape image.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file uploaded.")

    arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    h, w = img_bgr.shape[:2]
    regions = seg_adapter.predict_masks(img_bgr)

    results = []
    for reg in regions:
        rgb = CLASS_COLOR_PALETTE.get(reg.class_name, (128, 128, 128))
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        results.append({
            "class_id": reg.class_id,
            "class_name": reg.class_name,
            "polygon": reg.polygon,
            "confidence": reg.confidence,
            "bbox": reg.bbox,
            "area_pixels": reg.area_pixels,
            "area_ratio": reg.area_ratio,
            "color_hex": hex_color,
            "color_rgb": rgb,
            "model_identifier": reg.model_identifier,
            "model_version": reg.model_version
        })

    return {
        "image_width": w,
        "image_height": h,
        "region_count": len(results),
        "regions": results,
        "model_identifier": seg_adapter.name,
        "model_version": seg_adapter.version
    }
