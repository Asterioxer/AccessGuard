"""
Pixel-Level Accessibility Segmentation Module.
"""

from app.cv.segmentation.base import (
    SegmentationAdapter,
    SegmentedRegion,
    SEGMENTATION_CLASSES,
    CLASS_COLOR_PALETTE,
    CLASS_NAME_TO_ID,
    CLASS_ID_TO_NAME,
    mask_to_polygon,
    mask_to_rle,
    rle_to_mask,
    polygon_to_mask,
    compute_mask_iou,
)
from app.cv.segmentation.heuristic_seg_adapter import HeuristicSegmentationAdapter
from app.cv.segmentation.onnx_seg_adapter import ONNXSegmentationAdapter
from app.cv.segmentation.yolo_seg_adapter import YOLOSegmentationAdapter

__all__ = [
    "SegmentationAdapter",
    "SegmentedRegion",
    "SEGMENTATION_CLASSES",
    "CLASS_COLOR_PALETTE",
    "CLASS_NAME_TO_ID",
    "CLASS_ID_TO_NAME",
    "mask_to_polygon",
    "mask_to_rle",
    "rle_to_mask",
    "polygon_to_mask",
    "compute_mask_iou",
    "HeuristicSegmentationAdapter",
    "ONNXSegmentationAdapter",
    "YOLOSegmentationAdapter",
]
