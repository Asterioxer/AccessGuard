"""
Pixel-Level Accessibility Segmentation Base Interface and Utilities.

Defines the SegmentationAdapter contract, SegmentedRegion dataclass,
RLE compression, and polygon contour extraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

# 10 Standard AccessGuard Accessibility Segmentation Classes
SEGMENTATION_CLASSES: List[str] = [
    "traversable_sidewalk",
    "non_traversable_sidewalk",
    "stairs",
    "curb_ramp",
    "damaged_pavement",
    "pedestrian_obstruction",
    "open_drain",
    "crosswalk",
    "roadway",
    "unknown_infrastructure",
]

CLASS_NAME_TO_ID: Dict[str, int] = {cls: idx for idx, cls in enumerate(SEGMENTATION_CLASSES)}
CLASS_ID_TO_NAME: Dict[int, str] = {idx: cls for idx, cls in enumerate(SEGMENTATION_CLASSES)}

# Visual palette for segmentation overlays (RGB)
CLASS_COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "traversable_sidewalk": (46, 204, 113),       # Green
    "non_traversable_sidewalk": (231, 76, 60),    # Red
    "stairs": (230, 126, 34),                     # Orange
    "curb_ramp": (52, 152, 219),                  # Blue
    "damaged_pavement": (155, 89, 182),           # Purple
    "pedestrian_obstruction": (241, 196, 15),     # Yellow
    "open_drain": (192, 57, 43),                  # Dark Red
    "crosswalk": (26, 188, 156),                  # Turquoise
    "roadway": (127, 140, 141),                   # Slate Gray
    "unknown_infrastructure": (149, 165, 166),    # Light Gray
}


@dataclass
class SegmentedRegion:
    """Represents a pixel-level segmented region with genuine contour polygons."""
    class_id: int
    class_name: str
    polygon: List[List[float]]             # Normalized [[x, y], ...] coordinates (0.0 to 1.0)
    confidence: float                      # 0.0 to 1.0
    bbox: Tuple[float, float, float, float]# (ymin, xmin, ymax, xmax) normalized
    area_pixels: int                       # Raw pixel count
    area_ratio: float                      # Fraction of total image area
    model_identifier: str
    model_version: str
    mask: Optional[np.ndarray] = None      # Binary uint8 mask (0 or 255)
    rle: Optional[Dict[str, Any]] = None   # Run Length Encoding dict: {"counts": [...], "size": [H, W]}


def mask_to_polygon(
    mask: np.ndarray,
    min_area_pixels: int = 40,
    simplify_epsilon: float = 0.003
) -> List[List[float]]:
    """
    Extracts smoothed, normalized polygon coordinates [[x, y], ...] from a binary mask.
    Filters out noise contours smaller than min_area_pixels.
    """
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return []

    # Ensure binary uint8
    bin_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # Pick largest external contour
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area_pixels:
        return []

    # Approximate and smooth polygon
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, simplify_epsilon * peri, True)

    polygon: List[List[float]] = []
    for pt in approx:
        px, py = pt[0]
        norm_x = max(0.0, min(1.0, float(px) / w))
        norm_y = max(0.0, min(1.0, float(py) / h))
        polygon.append([round(norm_x, 4), round(norm_y, 4)])

    return polygon


def mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """Encodes a binary mask into Run-Length Encoding (RLE)."""
    h, w = mask.shape[:2]
    flat = mask.flatten(order="F") # Fortran order column-first
    diffs = np.diff(flat, prepend=0)
    runs = np.where(diffs != 0)[0]

    counts: List[int] = []
    prev = 0
    for r in runs:
        counts.append(int(r - prev))
        prev = r
    counts.append(int(len(flat) - prev))

    return {
        "counts": counts,
        "size": [int(h), int(w)]
    }


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Decodes RLE back to a binary uint8 mask."""
    h, w = rle["size"]
    counts = rle["counts"]
    flat = np.zeros(h * w, dtype=np.uint8)
    val = 0
    curr = 0
    for count in counts:
        flat[curr:curr + count] = val
        curr += count
        val = 1 - val
    return flat.reshape((h, w), order="F") * 255


def polygon_to_mask(polygon: List[List[float]], height: int, width: int) -> np.ndarray:
    """Converts normalized polygon [[x, y], ...] into a binary uint8 mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygon or len(polygon) < 3:
        return mask

    pts = np.array([[int(p[0] * width), int(p[1] * height)] for p in polygon], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Computes exact Intersection over Union (IoU) between two binary masks."""
    if mask_a.shape != mask_b.shape:
        mask_b = cv2.resize(mask_b, (mask_a.shape[1], mask_a.shape[0]), interpolation=cv2.INTER_NEAREST)

    bin_a = (mask_a > 0)
    bin_b = (mask_b > 0)

    intersection = np.logical_and(bin_a, bin_b).sum()
    union = np.logical_or(bin_a, bin_b).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


class SegmentationAdapter(ABC):
    """Abstract Base Class for pixel-level accessibility segmentation adapters."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.supported_classes = SEGMENTATION_CLASSES
        self.is_loaded = False

    @abstractmethod
    def load(self) -> bool:
        """Initializes model weights, runtime sessions, or morphological kernels."""
        pass

    @abstractmethod
    def predict_masks(self, image_bgr: np.ndarray) -> List[SegmentedRegion]:
        """
        Runs segmentation inference on image array.
        Returns a list of SegmentedRegion objects with genuine pixel masks and polygons.
        """
        pass
