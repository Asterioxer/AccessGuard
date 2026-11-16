"""
Spatial Association & Clustering for AccessGuard Multi-Model Evidence Fusion.
Associates heterogeneous detections (bounding boxes, point keypoints) across coordinate spaces.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.cv.base import DetectedBarrier
from app.cv.adapters.base_v2 import PointKeypoint


@dataclass
class SpatialCluster:
    """A group of spatially proximate detections and keypoints across models."""
    cluster_id: int
    detections: List[Tuple[str, DetectedBarrier]] = field(default_factory=list) # (adapter_name, barrier)
    keypoints: List[Tuple[str, PointKeypoint]] = field(default_factory=list)   # (adapter_name, keypoint)

    @property
    def consolidated_bbox(self) -> Tuple[float, float, float, float]:
        """Calculates confidence-weighted bounding box centroid."""
        if not self.detections:
            if not self.keypoints:
                return (0.0, 0.0, 0.0, 0.0)
            # Synthesize box around keypoints
            xs = [kp.x for _, kp in self.keypoints]
            ys = [kp.y for _, kp in self.keypoints]
            mean_x = float(np.mean(xs))
            mean_y = float(np.mean(ys))
            r = 0.04
            return (
                max(0.0, round(mean_y - r, 4)),
                max(0.0, round(mean_x - r, 4)),
                min(1.0, round(mean_y + r, 4)),
                min(1.0, round(mean_x + r, 4))
            )

        total_weight = 0.0
        weighted_ymin = 0.0
        weighted_xmin = 0.0
        weighted_ymax = 0.0
        weighted_xmax = 0.0

        for _, d in self.detections:
            w = max(0.1, d.confidence)
            ymin, xmin, ymax, xmax = d.bbox
            weighted_ymin += ymin * w
            weighted_xmin += xmin * w
            weighted_ymax += ymax * w
            weighted_xmax += xmax * w
            total_weight += w

        return (
            round(weighted_ymin / total_weight, 4),
            round(weighted_xmin / total_weight, 4),
            round(weighted_ymax / total_weight, 4),
            round(weighted_xmax / total_weight, 4)
        )


def calculate_iou(
    boxA: Tuple[float, float, float, float],
    boxB: Tuple[float, float, float, float]
) -> float:
    """
    Computes Intersection-over-Union (IoU) for two normalized boxes (ymin, xmin, ymax, xmax).
    """
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    inter_area = max(0.0, yB - yA) * max(0.0, xB - xA)
    if inter_area <= 0.0:
        return 0.0

    boxA_area = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxB_area = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def point_near_box(
    point: PointKeypoint,
    box: Tuple[float, float, float, float],
    margin: float = 0.04
) -> bool:
    """Checks if a point keypoint falls within or adjacent to a bounding box."""
    ymin, xmin, ymax, xmax = box
    if (ymin - margin) <= point.y <= (ymax + margin) and (xmin - margin) <= point.x <= (xmax + margin):
        return True
    
    # Check center distance
    center_y = (ymin + ymax) / 2.0
    center_x = (xmin + xmax) / 2.0
    dist = np.sqrt((point.x - center_x) ** 2 + (point.y - center_y) ** 2)
    return dist <= 0.08


def cluster_spatial_observations(
    detections_by_adapter: List[Tuple[str, DetectedBarrier]],
    keypoints_by_adapter: List[Tuple[str, PointKeypoint]],
    iou_threshold: float = 0.40
) -> List[SpatialCluster]:
    """
    Clusters bounding boxes and keypoints from multiple adapters into unified SpatialClusters.
    """
    clusters: List[SpatialCluster] = []
    cluster_counter = 0

    # 1. Cluster bounding boxes
    for adapter_name, barrier in detections_by_adapter:
        assigned = False
        for cluster in clusters:
            # Check against existing detections in cluster
            for _, existing_barrier in cluster.detections:
                iou = calculate_iou(barrier.bbox, existing_barrier.bbox)
                if iou >= iou_threshold:
                    cluster.detections.append((adapter_name, barrier))
                    assigned = True
                    break
            if assigned:
                break

        if not assigned:
            cluster_counter += 1
            new_cluster = SpatialCluster(cluster_id=cluster_counter)
            new_cluster.detections.append((adapter_name, barrier))
            clusters.append(new_cluster)

    # 2. Associate keypoints with existing clusters or create point clusters
    for adapter_name, kp in keypoints_by_adapter:
        assigned = False
        for cluster in clusters:
            if cluster.detections:
                for _, existing_barrier in cluster.detections:
                    if point_near_box(kp, existing_barrier.bbox):
                        cluster.keypoints.append((adapter_name, kp))
                        assigned = True
                        break
            elif cluster.keypoints:
                # Keypoint to keypoint distance
                for _, existing_kp in cluster.keypoints:
                    dist = np.sqrt((kp.x - existing_kp.x) ** 2 + (kp.y - existing_kp.y) ** 2)
                    if dist <= 0.06:
                        cluster.keypoints.append((adapter_name, kp))
                        assigned = True
                        break
            if assigned:
                break

        if not assigned:
            cluster_counter += 1
            new_cluster = SpatialCluster(cluster_id=cluster_counter)
            new_cluster.keypoints.append((adapter_name, kp))
            clusters.append(new_cluster)

    return clusters
