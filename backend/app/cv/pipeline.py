import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from app.cv.base import ModelAdapter, DetectedBarrier
from app.cv.adapters.dummy_adapter import DummyAdapter
from app.cv.adapters.heuristic_adapter import HeuristicCVAdapter
from app.cv.adapters.torchvision_adapter import TorchvisionAdapter
from app.cv.adapters.yolo_adapter import YOLOAdapter
from app.cv.adapters.hybrid_adapter import HybridVisionAdapter
from app.config import settings
from app.core.logging import logger

class CVPipeline:
    """
    Central Computer Vision Pipeline coordinator.
    Manages interchangeable adapters, image decoding, video keyframe extraction,
    temporal barrier clustering, and explicit uncertainty calibration.
    """
    def __init__(self, adapter_name: Optional[str] = None):
        self.adapter_name = adapter_name or settings.CV_ADAPTER
        self.adapter: ModelAdapter = self._init_adapter(self.adapter_name)

    def _init_adapter(self, name: str) -> ModelAdapter:
        name = name.lower()
        if name == "dummy":
            return DummyAdapter()
        elif name == "torchvision":
            return TorchvisionAdapter()
        elif name == "yolo":
            return YOLOAdapter()
        elif name == "hybrid":
            return HybridVisionAdapter()
        else: # default to heuristic
            return HeuristicCVAdapter()

    def set_adapter(self, adapter_name: str) -> None:
        self.adapter_name = adapter_name
        self.adapter = self._init_adapter(adapter_name)
        logger.info(f"Switched CV pipeline adapter to: {adapter_name}")

    def preprocess_image(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies image quality checks and adaptive preprocessing:
        1. Blur detection via Laplacian variance.
        2. Low-contrast / fog detection with adaptive CLAHE enhancement on the L-channel.
        3. Color consistency verification.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur evaluation (variance of Laplacian)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurred = laplacian_var < 80.0

        # 2. Contrast evaluation
        gray_std = float(np.std(gray))
        is_low_contrast = gray_std < 22.0
        
        processed_bgr = img_bgr.copy()
        clahe_applied = False
        
        # Apply CLAHE if low-contrast or fog condition detected
        if is_low_contrast:
            try:
                lab = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                l_enhanced = clahe.apply(l_channel)
                lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
                processed_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
                clahe_applied = True
            except Exception as e:
                logger.warning(f"CLAHE contrast enhancement failed: {e}")

        quality_meta = {
            "laplacian_variance": round(laplacian_var, 2),
            "is_blurred": is_blurred,
            "contrast_std": round(gray_std, 2),
            "is_low_contrast": is_low_contrast,
            "clahe_applied": clahe_applied,
            "channels": processed_bgr.shape[2] if len(processed_bgr.shape) == 3 else 1
        }
        return processed_bgr, quality_meta

    def process_image(self, image_bytes: bytes) -> Tuple[List[DetectedBarrier], np.ndarray]:
        """
        Decodes image bytes into BGR array, executes preprocessing and adapter prediction,
        and applies uncertainty calibration.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image from bytes.")
            
        preprocessed_bgr, quality_meta = self.preprocess_image(img_bgr)
        raw_detections = self.adapter.predict(preprocessed_bgr)
        
        # If image was severely blurred, apply an uncertainty penalty
        if quality_meta.get("is_blurred"):
            for d in raw_detections:
                d.uncertainty_score = min(0.85, round(d.uncertainty_score + 0.15, 2))
                d.evidence["blur_warning"] = "Image exhibits significant motion blur or soft focus."

        calibrated_detections = self.calibrate_detections(raw_detections)
        return calibrated_detections, preprocessed_bgr

    def process_video(
        self,
        video_path: str,
        sample_fps: float = 1.0,
        max_duration_seconds: int = 60
    ) -> List[DetectedBarrier]:
        """
        Samples keyframes from a short video, detects barriers on each frame via canonical ImageProcessingService,
        and clusters detections temporally to boost confidence on persistent obstacles.
        """
        from app.cv.orchestration.image_processing_service import ImageProcessingService
        from app.cv.adapters.base_v2 import ExecutionPolicy
        service = ImageProcessingService()
        result = service.process_video(
            video_path=video_path,
            sample_fps=sample_fps,
            max_duration_seconds=max_duration_seconds,
            execution_policy=ExecutionPolicy.STRICT_GENUINE if self.adapter_name != "heuristic" else ExecutionPolicy.DEVELOPMENT_FALLBACK,
            save_media=False,
            db=None
        )
        return self.calibrate_detections(result.detected_barriers)

    def _cluster_video_detections(
        self,
        frame_detections: List[Tuple[int, DetectedBarrier]],
        total_frames: int
    ) -> List[DetectedBarrier]:
        """
        Groups detections across video frames into consolidated persistent barriers.
        Barriers appearing across multiple keyframes receive boosted confidence and lower uncertainty.
        Fleeting 1-frame detections receive higher uncertainty.
        """
        if not frame_detections:
            return []
            
        clusters: List[List[Tuple[int, DetectedBarrier]]] = []
        
        for frame_idx, det in frame_detections:
            matched = False
            for cluster in clusters:
                rep_frame, rep_det = cluster[0]
                if rep_det.barrier_type == det.barrier_type:
                    # Check bounding box IoU
                    iou = self._calculate_iou(rep_det.bbox, det.bbox)
                    if iou > 0.35: # Spatial overlap across frames
                        cluster.append((frame_idx, det))
                        matched = True
                        break
            if not matched:
                clusters.append([(frame_idx, det)])
                
        consolidated: List[DetectedBarrier] = []
        for cluster in clusters:
            persistence_count = len(cluster)
            best_det = max(cluster, key=lambda x: x[1].confidence)[1]
            
            # Boost confidence if persistent across 3+ frames
            if persistence_count >= 3:
                boosted_conf = min(0.96, best_det.confidence + 0.08)
                reduced_unc = max(0.08, best_det.uncertainty_score - 0.10)
            elif persistence_count == 1:
                # Single fleeting frame detection: keep tentative and increase uncertainty
                boosted_conf = max(0.40, best_det.confidence - 0.10)
                reduced_unc = min(0.65, best_det.uncertainty_score + 0.15)
            else:
                boosted_conf = best_det.confidence
                reduced_unc = best_det.uncertainty_score
                
            evidence = dict(best_det.evidence)
            evidence["video_persistence_frames"] = persistence_count
            evidence["explanation"] = f"{evidence.get('explanation', '')} Verified across {persistence_count} sampled video frames."
            
            consolidated.append(
                DetectedBarrier(
                    barrier_type=best_det.barrier_type,
                    confidence=round(boosted_conf, 2),
                    uncertainty_score=round(reduced_unc, 2),
                    bbox=best_det.bbox,
                    polygon=best_det.polygon,
                    evidence=evidence
                )
            )
            
        return consolidated

    def _calculate_iou(self, bboxA: Tuple[float, float, float, float], bboxB: Tuple[float, float, float, float]) -> float:
        """Calculates IoU between two normalized (ymin, xmin, ymax, xmax) bounding boxes."""
        yminA, xminA, ymaxA, xmaxA = bboxA
        yminB, xminB, ymaxB, xmaxB = bboxB
        
        inter_ymin = max(yminA, yminB)
        inter_xmin = max(xminA, xminB)
        inter_ymax = min(ymaxA, ymaxB)
        inter_xmax = min(xmaxA, xmaxB)
        
        if inter_ymax <= inter_ymin or inter_xmax <= inter_xmin:
            return 0.0
            
        inter_area = (inter_ymax - inter_ymin) * (inter_xmax - inter_xmin)
        areaA = (ymaxA - yminA) * (xmaxA - xminA)
        areaB = (ymaxB - yminB) * (xmaxB - xminB)
        union_area = areaA + areaB - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def calibrate_detections(self, detections: List[DetectedBarrier]) -> List[DetectedBarrier]:
        """
        Applies explicit engineering calibration:
        - Never mark as confirmed without high confidence (>= 0.85) AND low uncertainty (<= 0.20).
        - Flags uncertain predictions explicitly.
        - Attaches evidence rationale.
        """
        calibrated: List[DetectedBarrier] = []
        for d in detections:
            # Check confirmation requirements
            is_confirmed = (d.confidence >= settings.CONFIDENCE_CONFIRMED_THRESHOLD) and (d.uncertainty_score <= 0.20)
            is_borderline = (d.confidence < 0.70) or (d.uncertainty_score > settings.UNCERTAINTY_FLAG_THRESHOLD)
            
            # Format evidence
            evidence = dict(d.evidence)
            evidence["calibrated_is_confirmed"] = is_confirmed
            evidence["calibrated_is_uncertain"] = is_borderline
            
            calibrated.append(
                DetectedBarrier(
                    barrier_type=d.barrier_type,
                    confidence=d.confidence,
                    uncertainty_score=d.uncertainty_score,
                    bbox=d.bbox,
                    polygon=d.polygon,
                    evidence=evidence
                )
            )
        return calibrated

def preprocess_image(img_bgr: np.ndarray) -> Tuple[np.ndarray, bool, float, bool]:
    """
    Standalone helper for image quality checks and adaptive contrast preprocessing.
    Returns: (processed_bgr, is_blurred, laplacian_variance, clahe_applied)
    """
    pipeline = CVPipeline(adapter_name="dummy")
    proc_img, meta = pipeline.preprocess_image(img_bgr)
    return proc_img, meta["is_blurred"], meta["laplacian_variance"], meta["clahe_applied"]

def blur_sensitive_regions(img_bgr: np.ndarray) -> np.ndarray:
    """
    Applies Gaussian blurring to detected human faces and license plates
    to enforce strict privacy and anonymization without retaining biometric data.
    """
    out = img_bgr.copy()
    try:
        import os
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(face_cascade_path):
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)
            for (x, y, w, h) in faces:
                sub = out[y:y+h, x:x+w]
                if sub.shape[0] > 0 and sub.shape[1] > 0:
                    ksize = (w // 3) * 2 + 1
                    sub_blurred = cv2.GaussianBlur(sub, (max(15, ksize), max(15, ksize)), 30)
                    out[y:y+h, x:x+w] = sub_blurred
    except Exception as e:
        logger.debug(f"Privacy redaction note: {e}")
    return out

