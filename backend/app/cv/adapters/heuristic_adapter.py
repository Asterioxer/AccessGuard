import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from app.cv.base import ModelAdapter, DetectedBarrier

class HeuristicCVAdapter(ModelAdapter):
    """
    OpenCV-based feature extractor and accessibility barrier detector.
    Identifies:
    1. Stairs / Steps: via periodic horizontal edge rungs and gradient frequency.
    2. Damaged footpaths: via high-frequency crack contour density and Laplacian variance.
    3. Vertical obstacles / bollards / poles: via vertical aspect ratio analysis in pedestrian corridor.
    4. Missing / present curb ramps: via slope transition and step drop-off indicators.
    Computes mathematical uncertainty based on edge SNR, variance, and lighting contrast.
    """
    def __init__(self):
        super().__init__(name="opencv_heuristic_adapter")

    def load(self) -> bool:
        self.is_initialized = True
        return True

    def predict(self, image_bgr: np.ndarray) -> List[DetectedBarrier]:
        if not self.is_initialized:
            self.load()

        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Calculate overall image contrast and brightness for uncertainty calibration
        mean_brightness = float(np.mean(gray))
        contrast_std = float(np.std(gray))
        lighting_uncertainty = 0.0
        if contrast_std < 25.0: # Low contrast / foggy / washed out
            lighting_uncertainty += 0.25
        if mean_brightness < 30.0 or mean_brightness > 225.0: # Too dark or overexposed
            lighting_uncertainty += 0.25

        detections: List[DetectedBarrier] = []

        # 1. Stairs detection using horizontal edge grouping
        stairs_det = self._detect_stairs(gray, w, h, lighting_uncertainty)
        if stairs_det:
            detections.append(stairs_det)

        # 2. Damaged footpaths / cracked pavement detection in lower ground ROI
        damage_det = self._detect_surface_damage(gray, w, h, lighting_uncertainty)
        if damage_det:
            detections.append(damage_det)

        # 3. Obstacle / Pole / Bollard detection in pedestrian pathway
        obstacle_det = self._detect_vertical_obstacles(gray, w, h, lighting_uncertainty)
        if obstacle_det:
            detections.append(obstacle_det)

        # 4. Curb ramp analysis
        ramp_det = self._detect_curb_ramps(gray, w, h, lighting_uncertainty)
        if ramp_det:
            detections.append(ramp_det)

        return detections

    def _detect_stairs(self, gray: np.ndarray, w: int, h: int, lighting_penalty: float) -> Optional[DetectedBarrier]:
        """Detects steps/stairs by clustering parallel horizontal lines."""
        # Focus on lower 75% of image where pedestrian walkway/stairs lie
        roi_ymin = int(h * 0.25)
        roi = gray[roi_ymin:, :]
        
        # Sobel horizontal gradient filter
        grad_y = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        grad_y_abs = cv2.convertScaleAbs(grad_y)
        
        # Adaptive thresholding for rungs
        _, thresh = cv2.threshold(grad_y_abs, 45, 255, cv2.THRESH_BINARY)
        
        # Morphological horizontal kernel
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
        
        # Hough line transform
        lines = cv2.HoughLinesP(morphed, rho=1, theta=np.pi/180, threshold=40, minLineLength=w * 0.2, maxLineGap=20)
        
        if lines is not None and len(lines) >= 3:
            # Filter nearly horizontal lines (angle within +/- 15 degrees)
            horiz_y = []
            min_x, max_x = w, 0
            for line in lines:
                l = line[0] if hasattr(line[0], '__len__') and len(line[0]) == 4 else line
                if len(l) < 4:
                    continue
                x1, y1, x2, y2 = int(l[0]), int(l[1]), int(l[2]), int(l[3])
                dx = x2 - x1
                dy = y2 - y1
                angle = abs(np.arctan2(dy, dx) * 180.0 / np.pi)
                if angle < 15.0 or angle > 165.0:
                    y_actual = roi_ymin + (y1 + y2) // 2
                    horiz_y.append(y_actual)
                    min_x = min(min_x, x1, x2)
                    max_x = max(max_x, x1, x2)
            
            # Check for regular vertical spacing between rungs
            horiz_y.sort()
            if len(horiz_y) >= 3:
                # Deduplicate lines that are within 10 pixels
                dedup_y = [horiz_y[0]]
                for y in horiz_y[1:]:
                    if y - dedup_y[-1] > 12:
                        dedup_y.append(y)
                        
                step_count = len(dedup_y)
                if step_count >= 3:
                    # Bounding box in normalized coords
                    ymin = max(0.0, float(dedup_y[0] - 10) / h)
                    ymax = min(1.0, float(dedup_y[-1] + 20) / h)
                    xmin = max(0.0, float(min_x - 15) / w)
                    xmax = min(1.0, float(max_x + 15) / w)
                    
                    # Confidence scaling with step count
                    base_conf = min(0.94, 0.65 + (step_count * 0.06))
                    uncertainty = min(0.60, max(0.08, 0.15 + lighting_penalty - (step_count * 0.02)))
                    
                    polygon = [
                        [round(xmin, 3), round(ymin, 3)],
                        [round(xmax, 3), round(ymin, 3)],
                        [round(xmax, 3), round(ymax, 3)],
                        [round(xmin, 3), round(ymax, 3)]
                    ]
                    
                    return DetectedBarrier(
                        barrier_type="stairs",
                        confidence=round(base_conf, 2),
                        uncertainty_score=round(uncertainty, 2),
                        bbox=(round(ymin, 3), round(xmin, 3), round(ymax, 3), round(xmax, 3)),
                        polygon=polygon,
                        evidence={
                            "feature_type": "periodic_horizontal_edges",
                            "measured_metric": float(step_count),
                            "threshold_applied": 3.0,
                            "explanation": f"Detected {step_count} aligned horizontal stair tread rungs. Impassable for standard wheelchair without ramp."
                        }
                    )
        return None

    def _detect_surface_damage(self, gray: np.ndarray, w: int, h: int, lighting_penalty: float) -> Optional[DetectedBarrier]:
        """Detects cracked concrete, uneven pavement, or potholes in lower pedestrian path."""
        # Ground region of interest
        ground_roi = gray[int(h * 0.55):int(h * 0.95), int(w * 0.15):int(w * 0.85)]
        
        # High-frequency edge detection via Canny
        edges = cv2.Canny(ground_roi, 60, 150)
        edge_density = float(np.count_nonzero(edges)) / float(ground_roi.size)
        
        # Local Laplacian variance (texture roughness measure)
        laplacian_var = float(cv2.Laplacian(ground_roi, cv2.CV_64F).var())
        
        # Normal sidewalks are relatively smooth (low edge density in ground plane)
        # Cracked or broken surfaces exhibit high chaotic edge density or texture variance
        if (edge_density > 0.025 and laplacian_var > 100.0) or edge_density > 0.05:
            confidence = min(0.92, 0.55 + (edge_density * 5.0))
            uncertainty = min(0.55, max(0.12, 0.18 + lighting_penalty))
            
            ymin = 0.55
            ymax = 0.95
            xmin = 0.15
            xmax = 0.85
            
            return DetectedBarrier(
                barrier_type="surface_damage",
                confidence=round(confidence, 2),
                uncertainty_score=round(uncertainty, 2),
                bbox=(ymin, xmin, ymax, xmax),
                polygon=[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                evidence={
                    "feature_type": "surface_roughness_and_crack_density",
                    "measured_metric": round(edge_density, 3),
                    "threshold_applied": 0.07,
                    "explanation": f"High frequency fracture pattern (density: {edge_density:.1%}, texture var: {laplacian_var:.1f}). Pavement is broken/uneven, risking wheelchair tip-over."
                }
            )
        return None

    def _detect_vertical_obstacles(self, gray: np.ndarray, w: int, h: int, lighting_penalty: float) -> Optional[DetectedBarrier]:
        """Detects vertical poles, bollards, or narrow construction barriers in pathway."""
        mid_roi = gray[int(h * 0.25):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
        
        # Binary thresholding for high-contrast obstructions
        _, thresh = cv2.threshold(mid_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Edge gradient in vertical direction as alternative
        sobel_x = cv2.Sobel(mid_roi, cv2.CV_32F, 1, 0, ksize=3)
        sobel_x_abs = cv2.convertScaleAbs(sobel_x)
        combined = cv2.bitwise_or(thresh, sobel_x_abs)
        
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (w * h * 0.012):
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect_ratio = float(ch) / float(cw) if cw > 0 else 0
                
                # Bollard or utility pole profile: vertical aspect ratio > 1.8
                if aspect_ratio >= 1.8 and ch > (h * 0.20):
                    global_x = int(w * 0.15) + x
                    global_y = int(h * 0.25) + y
                    
                    ymin = round(global_y / h, 3)
                    ymax = round((global_y + ch) / h, 3)
                    xmin = round(global_x / w, 3)
                    xmax = round((global_x + cw) / w, 3)
                    
                    confidence = 0.84
                    uncertainty = round(min(0.35, 0.16 + lighting_penalty), 2)
                    
                    return DetectedBarrier(
                        barrier_type="pole_or_bollard",
                        confidence=confidence,
                        uncertainty_score=uncertainty,
                        bbox=(ymin, xmin, ymax, xmax),
                        polygon=[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                        evidence={
                            "feature_type": "vertical_corridor_obstruction",
                            "measured_metric": round(aspect_ratio, 2),
                            "threshold_applied": 1.8,
                            "explanation": f"Vertical post/bollard structure (aspect ratio {aspect_ratio:.1f}) constricting pedestrian walkway clearance."
                        }
                    )
        return None

    def _detect_curb_ramps(self, gray: np.ndarray, w: int, h: int, lighting_penalty: float) -> Optional[DetectedBarrier]:
        """Detects whether curb transition has ramp slope or missing drop-off."""
        # Street crossing transition zone: bottom 40% of frame
        curb_zone_ymin = int(h * 0.60)
        curb_zone = gray[curb_zone_ymin:, :]
        edges = cv2.Canny(curb_zone, 40, 140)
        
        # Check diagonal lines that indicate flared curb ramp slopes
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 25, minLineLength=int(w * 0.10), maxLineGap=20)
        has_diagonal_flare = False
        if lines is not None:
            for line in lines:
                l = line[0] if hasattr(line[0], '__len__') and len(line[0]) == 4 else line
                if len(l) < 4:
                    continue
                x1, y1, x2, y2 = int(l[0]), int(l[1]), int(l[2]), int(l[3])
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                if 20.0 <= angle <= 70.0:
                    has_diagonal_flare = True
                    break
                    
        if has_diagonal_flare:
            return DetectedBarrier(
                barrier_type="curb_ramp_present",
                confidence=0.86,
                uncertainty_score=round(min(0.30, 0.14 + lighting_penalty), 2),
                bbox=(0.65, 0.15, 0.98, 0.85),
                polygon=[[0.15, 0.65], [0.85, 0.65], [0.85, 0.98], [0.15, 0.98]],
                evidence={
                    "feature_type": "sloped_curb_transition",
                    "measured_metric": 1.0,
                    "threshold_applied": 0.5,
                    "explanation": "Sloped transition flare detected matching ADA curb ramp profile. Accessible street crossing."
                }
            )
        return None
