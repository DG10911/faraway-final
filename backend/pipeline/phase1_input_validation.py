import cv2
import numpy as np
from typing import Tuple, Dict


class InputValidator:
    def __init__(self, blur_threshold: float = 100.0, occlusion_threshold: float = 0.3, quality_threshold: float = 0.5):
        self.blur_threshold = blur_threshold
        self.occlusion_threshold = occlusion_threshold
        self.quality_threshold = quality_threshold

    def validate(self, image: np.ndarray) -> Dict:
        rail_present = self.detect_rail_presence(image)
        if not rail_present:
            return {"status": "invalid_frame", "reason": "no_rail_detected"}

        blur_score = self.detect_blur(image)
        if blur_score < self.blur_threshold:
            return {"status": "invalid_frame", "reason": "blurred_image", "blur_score": float(blur_score)}

        occlusion_score = self.detect_occlusion(image)
        if occlusion_score > self.occlusion_threshold:
            return {"status": "invalid_frame", "reason": "occluded_image", "occlusion_score": float(occlusion_score)}

        quality_score = self.assess_quality(image)
        if quality_score < self.quality_threshold:
            return {"status": "invalid_frame", "reason": "poor_quality", "quality_score": float(quality_score)}

        return {"status": "valid", "blur_score": float(blur_score), "quality_score": float(quality_score)}

    @staticmethod
    def detect_rail_presence(image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
        if lines is not None:
            horizontal_lines = sum(1 for line in lines if abs(line[0][1] - line[0][3]) < 30)
            return horizontal_lines > 3
        return False

    @staticmethod
    def detect_blur(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def detect_occlusion(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_map = cv2.GaussianBlur(gray, (15, 15), 0)
        dark_mask = blur_map < 30
        occlusion_ratio = float(np.sum(dark_mask) / dark_mask.size)
        return occlusion_ratio

    @staticmethod
    def assess_quality(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = float(gray.std())
        brightness = float(gray.mean())
        normalized_contrast = min(contrast / 80.0, 1.0)
        normalized_brightness = 1.0 - abs(brightness - 128.0) / 128.0
        return 0.5 * normalized_contrast + 0.5 * normalized_brightness
