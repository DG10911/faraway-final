import cv2
import numpy as np
from typing import Tuple, Optional


class RailExtractor:
    def __init__(self):
        self.roi = None

    def extract_rail_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        rail_mask = self._segment_rail(image)
        cropped = self._crop_rail_region(image, rail_mask)
        if cropped is None:
            return None
        corrected = self._perspective_correct(cropped)
        return corrected

    @staticmethod
    def _segment_rail(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        return mask

    @staticmethod
    def _crop_rail_region(image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        white_pixels = np.where(mask > 0)
        if len(white_pixels[0]) == 0 or len(white_pixels[1]) == 0:
            return None
        y_min, y_max = int(white_pixels[0].min()), int(white_pixels[0].max())
        x_min, x_max = int(white_pixels[1].min()), int(white_pixels[1].max())
        padding = 20
        y_min = max(0, y_min - padding)
        y_max = min(image.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        cropped = image[y_min:y_max, x_min:x_max]
        return cropped

    @staticmethod
    def _perspective_correct(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=w // 2, maxLineGap=20)
        if lines is None:
            return image
        angles = [np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0]) for line in lines]
        median_angle = np.median(angles)
        if abs(median_angle) < 0.05:
            return image
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(median_angle), 1.0)
        corrected = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return corrected
