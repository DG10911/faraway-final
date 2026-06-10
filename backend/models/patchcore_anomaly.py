import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
import cv2


class PatchCoreMemory:
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.memory_bank = None
        self.nbrs = None

    def build(self, features: np.ndarray, coreset_ratio: float = 0.1):
        if coreset_ratio < 1.0:
            n_samples = max(1, int(len(features) * coreset_ratio))
            indices = np.random.choice(len(features), n_samples, replace=False)
            features = features[indices]
        self.memory_bank = features
        self.nbrs = NearestNeighbors(n_neighbors=1, metric="cosine").fit(features)

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.nbrs is None:
            raise RuntimeError("Memory bank not built. Call build() first.")
        distances, _ = self.nbrs.kneighbors(features)
        return distances[:, 0]


class PatchCoreDetector:
    def __init__(self, backbone: str = "dinov2", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone_name = backbone
        self.memory = None
        self.anomaly_threshold = None

    def fit(self, healthy_features: np.ndarray, threshold_percentile: float = 95.0):
        self.memory = PatchCoreMemory(feature_dim=healthy_features.shape[1])
        self.memory.build(healthy_features)
        train_scores = self.memory.score(healthy_features)
        self.anomaly_threshold = np.percentile(train_scores, threshold_percentile)
        return self

    def predict(self, patch_features: np.ndarray, patch_coords: List[Tuple], image_shape: Tuple[int, int]) -> dict:
        if self.memory is None:
            raise RuntimeError("Detector not fit. Call fit() first.")
        scores = self.memory.score(patch_features)
        anomaly_mask = self._create_anomaly_mask(scores, patch_coords, image_shape)
        anomaly_scores = self._create_score_map(scores, patch_coords, image_shape)
        smoothed = gaussian_filter(anomaly_scores, sigma=4)
        max_score = float(smoothed.max())
        is_anomaly = max_score > self.anomaly_threshold
        return {
            "anomaly_score": max_score,
            "anomaly_mask": anomaly_mask,
            "anomaly_map": smoothed,
            "threshold": float(self.anomaly_threshold),
            "is_anomaly": bool(is_anomaly),
            "patch_scores": scores.tolist(),
        }

    @staticmethod
    def _create_anomaly_mask(scores: np.ndarray, patch_coords: List[Tuple], image_shape: Tuple[int, int]) -> np.ndarray:
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.float32)
        for score, (x1, y1, x2, y2) in zip(scores, patch_coords):
            mask[y1:y2, x1:x2] = max(mask[y1:y2, x1:x2].max(), score)
        return (mask > mask.mean() + mask.std()).astype(np.uint8) * 255

    @staticmethod
    def _create_score_map(scores: np.ndarray, patch_coords: List[Tuple], image_shape: Tuple[int, int]) -> np.ndarray:
        h, w = image_shape
        score_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        for score, (x1, y1, x2, y2) in zip(scores, patch_coords):
            score_map[y1:y2, x1:x2] += score
            count_map[y1:y2, x1:x2] += 1
        count_map = np.maximum(count_map, 1e-6)
        return score_map / count_map
