import numpy as np
from typing import List, Optional, Tuple
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norms, 1e-8)


def kcenter_greedy_coreset(features: np.ndarray, n_select: int, pre_subsample: int = 30000, seed: int = 0) -> np.ndarray:
    """Greedy k-center coreset selection (the subsampling used in PatchCore).

    Iteratively picks the point farthest from the current coreset, so the
    memory bank covers the healthy manifold instead of oversampling dense
    regions the way random sampling does.
    """
    rng = np.random.default_rng(seed)
    if len(features) > pre_subsample:
        features = features[rng.choice(len(features), pre_subsample, replace=False)]
    n_select = min(n_select, len(features))
    selected = [int(rng.integers(len(features)))]
    min_dists = np.linalg.norm(features - features[selected[0]], axis=1)
    for _ in range(n_select - 1):
        idx = int(np.argmax(min_dists))
        selected.append(idx)
        new_dists = np.linalg.norm(features - features[idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
    return features[selected]


class PatchCoreMemory:
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.memory_bank = None
        self.nbrs = None

    def build(self, features: np.ndarray, coreset_ratio: float = 0.1, max_memory: int = 10000):
        features = _l2_normalize(features.astype(np.float32))
        n_select = min(max(1, int(len(features) * coreset_ratio)), max_memory)
        if n_select < len(features):
            features = kcenter_greedy_coreset(features, n_select)
        self.memory_bank = features
        self.nbrs = NearestNeighbors(n_neighbors=1, metric="cosine").fit(features)

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.nbrs is None:
            raise RuntimeError("Memory bank not built. Call build() first.")
        features = _l2_normalize(features.astype(np.float32))
        distances, _ = self.nbrs.kneighbors(features)
        return distances[:, 0]


class PatchCoreDetector:
    """Healthy-only anomaly detector over (flattened) DINOv2 patch-token features.

    The decision threshold is calibrated on raw 1-NN distances and compared
    against the raw max patch distance — the smoothed anomaly map is for
    visualization only, so calibration and decision live on the same scale.
    """

    def __init__(self, backbone: str = "dinov2", device: Optional[str] = None):
        self.device = device
        self.backbone_name = backbone
        self.memory = None
        self.anomaly_threshold = None
        self.healthy_scores = None

    def fit(self, healthy_features: np.ndarray, threshold_percentile: float = 95.0):
        if healthy_features.ndim == 3:
            healthy_features = healthy_features.reshape(-1, healthy_features.shape[-1])
        self.memory = PatchCoreMemory(feature_dim=healthy_features.shape[1])
        self.memory.build(healthy_features)
        self.healthy_scores = self.memory.score(healthy_features)
        self.anomaly_threshold = float(np.percentile(self.healthy_scores, threshold_percentile))
        return self

    def recalibrate(self, threshold_percentile: float):
        """Re-pick the operating threshold without rebuilding the memory bank."""
        if self.healthy_scores is None:
            raise RuntimeError("Detector not fit. Call fit() first.")
        self.anomaly_threshold = float(np.percentile(self.healthy_scores, threshold_percentile))
        return self.anomaly_threshold

    def score_features(self, features: np.ndarray) -> np.ndarray:
        if self.memory is None:
            raise RuntimeError("Detector not fit. Call fit() first.")
        if features.ndim == 3:
            features = features.reshape(-1, features.shape[-1])
        return self.memory.score(features)

    def predict(self, patch_features: np.ndarray, patch_coords: List[Tuple], image_shape: Tuple[int, int]) -> dict:
        """Score per-patch features. Accepts (n_patches, dim) CLS features or
        (n_patches, n_tokens, dim) token features; tokens are mapped to
        sub-cells of each patch box for a finer anomaly map."""
        if self.memory is None:
            raise RuntimeError("Detector not fit. Call fit() first.")

        if patch_features.ndim == 3:
            n_patches, n_tokens, _ = patch_features.shape
            grid = int(round(np.sqrt(n_tokens)))
            scores = self.memory.score(patch_features.reshape(-1, patch_features.shape[-1]))
            scores = scores.reshape(n_patches, n_tokens)
            unit_scores, unit_coords = self._token_units(scores, patch_coords, grid)
        else:
            unit_scores = self.memory.score(patch_features)
            unit_coords = list(patch_coords)

        anomaly_map = self._create_score_map(unit_scores, unit_coords, image_shape)
        smoothed = gaussian_filter(anomaly_map, sigma=4)
        max_score = float(np.max(unit_scores))  # raw scale — same scale as the calibrated threshold
        is_anomaly = max_score > self.anomaly_threshold
        anomaly_mask = (anomaly_map > self.anomaly_threshold).astype(np.uint8) * 255
        per_patch = unit_scores.reshape(len(patch_coords), -1).max(axis=1) if patch_features.ndim == 3 else unit_scores
        return {
            "anomaly_score": max_score,
            "anomaly_mask": anomaly_mask,
            "anomaly_map": smoothed,
            "threshold": float(self.anomaly_threshold),
            "is_anomaly": bool(is_anomaly),
            "patch_scores": per_patch.tolist(),
        }

    @staticmethod
    def _token_units(token_scores: np.ndarray, patch_coords: List[Tuple], grid: int) -> Tuple[np.ndarray, List[Tuple]]:
        """Subdivide each patch box into a grid x grid lattice of token cells."""
        unit_scores = []
        unit_coords = []
        for patch_idx, (x1, y1, x2, y2) in enumerate(patch_coords):
            cell_w = (x2 - x1) / grid
            cell_h = (y2 - y1) / grid
            for row in range(grid):
                for col in range(grid):
                    unit_scores.append(token_scores[patch_idx, row * grid + col])
                    unit_coords.append((
                        int(x1 + col * cell_w),
                        int(y1 + row * cell_h),
                        int(x1 + (col + 1) * cell_w),
                        int(y1 + (row + 1) * cell_h),
                    ))
        return np.array(unit_scores), unit_coords

    @staticmethod
    def _create_score_map(scores: np.ndarray, coords: List[Tuple], image_shape: Tuple[int, int]) -> np.ndarray:
        h, w = image_shape
        score_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        for score, (x1, y1, x2, y2) in zip(scores, coords):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            score_map[y1:y2, x1:x2] += score
            count_map[y1:y2, x1:x2] += 1
        count_map = np.maximum(count_map, 1e-6)
        return score_map / count_map
