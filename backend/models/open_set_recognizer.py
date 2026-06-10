import numpy as np
from typing import Dict, List, Optional, Tuple


class OpenSetRecognizer:
    def __init__(self, distance_threshold: float = 1.5, anomaly_score_threshold: float = 0.5):
        self.distance_threshold = distance_threshold
        self.anomaly_score_threshold = anomaly_score_threshold

    def recognize(
        self,
        query_embedding: np.ndarray,
        prototypes: Dict[str, np.ndarray],
        anomaly_score: float,
    ) -> Tuple[str, float]:
        if not prototypes:
            return ("unknown_anomaly", 0.0)

        proto_names = list(prototypes.keys())
        proto_array = np.array(list(prototypes.values()))
        distances = np.linalg.norm(query_embedding[np.newaxis, :] - proto_array, axis=1)
        min_dist = float(distances.min())
        nearest_idx = int(distances.argmin())
        nearest_class = proto_names[nearest_idx]
        confidence = float(1.0 / (1.0 + min_dist))

        is_open_set = min_dist > self.distance_threshold and anomaly_score > self.anomaly_score_threshold

        if is_open_set:
            return ("unknown_anomaly", confidence)
        return (nearest_class, confidence)

    def batch_recognize(
        self,
        query_embeddings: np.ndarray,
        prototypes: Dict[str, np.ndarray],
        anomaly_scores: np.ndarray,
    ) -> List[Dict]:
        results = []
        for emb, score in zip(query_embeddings, anomaly_scores):
            label, confidence = self.recognize(emb, prototypes, score)
            results.append({"label": label, "confidence": confidence})
        return results

    def calibrate_threshold(
        self,
        known_embeddings: np.ndarray,
        known_prototypes: Dict[str, np.ndarray],
        percentile: float = 95.0,
    ):
        distances = []
        for emb in known_embeddings:
            proto_array = np.array(list(known_prototypes.values()))
            dists = np.linalg.norm(emb[np.newaxis, :] - proto_array, axis=1)
            distances.append(dists.min())
        self.distance_threshold = float(np.percentile(distances, percentile))
        return self.distance_threshold
