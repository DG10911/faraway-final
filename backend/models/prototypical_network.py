import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class PrototypicalNetwork(nn.Module):
    def __init__(self, embedding_dim: int, n_ways: int, n_shots: int, n_queries: int = 15):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.prototypes = None
        self.class_names = None

    def compute_prototypes(self, support_embeddings: np.ndarray, support_labels: np.ndarray, class_names: List[str]):
        unique_labels = np.unique(support_labels)
        self.prototypes = {}
        self.class_names = class_names
        for label in unique_labels:
            class_embs = support_embeddings[support_labels == label]
            median_proto = np.median(class_embs, axis=0)
            self.prototypes[int(label)] = median_proto
        return self

    def classify(self, query_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.prototypes is None:
            raise RuntimeError("Prototypes not computed. Call compute_prototypes() first.")
        proto_array = np.array([self.prototypes[k] for k in sorted(self.prototypes.keys())])
        distances = np.linalg.norm(query_embeddings[:, np.newaxis, :] - proto_array[np.newaxis, :, :], axis=2)
        preds = np.argmin(distances, axis=1)
        confidences = 1.0 / (1.0 + distances)
        confidence_scores = np.max(confidences, axis=1)
        return preds, confidence_scores

    def open_set_classify(self, query_embeddings: np.ndarray, distance_threshold: float = 2.0) -> List[Dict]:
        if self.prototypes is None:
            raise RuntimeError("Prototypes not computed.")
        proto_array = np.array([self.prototypes[k] for k in sorted(self.prototypes.keys())])
        distances = np.linalg.norm(query_embeddings[:, np.newaxis, :] - proto_array[np.newaxis, :, :], axis=2)
        min_distances = np.min(distances, axis=1)
        pred_indices = np.argmin(distances, axis=1)
        results = []
        label_map = {i: name for i, name in enumerate(self.class_names)} if self.class_names else {}
        for i, (dist, pred_idx) in enumerate(zip(min_distances, pred_indices)):
            if dist > distance_threshold:
                results.append({"label": "unknown_anomaly", "confidence": float(1.0 / (1.0 + dist)), "distance": float(dist)})
            else:
                class_name = label_map.get(int(pred_idx), f"class_{pred_idx}")
                results.append({"label": class_name, "confidence": float(1.0 / (1.0 + dist)), "distance": float(dist)})
        return results


class EpisodicSampler:
    def __init__(self, embeddings: Dict[str, np.ndarray], n_ways: int, n_shots: int, n_queries: int = 15):
        self.embeddings = embeddings
        self.class_names = list(embeddings.keys())
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries

    def sample_episode(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        selected_classes = np.random.choice(self.class_names, self.n_ways, replace=False)
        support_x, support_y = [], []
        query_x, query_y = [], []
        episode_classes = []
        for label_idx, class_name in enumerate(selected_classes):
            class_embs = self.embeddings[class_name]
            n_total = len(class_embs)
            n_support = min(self.n_shots, n_total // 2)
            n_query = min(self.n_queries, n_total - n_support)
            if n_support < 1 or n_query < 1:
                continue
            indices = np.random.permutation(n_total)
            support_indices = indices[:n_support]
            query_indices = indices[n_support:n_support + n_query]
            support_x.append(class_embs[support_indices])
            support_y.extend([label_idx] * n_support)
            query_x.append(class_embs[query_indices])
            query_y.extend([label_idx] * n_query)
            episode_classes.append(class_name)
        if not support_x:
            return self.sample_episode()
        return (
            np.concatenate(support_x, axis=0),
            np.array(support_y),
            np.concatenate(query_x, axis=0),
            np.array(query_y),
            episode_classes,
        )


def evaluate_few_shot(
    embeddings: Dict[str, np.ndarray],
    n_ways: int,
    n_shots: int,
    n_episodes: int = 100,
    embedding_dim: Optional[int] = None,
    distance_threshold: float = 2.0,
    open_set_classes: Optional[List[str]] = None,
) -> Dict:
    sampler = EpisodicSampler(embeddings, n_ways, n_shots)
    accuracies = []
    open_set_detections = []

    for _ in range(n_episodes):
        s_x, s_y, q_x, q_y, class_names = sampler.sample_episode()
        if len(s_x) == 0:
            continue
        model = PrototypicalNetwork(
            embedding_dim=embedding_dim or s_x.shape[1],
            n_ways=len(np.unique(s_y)),
            n_shots=n_shots,
        )
        model.compute_prototypes(s_x, s_y, class_names)
        if open_set_classes:
            results = model.open_set_classify(q_x, distance_threshold)
            preds = np.array([0 if r["label"] == "unknown_anomaly" else class_names.index(r["label"]) for r in results])
            for i, r in enumerate(results):
                if r["label"] == "unknown_anomaly" and q_y[i] >= len(class_names):
                    open_set_detections.append(1)
                elif r["label"] != "unknown_anomaly" and q_y[i] < len(class_names):
                    open_set_detections.append(0)
        else:
            preds, _ = model.classify(q_x)
        correct = np.sum(preds == q_y)
        accuracies.append(correct / len(q_y))

    mean_acc = float(np.mean(accuracies)) if accuracies else 0.0
    std_acc = float(np.std(accuracies)) if accuracies else 0.0
    result = {"mean_accuracy": mean_acc, "std_accuracy": std_acc, "n_episodes": len(accuracies)}
    if open_set_detections:
        result["open_set_detection_rate"] = float(np.mean(open_set_detections))
    return result
