import numpy as np
from typing import Dict, List
from pathlib import Path


HARD_NEGATIVE_CLASSES = [
    "weld",
    "fishplate",
    "joint",
    "switch",
    "fastener",
    "stain",
    "rust",
    "water_mark",
]


class HardNegativeManager:
    def __init__(self):
        self.hard_negative_embeddings: Dict[str, np.ndarray] = {}

    def register(self, class_name: str, embeddings: np.ndarray):
        if class_name in HARD_NEGATIVE_CLASSES:
            self.hard_negative_embeddings[class_name] = embeddings

    def get_all(self) -> Dict[str, np.ndarray]:
        return dict(self.hard_negative_embeddings)

    def has_hard_negative(self, query_embedding: np.ndarray, threshold: float = 0.8) -> bool:
        if not self.hard_negative_embeddings:
            return False
        for class_name, emb_bank in self.hard_negative_embeddings.items():
            similarities = np.dot(emb_bank, query_embedding) / (
                np.linalg.norm(emb_bank, axis=1) * np.linalg.norm(query_embedding) + 1e-8
            )
            if similarities.max() > threshold:
                return True
        return False
