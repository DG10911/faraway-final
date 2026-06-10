import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
import imagehash
from PIL import Image


DEFECT_CLASSES = [
    "healthy",
    "crack",
    "squat",
    "spalling",
    "flaking",
    "shelling",
    "groove",
    "joint",
    "fastener",
]


class DatasetLoader:
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)

    def load_mendeley(self, path: str) -> Dict[str, List[np.ndarray]]:
        data = {cls: [] for cls in DEFECT_CLASSES}
        path = Path(path)
        if not path.exists():
            return data
        for cls in DEFECT_CLASSES:
            cls_dir = path / cls
            if cls_dir.exists():
                for img_file in cls_dir.glob("*.*"):
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        data[cls].append(img)
        return data

    def load_rsdds(self, path: str) -> Dict[str, List[np.ndarray]]:
        data = {"healthy": [], "defect": []}
        path = Path(path)
        if not path.exists():
            return data
        for cls in ["healthy", "defect"]:
            cls_dir = path / cls
            if cls_dir.exists():
                for img_file in cls_dir.glob("*.*"):
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        data[cls].append(img)
        return data

    def load_kaggle(self, path: str) -> Dict[str, List[np.ndarray]]:
        data = {"healthy": [], "defective": []}
        path = Path(path)
        if not path.exists():
            return data
        for cls in ["healthy", "defective"]:
            cls_dir = path / cls
            if cls_dir.exists():
                for img_file in cls_dir.glob("*.*"):
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        data[cls].append(img)
        return data

    def load_fastener(self, path: str) -> Dict[str, List[np.ndarray]]:
        data = {"healthy_fastener": [], "defective_fastener": []}
        path = Path(path)
        if not path.exists():
            return data
        for cls in ["healthy_fastener", "defective_fastener"]:
            cls_dir = path / cls
            if cls_dir.exists():
                for img_file in cls_dir.glob("*.*"):
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        data[cls].append(img)
        return data

    @staticmethod
    def deduplicate(images: List[np.ndarray], threshold: int = 5) -> List[np.ndarray]:
        hashes = []
        unique = []
        for img in images:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img)
            h = imagehash.phash(pil_img)
            if all(abs(h - existing) > threshold for existing in hashes):
                hashes.append(h)
                unique.append(img)
        return unique

    @staticmethod
    def split_by_defect_instance(embeddings: np.ndarray, labels: np.ndarray, test_ratio: float = 0.2, random_state: int = 42):
        unique_instances = np.unique(labels)
        train_idx, test_idx = train_test_split(
            np.arange(len(labels)),
            test_size=test_ratio,
            stratify=labels,
            random_state=random_state,
        )
        return embeddings[train_idx], embeddings[test_idx], labels[train_idx], labels[test_idx]
