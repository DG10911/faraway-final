import numpy as np
import json
import os
from typing import Dict, List, Optional
from pathlib import Path


class EmbeddingDatabase:
    def __init__(self, storage_path: str = "data/embeddings"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_path / "index.json"
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
        self._load_index()

    def _load_index(self):
        if self.index_file.exists():
            with open(self.index_file) as f:
                data = json.load(f)
            for class_name, meta in data.items():
                emb_path = self.storage_path / f"{class_name}.npy"
                if emb_path.exists():
                    self.embeddings[class_name] = np.load(emb_path)
                    self.metadata[class_name] = meta

    def _save_index(self):
        index_data = {}
        for class_name, meta in self.metadata.items():
            index_data[class_name] = meta
        with open(self.index_file, "w") as f:
            json.dump(index_data, f, indent=2)

    def store(self, class_name: str, embeddings: np.ndarray, meta: Optional[Dict] = None):
        emb_path = self.storage_path / f"{class_name}.npy"
        np.save(emb_path, embeddings)
        self.embeddings[class_name] = embeddings
        self.metadata[class_name] = meta or {"n_samples": len(embeddings), "dim": embeddings.shape[1]}
        self._save_index()

    def get(self, class_name: str) -> Optional[np.ndarray]:
        return self.embeddings.get(class_name)

    def get_all(self) -> Dict[str, np.ndarray]:
        return dict(self.embeddings)

    def list_classes(self) -> List[str]:
        return list(self.embeddings.keys())

    def delete(self, class_name: str):
        if class_name in self.embeddings:
            del self.embeddings[class_name]
            del self.metadata[class_name]
            emb_path = self.storage_path / f"{class_name}.npy"
            if emb_path.exists():
                emb_path.unlink()
            self._save_index()
