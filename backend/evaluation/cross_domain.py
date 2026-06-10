import numpy as np
from typing import Dict, List, Optional
from ..models.prototypical_network import PrototypicalNetwork, evaluate_few_shot
from ..models.dinov2_encoder import DINOv2Encoder
from ..datasets.dataset_loader import DatasetLoader
from ..utils.embeddings_db import EmbeddingDatabase
from .metrics import Evaluator


class CrossDomainEvaluator:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = DINOv2Encoder(device=self.device)
        self.evaluator = Evaluator()
        self.loader = DatasetLoader()
        self.db = EmbeddingDatabase()

    def evaluate_adaptation(
        self,
        source_data: Dict[str, List[np.ndarray]],
        target_data: Dict[str, List[np.ndarray]],
        n_shots: int = 5,
        n_episodes: int = 100,
    ) -> Dict:
        source_embeddings = self._embed_dataset(source_data)
        target_embeddings = self._embed_dataset(target_data)
        source_metrics = evaluate_few_shot(source_embeddings, len(source_embeddings), n_shots, n_episodes)
        target_metrics = evaluate_few_shot(target_embeddings, len(target_embeddings), n_shots, n_episodes)
        combined_embeddings = {**source_embeddings, **target_embeddings}
        combined_metrics = evaluate_few_shot(combined_embeddings, len(combined_embeddings), n_shots, n_episodes)
        return {
            "source_domain": source_metrics,
            "target_domain": target_metrics,
            "combined": combined_metrics,
            "domain_gap": {
                "accuracy_drop": round(source_metrics["mean_accuracy"] - target_metrics["mean_accuracy"], 4),
            },
        }

    def _embed_dataset(self, data: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
        embeddings = {}
        for class_name, images in data.items():
            class_embs = []
            for img in images:
                from ..pipeline.phase2_rail_extraction import RailExtractor
                from ..pipeline.phase3_patch_pipeline import PatchPipeline
                extractor = RailExtractor()
                patcher = PatchPipeline()
                rail = extractor.extract_rail_region(img)
                if rail is not None:
                    patches = patcher.extract_patches(rail)
                    patch_imgs = [p for p, _ in patches]
                    embs = self.encoder.embed_patches(patch_imgs)
                    class_embs.append(embs.mean(axis=0))
            if class_embs:
                embeddings[class_name] = np.array(class_embs)
        return embeddings

    def generate_comparison_table(self) -> Dict:
        return {
            "Baseline CNN": {"accuracy": 0.0, "f1": 0.0, "note": "Requires full retraining"},
            "Transfer Learning": {"accuracy": 0.0, "f1": 0.0, "note": "Fine-tune on 5 target images"},
            "RailGuard-FSL++": {"accuracy": 0.0, "f1": 0.0, "note": "5-shot prototype adaptation"},
            "methodology": "Train on Mendeley, adapt using 5 support images from RSDDs/Fastener dataset",
        }


import torch
