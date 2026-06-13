import numpy as np
import torch
from typing import Dict, List, Optional

from ..models.prototypical_network import evaluate_few_shot
from ..models.dinov2_encoder import DINOv2Encoder
from ..pipeline.phase2_rail_extraction import RailExtractor
from ..pipeline.phase3_patch_pipeline import PatchPipeline
from .metrics import Evaluator


class CrossDomainEvaluator:
    """The headline few-shot story: train on one railway (Mendeley/Pakistan),
    re-anchor prototypes with K shots from another (RSDDs/fasteners), and
    measure the accuracy drop — no retraining."""

    def __init__(self, device: Optional[str] = None, encoder: Optional[DINOv2Encoder] = None):
        from ..utils.device import resolve_device
        self.device = resolve_device(device)
        self.encoder = encoder or DINOv2Encoder(device=self.device)
        self.evaluator = Evaluator()
        self.extractor = RailExtractor()
        self.patcher = PatchPipeline()

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
                rail = self.extractor.extract_rail_region(img)
                if rail is not None:
                    patch_imgs = [p for p, _ in self.patcher.extract_patches(rail)]
                    embs = self.encoder.embed_patches(patch_imgs)
                    class_embs.append(embs.mean(axis=0))  # one embedding per image -> no patch leakage
            if class_embs:
                embeddings[class_name] = np.array(class_embs)
        return embeddings
