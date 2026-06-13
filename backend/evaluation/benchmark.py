import json
import time
import numpy as np
import torch
import torchvision
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from sklearn.linear_model import LogisticRegression

from .metrics import Evaluator
from ..models.prototypical_network import EpisodicSampler, evaluate_few_shot


@dataclass
class BenchmarkResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time_ms: float
    params_count: Optional[int] = None
    few_shot_mean: Optional[float] = None
    few_shot_std: Optional[float] = None
    unknown_defect_detection_rate: Optional[float] = None
    false_positive_rate: Optional[float] = None
    cross_domain_accuracy: Optional[float] = None
    auroc: Optional[float] = None
    mAP: Optional[float] = None

    def to_dict(self):
        return asdict(self)


class BenchmarkRunner:
    def __init__(self):
        self.evaluator = Evaluator()
        self.results = []

    def evaluate_model(self, model_name: str, predict_fn: Callable, test_data: tuple, **kwargs):
        x_test, y_test = test_data
        start = time.time()
        y_pred = predict_fn(x_test)
        elapsed = (time.time() - start) * 1000 / len(x_test)
        metrics = self.evaluator.compute_classification_metrics(y_test, y_pred)
        result = BenchmarkResult(
            model_name=model_name,
            accuracy=metrics["accuracy"],
            precision=metrics["precision_macro"],
            recall=metrics["recall_macro"],
            f1_score=metrics["f1_macro"],
            inference_time_ms=round(elapsed, 3),
            **{k: kwargs.get(k) for k in ["few_shot_mean", "few_shot_std", "unknown_defect_detection_rate", "false_positive_rate", "cross_domain_accuracy", "auroc", "mAP"] if k in kwargs and kwargs[k] is not None},
        )
        self.results.append(result)
        return result

    def generate_comparison(self) -> Dict:
        comparison = {}
        for r in self.results:
            comparison[r.model_name] = r.to_dict()
        return comparison

    def save_results(self, path: str):
        with open(path, "w") as f:
            json.dump(self.generate_comparison(), f, indent=2)


def extract_resnet18_features(images_by_class: Dict[str, List[np.ndarray]], device: str = "cpu", batch_size: int = 32) -> Dict[str, np.ndarray]:
    """ImageNet-pretrained ResNet-18 avgpool features (the classic transfer-learning backbone)."""
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    import cv2
    features = {}
    for cls_name, images in images_by_class.items():
        if not images:
            continue
        embs = []
        for i in range(0, len(images), batch_size):
            batch = []
            for img in images[i:i + batch_size]:
                resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                if resized.ndim == 2:
                    resized = np.stack([resized] * 3, axis=-1)
                batch.append(torch.from_numpy(resized).permute(2, 0, 1))
            x = torch.stack(batch).to(device)
            x = (x - mean) / std
            with torch.no_grad():
                out = model(x)
            embs.append(out.cpu().numpy())
        feats = np.concatenate(embs, axis=0)
        feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8)
        features[cls_name] = feats
    return features


def evaluate_linear_probe(embeddings: Dict[str, np.ndarray], n_ways: int, n_shots: int, n_episodes: int = 100, seed: int = 42) -> Dict:
    """Transfer-learning baseline: logistic regression trained per episode on
    the K support shots, evaluated on held-out queries. Same episodic protocol
    as the prototypical evaluation, so the comparison is apples-to-apples."""
    eligible = len([c for c, e in embeddings.items() if len(np.asarray(e)) >= n_shots + 1])
    n_ways = min(n_ways, max(2, eligible))
    sampler = EpisodicSampler(embeddings, n_ways, n_shots, seed=seed)
    accuracies = []
    for _ in range(n_episodes):
        ep = sampler.sample_episode()
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(ep["support_x"], ep["support_y"])
        preds = clf.predict(ep["query_x"])
        accuracies.append(float(np.mean(preds == ep["query_y"])))
    mean_acc = float(np.mean(accuracies)) if accuracies else 0.0
    std_acc = float(np.std(accuracies)) if accuracies else 0.0
    return {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "ci_95": float(1.96 * std_acc / np.sqrt(max(len(accuracies), 1))),
        "n_episodes": len(accuracies),
        "n_ways": n_ways,
        "n_shots": n_shots,
    }


def run_ablation(
    images_by_class: Dict[str, List[np.ndarray]],
    dinov2_embeddings: Dict[str, np.ndarray],
    n_ways: int = 5,
    n_shots: int = 5,
    n_episodes: int = 100,
    device: str = "cpu",
) -> Dict:
    """The full ablation table, all rows computed with the same episodes protocol:

    1. ResNet-18 (ImageNet) + nearest centroid  — naive transfer features
    2. ResNet-18 (ImageNet) + linear probe      — classic transfer learning
    3. DINOv2 + linear probe                    — backbone ablation
    4. DINOv2 + prototypical (ours)             — full method
    """
    resnet_feats = extract_resnet18_features(images_by_class, device=device)

    comparison = {
        "protocol": f"{n_ways}-way {n_shots}-shot, {n_episodes} episodes, mean ± std (95% CI)",
        "ResNet18 + NearestCentroid": evaluate_few_shot(resnet_feats, n_ways, n_shots, n_episodes),
        "ResNet18 + LinearProbe (transfer learning)": evaluate_linear_probe(resnet_feats, n_ways, n_shots, n_episodes),
        "DINOv2 + LinearProbe": evaluate_linear_probe(dinov2_embeddings, n_ways, n_shots, n_episodes),
        "DINOv2 + Prototypical (ours)": evaluate_few_shot(dinov2_embeddings, n_ways, n_shots, n_episodes),
        "DINOv2 + Prototypical + OpenSet (ours)": evaluate_few_shot(dinov2_embeddings, n_ways, n_shots, n_episodes, open_set=True),
    }
    return comparison
