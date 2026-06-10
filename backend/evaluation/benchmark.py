import json
import time
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from .metrics import Evaluator


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


ABLATIONS = ["Baseline CNN", "Transfer Learning", "RailGuard-FSL++"]


def create_ablation_comparison() -> Dict:
    runner = BenchmarkRunner()
    return {
        "ablation_study": ABLATIONS,
        "description": "Cross-domain comparison: Train on Mendeley, adapt to RSDDs/Fastener with 5 support images",
    }
