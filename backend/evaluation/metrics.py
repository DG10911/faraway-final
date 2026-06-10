import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)


class Evaluator:
    def __init__(self):
        self.metrics = {}

    def compute_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None) -> Dict:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        if y_score is not None and len(np.unique(y_true)) == 2:
            metrics["auroc"] = float(roc_auc_score(y_true, y_score))
            metrics["auprc"] = float(average_precision_score(y_true, y_score))
        self.metrics.update(metrics)
        return metrics

    def compute_few_shot_metrics(self, episode_results: List[float]) -> Dict:
        return {
            "mean_accuracy": float(np.mean(episode_results)),
            "std_accuracy": float(np.std(episode_results)),
            "ci_95": float(1.96 * np.std(episode_results) / np.sqrt(len(episode_results))),
            "n_episodes": len(episode_results),
        }

    def compute_anomaly_detection_metrics(self, y_true: np.ndarray, anomaly_scores: np.ndarray) -> Dict:
        metrics = {}
        if len(np.unique(y_true)) == 2:
            metrics["auroc"] = float(roc_auc_score(y_true, anomaly_scores))
            metrics["auprc"] = float(average_precision_score(y_true, anomaly_scores))
        thresholds = np.linspace(anomaly_scores.min(), anomaly_scores.max(), 100)
        best_f1 = 0
        best_threshold = thresholds[0]
        for t in thresholds:
            y_pred = (anomaly_scores > t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        metrics["best_f1"] = float(best_f1)
        metrics["best_threshold"] = float(best_threshold)
        y_pred_opt = (anomaly_scores > best_threshold).astype(int)
        metrics["precision_opt"] = float(precision_score(y_true, y_pred_opt, zero_division=0))
        metrics["recall_opt"] = float(recall_score(y_true, y_pred_opt, zero_division=0))
        return metrics

    def compute_cross_domain_metrics(self, source_metrics: Dict, target_metrics: Dict) -> Dict:
        return {
            "source_domain": source_metrics,
            "target_domain": target_metrics,
            "domain_gap": {
                k: float(source_metrics.get(k, 0) - target_metrics.get(k, 0))
                for k in ["accuracy", "f1_macro"]
                if k in source_metrics and k in target_metrics
            },
        }
