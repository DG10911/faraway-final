"""Conformal calibration of the anomaly threshold for guaranteed defect recall.

For a safety-critical inspector, a missed crack costs far more than a false
alarm. Split conformal prediction gives a finite-sample guarantee: pick the
threshold from a held-out calibration set of *defect* anomaly scores so that

    P(new defect's score >= threshold) >= target_recall

holds for any exchangeable data distribution (Vovk et al.; see also conformal
risk control applied to railway signalling, arXiv:2304.06052).
"""

import math
import numpy as np
from typing import Dict


def conformal_recall_threshold(defect_scores: np.ndarray, target_recall: float = 0.95) -> Dict:
    """Threshold with a finite-sample recall guarantee.

    Uses the conformal quantile: with n calibration defects, the threshold is
    the k-th smallest score where k = floor((n + 1) * (1 - target_recall)).
    k < 1 means n is too small for the requested guarantee — the only valid
    threshold is below every observed score (alarm on everything anomalous).
    """
    defect_scores = np.sort(np.asarray(defect_scores, dtype=np.float64))
    n = len(defect_scores)
    if n == 0:
        raise ValueError("Need at least one calibration defect score.")
    k = math.floor((n + 1) * (1.0 - target_recall))
    achievable = k >= 1
    if achievable:
        threshold = float(defect_scores[k - 1])
        guaranteed_recall = 1.0 - k / (n + 1)
    else:
        threshold = float(defect_scores[0] - 1e-6)
        guaranteed_recall = n / (n + 1)
    return {
        "threshold": threshold,
        "target_recall": target_recall,
        "guaranteed_recall": round(guaranteed_recall, 4),
        "n_calibration": n,
        "guarantee_achievable": achievable,
        "note": (
            "P(defect score >= threshold) >= guaranteed_recall, finite-sample, distribution-free"
            if achievable else
            f"Only {n} calibration defects: threshold set below all observed scores; "
            f"collect >= {math.ceil(1.0 / (1.0 - target_recall)) - 1} defects for the full guarantee"
        ),
    }


def evaluate_operating_point(healthy_scores: np.ndarray, defect_scores: np.ndarray, threshold: float) -> Dict:
    """Empirical recall / false-positive rate at a chosen threshold."""
    healthy_scores = np.asarray(healthy_scores)
    defect_scores = np.asarray(defect_scores)
    return {
        "threshold": float(threshold),
        "empirical_recall": float(np.mean(defect_scores >= threshold)) if len(defect_scores) else None,
        "false_positive_rate": float(np.mean(healthy_scores >= threshold)) if len(healthy_scores) else None,
        "n_defect": int(len(defect_scores)),
        "n_healthy": int(len(healthy_scores)),
    }
