import numpy as np
from typing import Dict, Tuple


class SeverityEstimator:
    def __init__(self):
        self.severity_levels = ["Low", "Medium", "High", "Critical"]

    def estimate(self, anomaly_score: float, anomaly_mask: np.ndarray, confidence: float) -> Dict:
        defect_area = float(np.sum(anomaly_mask > 0) / anomaly_mask.size) if anomaly_mask.size > 0 else 0.0
        defect_density = defect_area * anomaly_score
        composite = 0.3 * anomaly_score + 0.3 * defect_area * 10 + 0.2 * confidence + 0.2 * defect_density
        severity = self._map_severity(composite)
        return {
            "severity": severity,
            "defect_area": round(defect_area, 4),
            "anomaly_score": round(anomaly_score, 4),
            "confidence": round(confidence, 4),
            "defect_density": round(defect_density, 4),
            "composite_score": round(composite, 4),
        }

    def _map_severity(self, composite: float) -> str:
        if composite < 0.25:
            return "Low"
        elif composite < 0.50:
            return "Medium"
        elif composite < 0.75:
            return "High"
        else:
            return "Critical"
