import numpy as np

from backend.models.severity_estimator import SeverityEstimator
from backend.models.open_set_recognizer import OpenSetRecognizer
from backend.twins.digital_twin import DigitalTwinManager


class TestSeverity:
    def test_levels_ordered(self):
        est = SeverityEstimator()
        small_mask = np.zeros((100, 100), dtype=np.uint8)
        big_mask = np.full((100, 100), 255, dtype=np.uint8)
        low = est.estimate(0.1, small_mask, 0.2)
        high = est.estimate(0.95, big_mask, 0.95)
        levels = est.severity_levels
        assert levels.index(high["severity"]) > levels.index(low["severity"])


class TestOpenSetRecognizer:
    def test_far_query_rejected(self):
        rec = OpenSetRecognizer(distance_threshold=0.5, anomaly_score_threshold=0.1)
        protos = {"crack": np.array([1.0, 0.0]), "squat": np.array([0.0, 1.0])}
        label, _ = rec.recognize(np.array([-1.0, -1.0]), protos, anomaly_score=0.9)
        assert label == "unknown_anomaly"
        label, _ = rec.recognize(np.array([0.99, 0.01]), protos, anomaly_score=0.9)
        assert label == "crack"


class TestDigitalTwin:
    def test_report_and_rankings(self):
        twin = DigitalTwinManager()
        r1 = twin.report_defect("Track_A", 100.0, "crack", "Critical", 0.9, 0.8)
        r2 = twin.report_defect("Track_A", 250.0, "squat", "Low", 0.2, 0.6)
        status = twin.get_segment_status("Track_A")
        assert status["active_defects"] == 2
        ranks = status["priority_rankings"]
        assert ranks[0]["defect_type"] == "crack"  # critical outranks low
        assert ranks[0]["priority_rank"] == "#1"
        assert 0.0 <= status["overall_health"] <= 1.0
        assert r1["event_id"] != r2["event_id"]
