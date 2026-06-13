import numpy as np

from backend.utils.conformal import conformal_recall_threshold, evaluate_operating_point


class TestConformalRecall:
    def test_guarantee_holds_empirically(self):
        # Over many resamples, recall on fresh defects must meet the guarantee on average
        rng = np.random.default_rng(0)
        target = 0.9
        hits = []
        for _ in range(200):
            calib = rng.normal(loc=1.0, scale=0.2, size=60)
            test = rng.normal(loc=1.0, scale=0.2, size=200)
            result = conformal_recall_threshold(calib, target)
            hits.append(np.mean(test >= result["threshold"]))
        assert np.mean(hits) >= target - 0.02

    def test_small_n_flags_unachievable(self):
        result = conformal_recall_threshold(np.array([0.5, 0.6, 0.7]), target_recall=0.95)
        assert not result["guarantee_achievable"]
        assert result["threshold"] < 0.5  # below every observed score

    def test_threshold_is_low_quantile_of_defects(self):
        scores = np.linspace(0.5, 1.5, 100)
        result = conformal_recall_threshold(scores, target_recall=0.9)
        assert result["threshold"] <= np.quantile(scores, 0.15)

    def test_operating_point_metrics(self):
        healthy = np.array([0.1, 0.2, 0.3, 0.9])
        defect = np.array([0.5, 0.8, 1.0, 1.2])
        op = evaluate_operating_point(healthy, defect, threshold=0.4)
        assert op["empirical_recall"] == 1.0
        assert op["false_positive_rate"] == 0.25
