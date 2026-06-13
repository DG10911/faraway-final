import numpy as np

from backend.models.patchcore_anomaly import PatchCoreDetector, kcenter_greedy_coreset


def healthy_and_defect_features(dim=16, n_healthy=500, seed=0):
    rng = np.random.default_rng(seed)
    center = rng.normal(size=dim)
    center /= np.linalg.norm(center)
    healthy = center + 0.05 * rng.normal(size=(n_healthy, dim))
    defect = -center + 0.05 * rng.normal(size=(20, dim))  # opposite side of the sphere
    return healthy.astype(np.float32), defect.astype(np.float32)


class TestCoreset:
    def test_selects_requested_count(self):
        feats = np.random.rand(200, 8).astype(np.float32)
        core = kcenter_greedy_coreset(feats, 20)
        assert core.shape == (20, 8)

    def test_covers_outliers(self):
        # k-center must pick the single far-away point; random sampling usually misses it
        rng = np.random.default_rng(0)
        cluster = rng.normal(size=(199, 4)) * 0.01
        outlier = np.full((1, 4), 10.0)
        feats = np.concatenate([cluster, outlier]).astype(np.float32)
        core = kcenter_greedy_coreset(feats, 5)
        assert any(np.allclose(c, outlier[0]) for c in core)


class TestPatchCoreDetector:
    def test_detects_defects_and_passes_healthy(self):
        healthy, defect = healthy_and_defect_features()
        det = PatchCoreDetector().fit(healthy, threshold_percentile=99.0)
        healthy_scores = det.score_features(healthy[:50])
        defect_scores = det.score_features(defect)
        assert defect_scores.min() > healthy_scores.max()
        assert defect_scores.min() > det.anomaly_threshold

    def test_threshold_and_decision_same_scale(self):
        healthy, defect = healthy_and_defect_features()
        det = PatchCoreDetector().fit(healthy, threshold_percentile=95.0)
        coords = [(0, 0, 32, 32)] * len(defect)
        result = det.predict(defect, coords, (64, 64))
        # decision score must equal the max raw 1-NN distance, not a smoothed value
        assert np.isclose(result["anomaly_score"], det.score_features(defect).max())
        assert result["is_anomaly"]

    def test_token_features_accepted(self):
        healthy, defect = healthy_and_defect_features()
        det = PatchCoreDetector().fit(healthy)
        tokens = defect[:16].reshape(4, 4, 16)  # 4 patches x 4 tokens
        coords = [(0, 0, 32, 32), (32, 0, 64, 32), (0, 32, 32, 64), (32, 32, 64, 64)]
        result = det.predict(tokens, coords, (64, 64))
        assert result["anomaly_map"].shape == (64, 64)
        assert len(result["patch_scores"]) == 4
        assert result["is_anomaly"]

    def test_recalibrate(self):
        healthy, _ = healthy_and_defect_features()
        det = PatchCoreDetector().fit(healthy, threshold_percentile=95.0)
        t95 = det.anomaly_threshold
        t99 = det.recalibrate(99.0)
        assert t99 >= t95
