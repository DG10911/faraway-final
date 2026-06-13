import numpy as np
import pytest

from backend.models.prototypical_network import (
    EpisodicSampler,
    PrototypicalNetwork,
    evaluate_few_shot,
)


def make_clustered_embeddings(n_classes=5, n_per_class=30, dim=32, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return {
        f"class_{i}": centers[i] + spread * rng.normal(size=(n_per_class, dim))
        for i in range(n_classes)
    }


class TestEpisodicSampler:
    def test_support_query_disjoint(self):
        embs = make_clustered_embeddings()
        sampler = EpisodicSampler(embs, n_ways=3, n_shots=5, seed=1)
        ep = sampler.sample_episode()
        # disjointness: no support row appears among query rows
        for s in ep["support_x"]:
            assert not any(np.allclose(s, q) for q in ep["query_x"])

    def test_exact_shot_count(self):
        embs = make_clustered_embeddings()
        sampler = EpisodicSampler(embs, n_ways=4, n_shots=5, seed=1)
        ep = sampler.sample_episode()
        for label in range(4):
            assert np.sum(ep["support_y"] == label) == 5  # never silently reduced

    def test_rejects_too_small_classes(self):
        embs = {"a": np.random.rand(3, 8), "b": np.random.rand(3, 8)}
        with pytest.raises(ValueError):
            EpisodicSampler(embs, n_ways=2, n_shots=5)

    def test_open_set_episode_has_unknown_queries(self):
        embs = make_clustered_embeddings(n_classes=4)
        sampler = EpisodicSampler(embs, n_ways=3, n_shots=5, seed=1)
        ep = sampler.sample_episode(n_extra_open_set=1)
        assert (ep["query_y"] == -1).sum() > 0
        assert len(ep["class_names"]) == 3
        assert len(ep["unknown_classes"]) == 1
        # unknown class contributed no support
        assert set(np.unique(ep["support_y"])) == {0, 1, 2}


class TestPrototypicalNetwork:
    def test_separable_classes_high_accuracy(self):
        embs = make_clustered_embeddings(spread=0.02)
        result = evaluate_few_shot(embs, n_ways=5, n_shots=5, n_episodes=20)
        assert result["mean_accuracy"] > 0.95
        assert result["n_episodes"] == 20
        assert "ci_95" in result

    def test_open_set_unknowns_not_scored_as_class_zero(self):
        embs = make_clustered_embeddings(n_classes=5, spread=0.02)
        result = evaluate_few_shot(embs, n_ways=3, n_shots=5, n_episodes=20, open_set=True)
        # tight clusters far apart -> unknowns should mostly be rejected
        assert result["open_set_detection_rate"] > 0.5
        assert result["false_unknown_rate"] < 0.5

    def test_prototypes_are_normalized(self):
        net = PrototypicalNetwork(embedding_dim=8, n_ways=2, n_shots=3)
        support = np.random.rand(6, 8) * 10  # deliberately unnormalized
        labels = np.array([0, 0, 0, 1, 1, 1])
        net.compute_prototypes(support, labels, ["a", "b"])
        for proto in net.prototypes.values():
            assert np.isclose(np.linalg.norm(proto), 1.0, atol=1e-5)

    def test_calibrated_threshold_positive(self):
        net = PrototypicalNetwork(embedding_dim=8, n_ways=2, n_shots=3)
        support = np.random.rand(6, 8)
        labels = np.array([0, 0, 0, 1, 1, 1])
        net.compute_prototypes(support, labels, ["a", "b"])
        assert net.calibrated_threshold() > 0

    def test_classify_returns_known_labels(self):
        embs = make_clustered_embeddings(n_classes=3, spread=0.02)
        sampler = EpisodicSampler(embs, n_ways=3, n_shots=5, seed=0)
        ep = sampler.sample_episode()
        net = PrototypicalNetwork(embedding_dim=32, n_ways=3, n_shots=5)
        net.compute_prototypes(ep["support_x"], ep["support_y"], ep["class_names"])
        preds, conf = net.classify(ep["query_x"])
        assert set(preds).issubset({0, 1, 2})
        assert np.all((conf > 0) & (conf <= 1))
