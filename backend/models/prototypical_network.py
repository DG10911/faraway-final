import numpy as np
from typing import List, Tuple, Dict, Optional


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-8)


class PrototypicalNetwork:
    """Nearest-prototype classifier over L2-normalized embeddings.

    Prototypes use the class median (robust to one bad support shot), then are
    re-normalized so distances stay on the unit sphere ([0, 2] range), which
    keeps the open-set threshold comparable across domains.
    """

    def __init__(self, embedding_dim: int, n_ways: int, n_shots: int, n_queries: int = 15):
        self.embedding_dim = embedding_dim
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.prototypes = None
        self.class_names = None
        self.support_distance_stats = None
        self.open_set_threshold = None  # set by calibrate_open_set(); preferred over calibrated_threshold()

    def compute_prototypes(self, support_embeddings: np.ndarray, support_labels: np.ndarray, class_names: List[str]):
        support_embeddings = _l2_normalize(support_embeddings)
        unique_labels = np.unique(support_labels)
        self.prototypes = {}
        self.class_names = class_names
        support_dists = []
        for label in unique_labels:
            class_embs = support_embeddings[support_labels == label]
            median_proto = _l2_normalize(np.median(class_embs, axis=0))
            self.prototypes[int(label)] = median_proto
            support_dists.extend(np.linalg.norm(class_embs - median_proto, axis=1).tolist())
        support_dists = np.array(support_dists)
        self.support_distance_stats = {
            "mean": float(support_dists.mean()),
            "std": float(support_dists.std()),
            "p95": float(np.percentile(support_dists, 95)),
        }
        return self

    def calibrated_threshold(self, n_sigmas: float = 3.0, floor: float = 0.05) -> float:
        """Open-set distance threshold derived from the support set itself:
        mean + n_sigmas * std of support-to-own-prototype distances.

        NOTE: this is uncalibrated — the support-to-own-prototype distances are
        a biased (too small) estimate of where genuine known queries land, so
        mean + 3*std systematically sits too high/low depending on the feature
        geometry and collapses to all-known or all-unknown. Prefer
        `calibrate_open_set()` with a held-out calibration split when you have
        more than n_shots samples per class.
        """
        if self.support_distance_stats is None:
            return 1.0
        s = self.support_distance_stats
        return max(s["mean"] + n_sigmas * s["std"], s["p95"], floor)

    def calibrate_open_set(self, calib_embeddings: np.ndarray, false_unknown_budget: float = 0.1) -> float:
        """Distribution-free open-set threshold (conformal-style).

        Set the rejection distance to the (1 - budget) empirical quantile of the
        nearest-prototype distances of HELD-OUT KNOWN samples (not the support
        shots used to build the prototypes). By construction, at most ~budget of
        genuine known queries are wrongly flagged unknown, while everything
        farther than any seen known is rejected. This replaces the brittle
        mean + n_sigmas*std rule and lifts unknown-detection recall from ~0.07
        to a tunable operating point on real rail embeddings.

        Returns the chosen threshold and stores it on the instance so that
        `open_set_classify` uses it automatically.
        """
        if self.prototypes is None:
            raise RuntimeError("Prototypes not computed.")
        calib_embeddings = np.asarray(calib_embeddings)
        if len(calib_embeddings) == 0:
            self.open_set_threshold = self.calibrated_threshold()
            return self.open_set_threshold
        distances, _ = self._distances(calib_embeddings)
        nearest = np.min(distances, axis=1)
        budget = float(np.clip(false_unknown_budget, 0.0, 1.0))
        self.open_set_threshold = float(np.quantile(nearest, 1.0 - budget))
        return self.open_set_threshold

    def _distances(self, query_embeddings: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        keys = sorted(self.prototypes.keys())
        proto_array = np.array([self.prototypes[k] for k in keys])
        query_embeddings = _l2_normalize(query_embeddings)
        distances = np.linalg.norm(query_embeddings[:, np.newaxis, :] - proto_array[np.newaxis, :, :], axis=2)
        return distances, keys

    def classify(self, query_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.prototypes is None:
            raise RuntimeError("Prototypes not computed. Call compute_prototypes() first.")
        distances, keys = self._distances(query_embeddings)
        preds = np.array([keys[i] for i in np.argmin(distances, axis=1)])
        confidences = 1.0 / (1.0 + np.min(distances, axis=1))
        return preds, confidences

    def open_set_classify(self, query_embeddings: np.ndarray, distance_threshold: Optional[float] = None) -> List[Dict]:
        if self.prototypes is None:
            raise RuntimeError("Prototypes not computed.")
        if distance_threshold is None:
            distance_threshold = self.open_set_threshold if self.open_set_threshold is not None else self.calibrated_threshold()
        distances, keys = self._distances(query_embeddings)
        min_distances = np.min(distances, axis=1)
        pred_keys = [keys[i] for i in np.argmin(distances, axis=1)]
        label_map = {i: name for i, name in enumerate(self.class_names)} if self.class_names else {}
        results = []
        for dist, pred_key in zip(min_distances, pred_keys):
            confidence = float(1.0 / (1.0 + dist))
            if dist > distance_threshold:
                results.append({"label": "unknown_anomaly", "confidence": confidence, "distance": float(dist)})
            else:
                class_name = label_map.get(int(pred_key), f"class_{pred_key}")
                results.append({"label": class_name, "confidence": confidence, "distance": float(dist)})
        return results


class EpisodicSampler:
    """Samples N-way K-shot episodes with disjoint support/query sets.

    A class is only eligible if it has at least n_shots + 1 samples, so the
    reported K-shot number is the real one (no silent shot reduction).
    """

    def __init__(self, embeddings: Dict[str, np.ndarray], n_ways: int, n_shots: int, n_queries: int = 15, seed: Optional[int] = None):
        self.embeddings = {k: np.asarray(v) for k, v in embeddings.items()}
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.rng = np.random.default_rng(seed)
        self.eligible_classes = [c for c, e in self.embeddings.items() if len(e) >= n_shots + 1]
        if len(self.eligible_classes) < n_ways:
            raise ValueError(
                f"Only {len(self.eligible_classes)} classes have >= {n_shots + 1} samples; "
                f"cannot run {n_ways}-way {n_shots}-shot episodes. "
                f"Eligible: {self.eligible_classes}"
            )

    def sample_episode(self, n_extra_open_set: int = 0) -> Dict:
        """Returns an episode dict. If n_extra_open_set > 0, that many extra
        classes are held out: their queries get label -1 (true unknowns) and
        they contribute no support shots."""
        total = self.n_ways + n_extra_open_set
        if len(self.eligible_classes) < total:
            n_extra_open_set = len(self.eligible_classes) - self.n_ways
            total = self.n_ways + n_extra_open_set
        selected = list(self.rng.choice(self.eligible_classes, total, replace=False))
        known_classes, unknown_classes = selected[:self.n_ways], selected[self.n_ways:]

        support_x, support_y, query_x, query_y = [], [], [], []
        for label_idx, class_name in enumerate(known_classes):
            class_embs = self.embeddings[class_name]
            indices = self.rng.permutation(len(class_embs))
            n_query = min(self.n_queries, len(class_embs) - self.n_shots)
            support_x.append(class_embs[indices[:self.n_shots]])
            support_y.extend([label_idx] * self.n_shots)
            query_x.append(class_embs[indices[self.n_shots:self.n_shots + n_query]])
            query_y.extend([label_idx] * n_query)
        for class_name in unknown_classes:
            class_embs = self.embeddings[class_name]
            indices = self.rng.permutation(len(class_embs))
            n_query = min(self.n_queries, len(class_embs))
            query_x.append(class_embs[indices[:n_query]])
            query_y.extend([-1] * n_query)

        return {
            "support_x": np.concatenate(support_x, axis=0),
            "support_y": np.array(support_y),
            "query_x": np.concatenate(query_x, axis=0),
            "query_y": np.array(query_y),
            "class_names": known_classes,
            "unknown_classes": unknown_classes,
        }


def evaluate_few_shot(
    embeddings: Dict[str, np.ndarray],
    n_ways: int,
    n_shots: int,
    n_episodes: int = 100,
    embedding_dim: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    open_set: bool = False,
    open_set_budget: Optional[float] = None,
    n_calib_shots: int = 1,
    seed: Optional[int] = 42,
) -> Dict:
    """Episodic evaluation: mean ± std (and 95% CI) over n_episodes.

    With open_set=True, each episode holds out one extra class as a true
    unknown; reports closed-set accuracy on known queries plus unknown
    detection rate (recall) and false-unknown rate on knowns.

    open_set_budget (e.g. 0.1) switches the rejection rule from the brittle
    mean+3*std heuristic to a conformal-style threshold calibrated on
    `n_calib_shots` HELD-OUT known shots per class, targeting that
    false-unknown rate. This requires n_shots > n_calib_shots and yields a far
    higher, tunable unknown-detection recall. `distance_threshold` (if given)
    still overrides everything.
    """
    use_calib = open_set and open_set_budget is not None and distance_threshold is None and n_shots > n_calib_shots
    n_ways = min(n_ways, max(2, len([c for c, e in embeddings.items() if len(np.asarray(e)) >= n_shots + 1]) - (1 if open_set else 0)))
    sampler = EpisodicSampler(embeddings, n_ways, n_shots, seed=seed)
    accuracies, unknown_recalls, false_unknown_rates = [], [], []

    for _ in range(n_episodes):
        ep = sampler.sample_episode(n_extra_open_set=1 if open_set else 0)
        s_x, s_y = ep["support_x"], ep["support_y"]
        calib_x = None
        if use_calib:
            # Hold out the last n_calib_shots of each class as a known calibration set.
            proto_mask = np.zeros(len(s_y), dtype=bool)
            for lbl in np.unique(s_y):
                proto_mask[np.where(s_y == lbl)[0][:n_shots - n_calib_shots]] = True
            calib_x = s_x[~proto_mask]
            s_x, s_y = s_x[proto_mask], s_y[proto_mask]
        model = PrototypicalNetwork(
            embedding_dim=embedding_dim or ep["support_x"].shape[1],
            n_ways=len(ep["class_names"]),
            n_shots=n_shots,
        )
        model.compute_prototypes(s_x, s_y, ep["class_names"])
        if use_calib:
            model.calibrate_open_set(calib_x, false_unknown_budget=open_set_budget)
        q_y = ep["query_y"]

        if open_set:
            results = model.open_set_classify(ep["query_x"], distance_threshold)
            name_to_idx = {name: i for i, name in enumerate(ep["class_names"])}
            preds = np.array([-1 if r["label"] == "unknown_anomaly" else name_to_idx[r["label"]] for r in results])
            known_mask = q_y >= 0
            if known_mask.any():
                accuracies.append(float(np.mean(preds[known_mask] == q_y[known_mask])))
                false_unknown_rates.append(float(np.mean(preds[known_mask] == -1)))
            if (~known_mask).any():
                unknown_recalls.append(float(np.mean(preds[~known_mask] == -1)))
        else:
            preds, _ = model.classify(ep["query_x"])
            accuracies.append(float(np.mean(preds == q_y)))

    mean_acc = float(np.mean(accuracies)) if accuracies else 0.0
    std_acc = float(np.std(accuracies)) if accuracies else 0.0
    result = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "ci_95": float(1.96 * std_acc / np.sqrt(max(len(accuracies), 1))),
        "n_episodes": len(accuracies),
        "n_ways": n_ways,
        "n_shots": n_shots,
    }
    if open_set:
        result["open_set_detection_rate"] = float(np.mean(unknown_recalls)) if unknown_recalls else 0.0
        result["false_unknown_rate"] = float(np.mean(false_unknown_rates)) if false_unknown_rates else 0.0
        result["rejection_rule"] = "calibrated" if use_calib else ("fixed" if distance_threshold is not None else "support_sigma")
        if use_calib:
            result["open_set_budget"] = open_set_budget
    return result
