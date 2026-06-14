"""Tests for the conformal recall guarantee and synthetic augmentation
(numpy/OpenCV only — no torch needed)."""
import numpy as np
import pytest

from backend.utils.conformal import conformal_recall_threshold, evaluate_operating_point
from backend.datasets.synthetic_augment import synthesize_defect, cutpaste, expand_support, DEFECT_KINDS


def test_conformal_guarantee_holds_on_holdout():
    rng = np.random.default_rng(0)
    # calibration + test defect scores from the same distribution (exchangeable)
    cal = rng.normal(0.35, 0.06, 60)
    test = rng.normal(0.35, 0.06, 400)
    res = conformal_recall_threshold(cal, target_recall=0.9)
    empirical = float(np.mean(test >= res["threshold"]))
    # guaranteed lower bound should hold (with generous slack for finite-sample noise)
    assert res["guaranteed_recall"] >= 0.88
    assert empirical >= res["guaranteed_recall"] - 0.08


def test_conformal_monotonic_threshold():
    rng = np.random.default_rng(1)
    cal = rng.normal(0.4, 0.05, 100)
    t90 = conformal_recall_threshold(cal, 0.90)["threshold"]
    t99 = conformal_recall_threshold(cal, 0.99)["threshold"]
    # higher target recall => lower (more permissive) threshold
    assert t99 <= t90


def test_conformal_too_few_defects():
    res = conformal_recall_threshold(np.array([0.3, 0.4, 0.5]), target_recall=0.99)
    assert res["guarantee_achievable"] is False


def test_operating_point_fields():
    op = evaluate_operating_point(np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5]), 0.35)
    assert op["empirical_recall"] == 1.0
    assert op["false_positive_rate"] == 0.0


def test_synthesize_each_kind_changes_image():
    rng = np.random.default_rng(2)
    healthy = (rng.normal(140, 12, (128, 128, 3)).clip(0, 255)).astype(np.uint8)
    for kind in DEFECT_KINDS:
        out = synthesize_defect(healthy, kind, seed=3)
        assert out.shape == healthy.shape and out.dtype == np.uint8
        assert np.abs(out.astype(int) - healthy.astype(int)).sum() > 0  # actually drew something


def test_cutpaste_and_expand():
    rng = np.random.default_rng(4)
    healthy = (rng.normal(140, 12, (128, 128, 3)).clip(0, 255)).astype(np.uint8)
    defect = (rng.normal(110, 20, (128, 128, 3)).clip(0, 255)).astype(np.uint8)
    cp = cutpaste(healthy, defect, seed=5, seamless=False)
    assert cp.shape == healthy.shape
    grown = expand_support([healthy, healthy], n_per_kind=2)
    assert all(len(v) == 2 for v in grown.values())
