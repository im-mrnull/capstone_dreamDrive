"""Split-conformal engine: correctness on synthetic data with known coverage."""

import numpy as np
import pytest

from conftest import load_script

cw = load_script("conformal_wrapper.py")


def test_conformal_radius_order_statistic():
    # scores 1..100, alpha=0.1 -> rank ceil(101*0.9)=91 -> 91st order stat = 91.
    s = np.arange(1, 101, dtype=float)
    assert cw.conformal_radius(s, 0.1) == pytest.approx(91.0)
    # too few samples to guarantee -> infinite radius.
    assert cw.conformal_radius(np.array([1.0, 2.0]), 0.1) == float("inf")
    assert cw.conformal_radius(np.array([]), 0.1) == float("inf")


def test_split_conformal_achieves_nominal_coverage():
    """Exchangeable calibration/eval -> marginal coverage ~ 1 - alpha."""
    rng = np.random.default_rng(0)
    n, T, alpha = 8000, 5, 0.1
    scores = np.abs(rng.normal(size=(n, T))) * (1 + np.arange(T))  # grows with step
    valid = np.ones((n, T), dtype=bool)
    classes = np.array(["A"] * n, dtype=object)
    half = n // 2
    out = cw.evaluate_split(scores[:half], valid[:half], classes[:half],
                            scores[half:], valid[half:], classes[half:],
                            alpha=alpha, min_cal=30)
    assert out["marginal"]["coverage"] == pytest.approx(0.9, abs=0.02)


def test_findings_robust_across_alpha():
    """Nominal coverage + wider-band-for-tighter-target hold at several targets,
    not just alpha=0.1 -- the reported findings aren't artifacts of the 90% choice."""
    rng = np.random.default_rng(3)
    n, T = 8000, 5
    scores = np.abs(rng.normal(size=(n, T))) * (1 + np.arange(T))
    valid = np.ones((n, T), dtype=bool)
    classes = np.array(["A"] * n, dtype=object)
    half = n // 2
    prev = None
    for alpha in (0.2, 0.1, 0.05):                # 80% -> 90% -> 95% target
        out = cw.evaluate_split(scores[:half], valid[:half], classes[:half],
                                scores[half:], valid[half:], classes[half:],
                                alpha=alpha, min_cal=30)
        assert out["marginal"]["coverage"] == pytest.approx(1 - alpha, abs=0.02)
        radii = np.array(out["marginal_radius_by_step"])
        if prev is not None:
            assert np.all(radii >= prev - 1e-9)  # tighter target -> wider band
        prev = radii


def test_class_conditional_balances_heterogeneous_classes():
    """One global radius under-covers the wide class and over-covers the tight
    one; per-class radii pull both back toward nominal."""
    rng = np.random.default_rng(1)
    T, alpha = 4, 0.1
    tight = np.abs(rng.normal(scale=1.0, size=(4000, T)))
    wide = np.abs(rng.normal(scale=6.0, size=(4000, T)))
    scores = np.vstack([tight, wide])
    classes = np.array(["tight"] * 4000 + ["wide"] * 4000, dtype=object)
    valid = np.ones_like(scores, dtype=bool)
    idx = rng.permutation(scores.shape[0])
    cal, ev = idx[:4000], idx[4000:]
    out = cw.evaluate_split(scores[cal], valid[cal], classes[cal],
                            scores[ev], valid[ev], classes[ev], alpha, min_cal=30)
    marg = out["marginal"]["per_class_coverage"]
    cc = out["class_conditional"]["per_class_coverage"]
    # marginal: the wide class is under-covered, the tight class over-covered.
    assert marg["wide"] < 0.9 < marg["tight"]
    # class-conditional pulls BOTH toward nominal (within a few points).
    assert cc["wide"] == pytest.approx(0.9, abs=0.03)
    assert cc["tight"] == pytest.approx(0.9, abs=0.03)


def test_small_class_falls_back_to_marginal():
    rng = np.random.default_rng(2)
    T = 3
    big = np.abs(rng.normal(size=(2000, T)))
    small = np.abs(rng.normal(size=(5, T)))          # below min_cal
    scores = np.vstack([big, small])
    classes = np.array(["big"] * 2000 + ["rare"] * 5, dtype=object)
    valid = np.ones_like(scores, dtype=bool)
    marg = cw.radii_per_timestep(scores, valid, 0.1)
    pc = cw.radii_per_class(scores, valid, classes, 0.1, marg, min_cal=30)
    assert np.allclose(pc["rare"], marg)      # rare class uses the marginal radius
    assert not np.allclose(pc["big"], marg)   # big class gets its own
