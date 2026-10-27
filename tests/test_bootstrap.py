"""Scene-level bootstrap helpers (scripts/dd_bootstrap.py)."""

import numpy as np
import pytest

from conftest import load_script

bs = load_script("dd_bootstrap.py")


def test_ratio_estimate_is_pooled_mean():
    scene = np.array(["a", "a", "b", "c", "c", "c"], dtype=object)
    x = np.array([1.0, 3.0, 10.0, 0.0, 0.0, 6.0])
    codes, scenes = bs.scene_codes(scene)
    counts = bs.bootstrap_counts(len(scenes), 500, seed=0)
    r = bs.ratio_ci(codes, len(scenes), x, np.ones_like(x), counts)
    assert r["estimate"] == pytest.approx(x.mean())
    lo, hi = r["ci95"]
    assert lo <= x.mean() <= hi


def test_counts_resample_scenes_with_replacement():
    counts = bs.bootstrap_counts(22, 1000, seed=1)
    assert counts.shape == (1000, 22)
    assert (counts.sum(axis=1) == 22).all()
    assert (counts == 0).any()            # some scenes left out of some replicates


def test_ci_covers_truth_for_clustered_data():
    """Scene-level resampling is calibrated when scenes carry shared offsets:
    the 95% CI covers the true mean in roughly 95% of simulated datasets."""
    rng = np.random.default_rng(2)
    hits = 0
    for rep in range(200):
        S, m = 22, 40
        offset = rng.normal(scale=1.0, size=S)                 # scene effect
        x = (offset[:, None] + rng.normal(scale=0.5, size=(S, m))).ravel()
        codes = np.repeat(np.arange(S), m)
        counts = bs.bootstrap_counts(S, 400, seed=rep)
        lo, hi = bs.ratio_ci(codes, S, x, np.ones_like(x), counts)["ci95"]
        hits += lo <= 0.0 <= hi
    assert 0.85 <= hits / 200 <= 0.99


def test_scene_codes_share_axis_across_models():
    scenes = np.array(["s1", "s2", "s3"], dtype=object)
    c1, _ = bs.scene_codes(np.array(["s3", "s1"], dtype=object), scenes)
    assert c1.tolist() == [2, 0]
