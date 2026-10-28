"""Multi-future scorecard metrics (scripts/multifuture_scorecard.py) on toy inputs."""

import numpy as np
import pytest

from conftest import load_script

mf = load_script("multifuture_scorecard.py")


def one_step(points):
    """(1, K, 60, 2) samples with only the first future step meaningful."""
    s = np.zeros((1, len(points), 60, 2))
    s[0, :, 0] = points
    return s


def test_energy_score_matches_hand_computation():
    fm = np.zeros((1, 60), bool); fm[0, 0] = True
    gt = np.zeros((1, 60, 2)); gt[0, 0] = (1.0, 0.0)
    samples = one_step([(0.0, 0.0), (2.0, 0.0)])
    # E||X - y|| = 1, E||X - X'|| = (0 + 2 + 2 + 0) / 4 = 1  ->  1 - 0.5 * 1
    assert mf.energy_score(samples, np.full((1, 2), 0.5), gt, fm)[0] == pytest.approx(0.5)
    # all weight on a sample at the truth scores 0
    assert mf.energy_score(one_step([(1.0, 0.0), (5.0, 0.0)]), np.array([[1.0, 0.0]]), gt, fm)[0] == pytest.approx(0.0)


def test_ece_is_zero_when_confidence_matches_accuracy():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.05, 0.95, 200_000)
    acc = (rng.uniform(size=conf.size) < conf).astype(float)
    b = np.clip(np.digitize(conf, mf.ECE_BINS) - 1, 0, len(mf.ECE_BINS) - 2)
    sums = {}
    for k in range(len(mf.ECE_BINS) - 1):
        sel = b == k
        sums[f"ece_n{k}"], sums[f"ece_conf{k}"], sums[f"ece_acc{k}"] = (np.array([sel.sum()], float),
                                                                       np.array([conf[sel].sum()]),
                                                                       np.array([acc[sel].sum()]))
    assert mf.ece(sums.get) < 0.01
    over = {k: (v * 0.0 + sums[k.replace("acc", "n")] if "acc" in k else v) for k, v in sums.items()}
    assert mf.ece(over.get) > 0.3                     # always right while ~50% confident


def test_implausible_flags_teleporting_samples():
    t = np.arange(1, 61) * mf.DT
    walk = np.stack([1.2 * t, 0 * t], -1)[None, None]                 # 1.2 m/s pedestrian
    jump = walk.copy(); jump[..., 30:, 0] += 20.0                       # 20 m teleport
    s = np.concatenate([walk, jump], axis=1)
    assert mf.implausible(s, np.array(["Pedestrian"]))[0] == pytest.approx(0.5)
