"""Best-of-K decomposition curve helpers in script 7 (pure-numpy, no torch)."""

import numpy as np
import pytest

from conftest import load_script

t7 = load_script("7_train_eval_trajectronpp.py")


def test_best_of_curve_monotone_and_endpoints():
    rng = np.random.default_rng(0)
    ade_s = rng.uniform(0.5, 5.0, size=20)
    curve = t7.best_of_curve(ade_s, k_list=(1, 2, 5, 10, 20))
    # best-of-1 is the first sample; best-of-20 is the global min.
    assert curve[0] == pytest.approx(ade_s[0])
    assert curve[-1] == pytest.approx(ade_s.min())
    # monotone non-increasing in K.
    for a, b in zip(curve, curve[1:]):
        assert b <= a + 1e-12
    # mean-over-samples is never better than any best-of-K.
    assert float(ade_s.mean()) >= curve[-1] - 1e-12


def test_best_of_curve_clamps_k_to_sample_count():
    ade_s = np.array([3.0, 1.0, 2.0])  # only 3 samples
    curve = t7.best_of_curve(ade_s, k_list=(1, 2, 5, 10, 20))
    assert curve[0] == pytest.approx(3.0)
    # every K >= 3 collapses to the min of the 3 available samples.
    assert all(c == pytest.approx(1.0) for c in curve[2:])


def test_node_sample_errors_matches_min_metrics():
    # 3 samples, 4 future steps, last step invalid (NaN).
    fut = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [np.nan, np.nan]])
    pred = np.zeros((3, 4, 2))
    pred[0, :, 0] = [1.0, 2.0, 3.0, 0.0]   # sample 0 perfect on valid steps
    pred[1, :, 0] = [0.0, 0.0, 0.0, 0.0]   # sample 1 off
    pred[2, :, 0] = [2.0, 2.0, 2.0, 0.0]
    ade_s, fde_s, path = t7._node_sample_errors(pred, fut)
    assert ade_s.shape == (3,) and fde_s.shape == (3,)
    assert ade_s[0] == pytest.approx(0.0)          # sample 0 is exact
    # _node_metrics returns the min-over-samples, matching ade_s.min().
    m = t7._node_metrics(pred, fut)
    assert m[0] == pytest.approx(ade_s.min())
    assert m[1] == pytest.approx(fde_s.min())
    # path over the valid consecutive pairs (steps 0->1->2): 1 + 1 = 2.
    assert path == pytest.approx(2.0)


def test_curve_per_class_endpoints():
    curve = np.array([[4.0, 2.0], [2.0, 1.0], [6.0, 3.0]])   # (N, 2): K=1, K=max
    mean_s = np.array([3.0, 1.5, 4.5])
    types = np.array(["Car", "Car", "Pedestrian"], dtype=object)
    out = t7._curve_per_class(curve, mean_s, types, k_list=(1, 20))
    assert out["classes"]["Car"]["n"] == 2
    assert out["classes"]["Car"]["best_of_1"] == pytest.approx(3.0)   # (4+2)/2
    assert out["classes"]["Car"]["best_of_max"] == pytest.approx(1.5)  # (2+1)/2
    assert out["classes"]["Pedestrian"]["mean_sample"] == pytest.approx(4.5)
