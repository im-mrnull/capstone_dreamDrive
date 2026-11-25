"""Chaos-gap motion sanity check (scripts/cg_motion_check.py): a prediction that moves like the
agents passes, one that stays put (or overshoots) fails, the ADE threshold applies on its own,
and the observations come from the dump's own data and split files in canonical row order."""

import json
import sys

import numpy as np
import pytest

from conftest import load_script

mc = load_script("cg_motion_check.py")
TO, TF = 20, 60


def _windows(n=60, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.uniform(-8, 8, (n, 1, 2))
    obs = v * np.arange(-TO + 1, 1)[None, :, None] / 10                 # 10 Hz, last observed frame at 0
    gt = v * np.arange(1, TF + 1)[None, :, None] / 10
    fm = np.ones((n, TF), bool)
    fm[::6, 5:] = False                                                 # some futures end before 1 s
    return obs, gt, fm


def test_constant_velocity_passes_and_a_stationary_prediction_fails():
    obs, gt, fm = _windows()
    good = mc.check(obs, gt, gt, fm, max_ade=10)
    assert good["pass"] and good["median_ratio_moving"] == pytest.approx(1.0) and good["ade"] == pytest.approx(0)
    still = mc.check(obs, np.repeat(obs[:, -1:], TF, 1), gt, fm, max_ade=100)
    assert not still["pass"] and still["median_ratio_moving"] == pytest.approx(0)
    fast = mc.check(obs, 1.5 * gt, gt, fm, max_ade=100)                 # overshooting fails too
    assert not fast["pass"] and fast["median_ratio_moving"] == pytest.approx(1.5)
    assert good["n_moving"] == int(np.sum(fm[:, 9] & (np.linalg.norm(obs[:, -1] - obs[:, -11], axis=1) >= 3)))


def test_the_ade_threshold_applies_on_its_own():
    obs, gt, fm = _windows()
    noisy = gt + np.random.default_rng(1).normal(0, 3, gt.shape)        # right on average, far off per step
    r = mc.check(obs, noisy, gt, fm, max_ade=1.0)
    assert not r["pass"] and r["ade"] > 1.0


def test_main_reads_the_observations_of_the_dumps_rows(tmp_path, monkeypatch):
    obs, gt, fm = _windows(12)
    scene = np.asarray([f"s{i % 3}" for i in range(12)], dtype=object)  # s0 train, s1 val, s2 test
    np.savez(tmp_path / "w.npz", obs=obs.astype(np.float32), fut=gt.astype(np.float32), fut_mask=fm, scene_id=scene)
    (tmp_path / "s.json").write_text(json.dumps({"splits": {"train": ["s0"], "val": ["s1"], "test": ["s2"]}}))
    rows = np.concatenate([np.flatnonzero(scene == "s1"), np.flatnonzero(scene == "s2")])
    np.savez(tmp_path / "p.npz", pred=gt[rows], gt=gt[rows], fut_mask=fm[rows],
             split=np.asarray(["val"] * 4 + ["test"] * 4, dtype=object))
    (tmp_path / "m.json").write_text(json.dumps({"input": str(tmp_path / "w.npz"), "splits_file": str(tmp_path / "s.json")}))
    monkeypatch.setattr(sys, "argv", ["cg_motion_check.py", "--metrics", str(tmp_path / "m.json"),
                                      "--dump", str(tmp_path / "p.npz"), "--max-ade", "1"])
    assert mc.main() == 0
    np.savez(tmp_path / "p.npz", pred=np.repeat(obs[rows, -1:], TF, 1), gt=gt[rows], fut_mask=fm[rows],
             split=np.asarray(["val"] * 4 + ["test"] * 4, dtype=object))
    assert mc.main() == 1
