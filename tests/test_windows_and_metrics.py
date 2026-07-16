"""Invariants of the window datasets and correctness of the metric math."""

from pathlib import Path

import numpy as np
import pytest

from conftest import load_script

cv = load_script("model1_baseline_cv_eval.py")
import dd_data  # noqa: E402

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


@pytest.mark.parametrize("npz_name", ["windows_obs20_pred60.npz", "windows_world_obs20_pred60.npz"])
def test_window_centering_invariant(npz_name):
    path = OUTPUTS / npz_name
    if not path.exists():
        pytest.skip(f"{npz_name} not generated")
    data = np.load(path, allow_pickle=False)
    obs, fut = data["obs"], data["fut"]
    assert obs.shape[1:] == (20, 2) and fut.shape[1:] == (60, 2)
    assert obs.shape[0] == fut.shape[0] == data["scene_id"].shape[0]
    # Every window is centered on the last observed point.
    assert np.abs(obs[:, -1, :]).max() == 0.0
    assert np.isfinite(obs).all() and np.isfinite(fut).all()


def test_cv_is_exact_on_constant_velocity_motion():
    n, obs_len, pred_len = 16, 20, 60
    rng = np.random.default_rng(0)
    vel = rng.uniform(-2, 2, size=(n, 1, 2)).astype(np.float32)
    t_obs = np.arange(-obs_len + 1, 1, dtype=np.float32).reshape(1, obs_len, 1)
    t_fut = np.arange(1, pred_len + 1, dtype=np.float32).reshape(1, pred_len, 1)
    obs = vel * t_obs  # centered: obs[:, -1] == 0
    fut = vel * t_fut
    pred = cv.compute_cv_predictions(obs, pred_len=pred_len, vel_k=1)
    mask = np.ones((n, pred_len), dtype=bool)
    ade, fde, _ = dd_data.masked_ade_fde_np(pred, fut, mask)
    assert ade.max() < 1e-4 and fde.max() < 1e-4


def test_ade_fde_definitions():
    pred = np.zeros((1, 3, 2), dtype=np.float32)
    fut = np.zeros((1, 3, 2), dtype=np.float32)
    fut[0, :, 0] = [3.0, 4.0, 12.0]  # per-step L2 errors: 3, 4, 12
    mask = np.ones((1, 3), dtype=bool)
    ade, fde, _ = dd_data.masked_ade_fde_np(pred, fut, mask)
    assert ade[0] == pytest.approx((3 + 4 + 12) / 3)
    assert fde[0] == pytest.approx(12.0)


def test_stgat_dataset_consistency():
    path = OUTPUTS / "stgat_dataset_world_obs20_pred60.npz"
    if not path.exists():
        pytest.skip("world ST-GAT dataset not generated")
    d = np.load(path, allow_pickle=True)
    sse = d["seq_start_end"]
    assert sse[0, 0] == 0
    assert (sse[1:, 0] == sse[:-1, 1]).all(), "seq_start_end must be contiguous"
    assert sse[-1, 1] == d["obs_traj"].shape[0]
    # Relative displacements must integrate back to absolute positions.
    obs, obs_rel = d["obs_traj"], d["obs_traj_rel"]
    recon = obs[:, :, 0:1] + np.cumsum(obs_rel, axis=-1)
    assert np.allclose(recon[:, :, 1:], obs[:, :, 1:], atol=1e-3)
