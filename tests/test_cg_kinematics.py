"""Chaos-gap kinematics (scripts/cg_kinematics.py): known motions give known statistics, and
missing future frames never create spurious steps."""

import numpy as np

from conftest import load_script

kin = load_script("cg_kinematics.py")
TO, TF = 20, 60


def _split(traj, fut_mask=None):
    """(N, T, 2) positions -> obs (N, 2, To), pred (N, 2, Tf), fut_mask (N, Tf)."""
    traj = np.asarray(traj, dtype=np.float64)
    fm = np.ones((len(traj), TF), bool) if fut_mask is None else fut_mask
    return traj[:, :TO].transpose(0, 2, 1), traj[:, TO:].transpose(0, 2, 1), fm


def test_straight_line_at_constant_speed():
    t = np.arange(TO + TF) * kin.DT
    f = kin.agent_features(*_split(np.stack([5.0 * t, np.zeros_like(t)], -1)[None]))
    assert np.allclose(f["speed"], 5.0) and np.allclose(f["accel_p90"], 0, atol=1e-9)
    assert np.allclose(f["yaw_rate"], 0, atol=1e-9) and np.allclose(f["jitter"], 0, atol=1e-9)
    assert f["stationary"][0] == 0 and f["fut_valid"][0] == 1


def test_circle_gives_speed_over_radius():
    t = np.arange(TO + TF) * kin.DT
    r, w = 20.0, 0.25                                                      # 5 m/s on a 20 m circle
    f = kin.agent_features(*_split(np.stack([r * np.cos(w * t), r * np.sin(w * t)], -1)[None]))
    assert abs(f["speed"][0] / (r * w) - 1) < 0.01 and abs(f["yaw_rate"][0] - w) < 1e-3   # 1 s chords
    assert abs(f["accel_p90"][0] / (r * w ** 2) - 1) < 0.01               # centripetal


def test_standing_still_and_white_noise_jitter():
    rng = np.random.default_rng(0)
    sd = 0.05
    f = kin.agent_features(*_split(rng.normal(0, sd, (400, TO + TF, 2))))
    assert np.all(f["stationary"] == 1)
    assert abs(np.median(f["jitter"]) - np.sqrt(6) * sd * np.sqrt(2)) < 0.02   # 2-D norm of sqrt(6) sd per axis
    assert np.isnan(kin.agent_features(*_split(np.zeros((1, TO + TF, 2))))["yaw_rate"][0])


def test_missing_future_frames_are_skipped():
    t = np.arange(TO + TF) * kin.DT
    traj = np.stack([3.0 * t, np.zeros_like(t)], -1)[None].copy()
    fm = np.ones((1, TF), bool)
    fm[0, 10:25] = False
    traj[0, TO + 10:TO + 25] = 0.0                                         # garbage where absent
    f = kin.agent_features(*_split(traj, fm))
    assert np.allclose(f["speed"], 3.0) and np.allclose(f["jitter"], 0, atol=1e-9)
    assert np.isclose(f["fut_valid"][0], 45 / 60)
