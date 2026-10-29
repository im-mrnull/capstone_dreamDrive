"""Counterfactual intent branches (scripts/counterfactual_sim.py) on synthetic scenes."""

import numpy as np

from conftest import load_script

cf = load_script("counterfactual_sim.py")
rs, cl = cf.rs, cf.cl

T = 100
T0 = 19


def lead_car():
    """A car 10 m ahead of a stationary ego, driving away along its lane at 5 m/s."""
    t = np.arange(T) * cl.DT
    return np.stack([10.0 + 5.0 * t, np.zeros(T)], 1)


def world(track, cls="Car", dose=cf.DOSES["primary"], branch=None):
    route = cl.Route(np.array([[0.0, 0.0], [600.0, 0.0]]))
    pos = track[:, None, :].astype(float)
    mask = np.ones((T, 1), bool)
    sc = {"sid": "synthetic", "pos": pos, "mask": mask, "cls": np.array([cls]), "agent_id": np.array([7]),
          "heading": cl.agent_headings(pos, mask, route), "route": route, "s_h": np.zeros(T),
          "v_h": np.zeros(T), "T": T}
    log = {"sc": sc, "pos": pos, "mask": mask, "heading": sc["heading"], "vlog": rs.logged_speed(pos, mask)}
    ep = rs.episode_scene(log)
    w = cf.CFWorld(ep, log, T0, "nominal", 0, dose)
    if branch is not None:
        w.branch[0] = branch
    ep["world"] = w
    return ep, w


def drive(ep, w):
    for g in range(T0 + 1, T0 + 1 + cl.EP_LEN):
        w.step(g, np.array([cl.EGO_CENTRE, 0.0]), np.array([1.0, 0.0]), 0.0)
    return ep["pos"][:, 0]


def test_lead_car_is_a_target_and_dose_zero_replays_the_log():
    ep, w = world(lead_car(), dose=(0.0, 1.0), branch="assertive")
    assert w.targets == [0]
    assert np.array_equal(drive(ep, w), lead_car())
    assert not np.isfinite(w.first_dev).any()


def test_hesitation_brakes_comfortably_and_falls_at_least_dose_behind():
    ep, w = world(lead_car(), branch="hesitate")
    tau = w.tau[(0, "hesitate")]
    v = 5.0 * np.diff(tau[T0:T0 + cl.EP_LEN + 1])                 # m per frame -> m/s: 5 m/s x rate
    b_comf = rs.IDM["vehicle"][1]
    assert (np.diff(v) / cl.DT >= -b_comf - 1e-9).all()
    lag = (T0 + cl.EP_LEN - tau[T0 + cl.EP_LEN]) * cl.DT
    assert lag >= cf.DOSES["primary"][0]
    x = drive(ep, w)
    assert x[T0 + cl.EP_LEN, 0] < lead_car()[T0 + cl.EP_LEN, 0] - 5.0     # held back on its own path
    assert np.allclose(x[:, 1], 0.0)


def test_assertive_goes_first_within_its_rate_and_acceleration():
    ep, w = world(lead_car(), branch="assertive")
    tau = w.tau[(0, "assertive")]
    r = np.diff(tau[T0:T0 + cl.EP_LEN + 1])
    dose, rate = cf.DOSES["primary"]
    assert r.max() <= rate + 1e-9 and r.min() >= 1.0 - 1e-9
    a_max = rs.IDM["vehicle"][0]
    assert (np.diff(5.0 * r) / cl.DT <= a_max + 1e-9).all()
    assert (tau[T0 + cl.EP_LEN] - (T0 + cl.EP_LEN)) * cl.DT >= 1.0             # well ahead of its log


def test_branch_draw_is_paired_and_follows_the_prior():
    a = cf.branch_draw("s", 19, "nominal", 0, 3)
    assert a == cf.branch_draw("s", 19, "nominal", 0, 3)
    freq = np.mean([[cf.branch_draw("s", 19, "nominal", d, i) == b for b in cf.BRANCHES]
                    for d in range(20) for i in range(100)], axis=0)
    assert np.allclose(freq, cf.PRIOR, atol=0.04)


def test_ideal_observer_keeps_all_branches_until_one_is_ruled_out():
    ep, w = world(lead_car(), branch="hesitate")
    post0 = w.posterior(0, T0)
    assert post0 == dict(zip(cf.BRANCHES, cf.PRIOR))
    drive(ep, w)
    assert w.posterior(0, T0 + cl.EP_LEN) == {"hesitate": 1.0}
    preds = {n: cf.IdealPredictor(n).at(ep, T0)[0][1] for n in cf.IDEAL}
    assert preds["Ideal set"].shape[0] == 3 and preds["Branch oracle"].shape[0] == 1
    mean = sum(p * w.branch_future(0, b, T0) for b, p in post0.items())
    assert np.allclose(preds["Ideal single (mean)"][0], mean)
    assert np.allclose(preds["Ideal single (most likely)"][0], w.branch_future(0, "logged", T0))
