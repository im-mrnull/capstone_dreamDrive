"""Lateral intent branches (scripts/counterfactual_sim.py, path_lateral) on synthetic lateral tracks."""

import numpy as np

from conftest import load_script

cf = load_script("counterfactual_sim.py")
DT = cf.cl.DT
EDGE = 2.0
OK = np.ones(120, bool)


def crossing(d0=8.0, v=-1.4, T=120):
    """A road user approaching / crossing the route from d0 at v m/s (frames 0..T-1)."""
    return d0 + v * DT * np.arange(T)


def own_limits_ok(d, t_div, vmax, a):
    """The branch's own lateral speed and acceleration stay within the limits (10% slack for the
    frame that snaps onto a target)."""
    w = np.diff(d[t_div:]) / DT
    return np.abs(w).max() <= vmax + 1e-6 and np.abs(np.diff(w)).max() / DT <= 1.1 * a


def no_reversal(d, t_div):
    w = np.diff(d[t_div:])
    return not ((w > 1e-9).any() and (w < -1e-9).any())


def test_logged_and_zero_limits_are_the_log():
    d = crossing()
    for ctx in ("same", "crossing"):
        for b in cf.PATH_BRANCHES:
            assert np.array_equal(cf.path_lateral(b, ctx, d, OK, 20, EDGE, 0.0, 0.0), d)
        assert np.array_equal(cf.path_lateral("logged", ctx, d, OK, 20, EDGE, 1.0, 1.0), d)


def test_stay_out_waits_at_the_edge():
    d = crossing()
    s = cf.path_lateral("stay out", "crossing", d, OK, 20, EDGE, 1.0, 1.0)
    assert np.array_equal(s[:21], d[:21])                    # the log until it must brake
    assert s.min() >= EDGE - 0.01 and abs(s[-1] - EDGE) < 0.01   # stops at the edge (discretisation)
    assert own_limits_ok(s, 20, 1.4, 1.0) and no_reversal(s, 20)


def test_stay_out_braking_late_holds_where_it_stops():
    d = crossing(d0=5.0)                                     # only 0.2 m from the edge at the branch start
    s = cf.path_lateral("stay out", "crossing", d, OK, 20, EDGE, 1.0, 1.0)
    assert EDGE - 1.0 < s.min() < EDGE and s[-1] == s.min()  # dips in, then holds: never backs out
    assert own_limits_ok(s, 20, 1.4, 1.0) and no_reversal(s, 20)


def test_stay_out_inside_the_corridor_holds_its_offset():
    d = np.linspace(-0.5, 0.5, 120)                          # drifting across the centreline
    s = cf.path_lateral("stay out", "same", d, OK, 20, EDGE, 1.0, 1.0)
    assert abs(s[-1] - d[20]) < 0.1 and no_reversal(s, 20)


def test_cut_in_reaches_the_centreline_within_the_limits():
    d = np.full(120, 4.0)                                    # passes alongside, 4 m to the left
    m = cf.path_lateral("enter", "same", d, OK, 20, EDGE, 1.5, 2.0)
    assert np.array_equal(m[:21], d[:21]) and abs(m[-1]) < 1e-9
    assert own_limits_ok(m, 20, 1.5, 2.0) and no_reversal(m, 20)


def test_cross_goes_to_the_far_edge():
    d = np.full(120, 3.0)                                    # a pedestrian waiting at the kerb
    c = cf.path_lateral("enter", "crossing", d, OK, 20, EDGE, 1.0, 1.0)
    assert abs(c[-1] + EDGE) < 1e-9 and own_limits_ok(c, 20, 1.0, 1.0) and no_reversal(c, 20)


def test_no_enter_branch_when_moving_away():
    d = crossing(d0=3.0, v=+0.5)                             # moving away from the route
    for ctx in ("same", "crossing"):
        assert np.array_equal(cf.path_lateral("enter", ctx, d, OK, 20, EDGE, 1.0, 1.0), d)


def test_context_from_heading():
    tan = np.array([1.0, 0.0])
    assert cf.context_of(np.array([0.9, 0.3]) / np.hypot(0.9, 0.3), tan) == "same"
    assert cf.context_of(np.array([0.0, 1.0]), tan) == "crossing"
    assert cf.context_of(np.array([-1.0, 0.1]) / np.hypot(1.0, 0.1), tan) == "oncoming"


def test_branch_draw_uses_the_given_prior():
    for prior in cf.PRIOR_VARIANTS.values():
        draws = [cf.branch_draw("s", 40, "nominal", k, 7, cf.PATH_BRANCHES, prior) for k in range(2000)]
        assert all(abs(draws.count(b) / len(draws) - p) < 0.04 for b, p in zip(cf.PATH_BRANCHES, prior))


def test_retimed_hold_equals_the_log_where_the_lateral_steps_agree():
    route = cf.cl.Route(np.array([[0.0, 0.0], [300.0, 0.0]]))
    rng = np.random.default_rng(0)
    T = 80
    x = 10.0 + 0.8 * np.arange(T) + rng.normal(0, 0.05, T)                    # noisy logged steps
    y = 3.0 - np.clip(0.06 * (np.arange(T) - 40), 0, None) + rng.normal(0, 0.05, T)   # cuts in from frame 40
    P, M = np.stack([x, y], 1), np.ones(T, bool)
    _, _, d_log, s_log = cf.lateral_frame(route, P, M)
    same = cf.retimed_hold(route, P, M, s_log, d_log, d_log.copy(), 20)
    assert np.allclose(same, P)                                               # never leaves the log: unchanged
    d_b = d_log.copy()
    d_b[45:] = d_log[45]                                                      # holds from frame 45
    held = cf.retimed_hold(route, P, M, s_log, d_log, d_b, 20)
    assert np.allclose(held[:46], P[:46])
    step = np.linalg.norm(np.diff(held, axis=0), axis=1)
    assert (np.diff(held[:, 0]) > 0).all() and step.max() < 1.2               # no reversal, no jumps


def test_lateral_frame_does_not_jump_to_another_route_segment():
    route = cf.cl.Route(np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 6.0], [0.0, 6.0]]))   # doubles back 6 m away
    T = 60
    P, M = np.stack([20.0 + 1.0 * np.arange(T), np.full(T, 2.8)], 1), np.ones(T, bool)
    _, _, d, _ = cf.lateral_frame(route, P, M)
    assert np.abs(np.diff(d)).max() < 0.2                                     # stays on the first segment
