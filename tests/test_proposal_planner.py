"""Common proposal planner (scripts/proposal_planner.py) on synthetic straight-road scenes."""

import numpy as np

from conftest import load_script

pp = load_script("proposal_planner.py")
cl = pp.cl

ROUTE = cl.Route(np.array([[0.0, 0.0], [600.0, 0.0]]))
WALLS = [(x, y) for x in range(-5, 160, 5) for y in (-2.6, 2.6)]     # parked cars: no room to swerve
FINE = (-6.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
V2 = {"hard_s": 1.0, "soft_w": 5.0, "fallback": "least_penetration"}
V2_PLAN = cl.plan                       # closed_loop_sim.plan (pp.use replaces it)
assert V2_PLAN.__module__ == "closed_loop_sim"


def lead_x(t, x0=20.0, v=8.0, dec=3.0, t_brake=1.0):
    """A lead car at constant speed until t_brake, then braking to a stop."""
    tb = np.minimum(t, t_brake)
    tt = np.clip(t - t_brake, 0.0, v / dec)
    return x0 + v * tb + v * tt - 0.5 * dec * tt ** 2


def scene(t, a_prev=0.0, d_prev=0.0):
    """Planner inputs at time t: the lead car predicted at constant velocity, parked cars."""
    lx = lead_x(t)
    lv = (lx - lead_x(t - 0.1)) / 0.1
    fut = np.stack([lx + lv * np.arange(1, 31) * cl.DT, np.zeros(30)], 1)[None]
    pos = np.array([[lx, 0.0]] + [list(w) for w in WALLS])
    sc = {"cls": np.array(["Car"] * len(pos)), "pos_now": pos, "head_now": np.tile([1.0, 0.0], (len(pos), 1)),
          "cmd_prev": (a_prev, d_prev)}
    preds = [(0, fut, np.zeros(30))] + [(i + 1, np.repeat(np.array(w)[None, None], 30, 1), np.zeros(30))
                                        for i, w in enumerate(WALLS)]
    return sc, preds


def follow(cfg):
    """Drive 5 s behind the braking lead car; (lowest command, smallest bumper gap)."""
    pp.use({**V2, **cl.V3_DEFAULTS, **pp.PP_DEFAULTS, **cfg})
    s, d, v, a_cmd, dt = 0.0, 0.0, 8.0, 0.0, 0.0
    cmds, gaps = [], []
    for f in range(50):
        if f % cl.REPLAN == 0:
            sc, preds = scene(f * cl.DT, a_cmd, dt)
            a_cmd, dt = pp.plan(sc, ROUTE, s, d, v, 10.0, preds)
            cmds.append(a_cmd)
        (s,), (d,), (v,) = cl.rollout(s, d, v, a_cmd, dt, 1)
        gaps.append(lead_x((f + 1) * cl.DT) - s - pp.EGO_FRONT - 2.15)
    return min(cmds), min(gaps)


def test_defaults_are_closed_loop_sim_plan():
    pp.use({**V2, **cl.V3_DEFAULTS, **pp.PP_DEFAULTS})
    for t in (0.0, 1.4, 2.6):
        sc, preds = scene(t)
        assert pp.plan(sc, ROUTE, 3.0, 0.0, 8.0, 10.0, preds) == V2_PLAN(dict(sc), ROUTE, 3.0, 0.0, 8.0, 10.0, preds)


def test_idm_proposals_follow_a_braking_lead_without_harsh_braking():
    a_v2, gap_v2 = follow({})
    assert a_v2 <= cl.HARSH_DECEL                           # planner v2 brakes late and hard
    a_pp, gap_pp = follow({"accels": FINE, "clear_w": 2.0, "jerk_w": 0.2, "idm_fracs": (0.2, 0.4, 0.6, 0.8, 1.0),
                           "prog_w": 2.0, "const": "fallback"})
    assert a_pp > cl.HARSH_DECEL and gap_pp > 0.0 and gap_pp > gap_v2


def test_frenet_projection_is_exact_on_a_straight_route():
    s, d = pp.frenet(ROUTE, 10.0, np.array([[[12.3, 0.7], [40.05, -1.2], [np.nan, np.nan]]]))
    assert np.allclose(s[0, :2], [12.3, 40.05]) and np.allclose(d[0, :2], [0.7, -1.2])
    assert np.isnan(s[0, 2]) and np.isnan(d[0, 2])


def test_safety_margin_keeps_more_room_from_a_parked_car():
    sc = {"cls": np.array(["Car"]), "pos_now": np.array([[12.0, 2.1]]), "head_now": np.array([[1.0, 0.0]]),
          "cmd_prev": (0.0, 0.0)}
    preds = [(0, np.repeat(np.array([[12.0, 2.1]])[None], 30, 1), np.zeros(30))]
    picks = {}
    for m in (0.0, 0.75):
        pp.use({**V2, **cl.V3_DEFAULTS, **pp.PP_DEFAULTS, "margin_m": m})
        picks[m] = pp.plan(dict(sc), ROUTE, 0.0, 0.0, 6.0, 6.0, preds)
    assert picks[0.0] == (0.0, 0.0) and picks[0.75][1] < 0.0          # moves away from the car
