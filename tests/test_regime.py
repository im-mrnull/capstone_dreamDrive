"""The regime map's world and execution (scripts/counterfactual_sim.py, docs/wiki/headroom-test.md
section 14): recognition delay, anchoring, intent leakage (gate B: identical delayed histories,
different current positions and timing rates), exact commitment and hold-aware proposals."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

import counterfactual_sim as cf
import proposal_planner as pp

cl = cf.cl
DT = cl.DT
T, T_DIV, X0, V = 120, 10, 10.0, 1.0        # frames, branch start, logged path x = X0 + V * DT * tau (slow: the
                                            # branches stay within 0.3 m of each other for a while)


def ramp(rate: float, start: int = T_DIV) -> np.ndarray:
    """A timing schedule: log time = frame up to start, then rate log frames per frame."""
    g = np.arange(T, dtype=float)
    return np.where(g <= start, g, start + rate * (g - start))


def world(real: np.ndarray, delay: int = 0, anchor: bool = True) -> cf.CFWorld:
    """One target (agent 0) on a straight constant-speed logged path, realised branch "hesitate"
    with schedule real; the other branches: logged (rate 1) and assertive (rate 1.5)."""
    w = object.__new__(cf.CFWorld)
    P = np.zeros((T, 1, 2))
    P[:, 0, 0] = X0 + V * DT * np.arange(T)
    M = np.ones((T, 1), bool)
    w.log = {"pos": P, "mask": M, "heading": np.zeros((T, 1))}
    w.sc = {"T": T, "pos": P.copy(), "mask": M.copy(), "heading": np.zeros((T, 1)), "cls": np.array(["Car"]),
            "world": w}
    w.t0, w.targets, w.t_div, w.branch = 0, [0], {0: T_DIV}, {0: "hesitate"}
    w.tau = {(0, "logged"): ramp(1.0), (0, "hesitate"): real, (0, "assertive"): ramp(1.5)}
    w.delay, w.anchor, w.first_dev = delay, anchor, np.full(1, np.inf)
    return w


def advance(w: cf.CFWorld, t: int) -> None:
    for g in range(1, t + 1):
        w.step(g, None, None, 0.0)


def paired(t: int, delay: int):
    """Worlds A, B, C: identical realised histories up to t - delay; B's timing rate differs from
    A's in (t - delay, t] (so does its current position); C reaches B's position at t by a
    different rate profile; all three continue with A's increments after t."""
    a = ramp(0.5)
    lo = t - delay
    b, c = a.copy(), a.copy()
    b[lo:t + 1] = a[lo] + 1.4 * np.arange(t - lo + 1)
    mid = lo + (t - lo) // 2
    c[lo:mid + 1] = a[lo] + 2.0 * np.arange(mid - lo + 1)
    c[mid:t + 1] = np.linspace(c[mid], b[t], t - mid + 1)
    for s in (b, c):
        s[t:] = s[t] + (a[t:] - a[t])
    return [world(s, delay) for s in (a, b, c)]


def test_anchored_future_is_the_branch_future_for_the_realised_branch():
    w = world(ramp(0.5))
    for t in (0, 15, 40):
        f1 = w.branch_future(0, "hesitate", t)
        w.anchor = False
        assert np.array_equal(f1, w.branch_future(0, "hesitate", t))
        w.anchor = True


def test_delayed_posterior_is_the_undelayed_posterior_earlier():
    for delay in (10, 20):
        wd, w0 = world(ramp(0.5), delay), world(ramp(0.5), 0)
        for t in range(0, 80):
            assert wd.posterior(0, t) == w0.posterior(0, max(0, t - delay))
    assert len(world(ramp(0.5), 0).posterior(0, 40)) == 1 and len(world(ramp(0.5), 20).posterior(0, 25)) == 3


def test_gate_b_identical_delayed_histories_give_identical_weights_and_anchor_shifted_futures():
    t, delay = 30, 20
    A, B, C = paired(t, delay)
    for w in (A, B, C):
        advance(w, t)
    pa, pb = A.posterior(0, t), B.posterior(0, t)
    assert pa == pb and len(pa) > 1                                          # still ambiguous at t
    shift = B.sc["pos"][t, 0] - A.sc["pos"][t, 0]
    assert np.linalg.norm(shift) > 1.0                                       # the current positions differ
    for b in pa:
        fa, fb = A.branch_future(0, b, t), B.branch_future(0, b, t)
        assert np.allclose(fb - fa, shift)                                   # only the anchor shift
        assert np.allclose(np.diff(fa, axis=0), np.diff(fb, axis=0))         # identical increments
    # collision geometry: the world writes each episode's true current state
    for w in (A, B):
        assert np.allclose(w.sc["pos"][t, 0], w.position(0, w.tau[(0, "hesitate")][t])[0])


def test_gate_b_the_realised_rate_never_reaches_the_predictor_or_the_planner():
    """B and C share the delayed history and the current position but reached it at different
    timing rates: the predictor output, and what the planner derives from it, are identical."""
    t, delay = 30, 20
    _, B, C = paired(t, delay)
    for w in (B, C):
        advance(w, t)
    assert np.allclose(B.sc["pos"][t], C.sc["pos"][t])
    assert not np.allclose(B.sc["pos"][t - 5], C.sc["pos"][t - 5])          # different recent motion
    pred = cf.IdealPredictor("Ideal set", weighted=True)
    ob, oc = pred.at(B.sc, t), pred.at(C.sc, t)
    for (ia, sa, ba, wa), (ib, sb, bb, wb) in zip(ob, oc):
        assert ia == ib and np.array_equal(wa, wb) and np.allclose(sa, sb, equal_nan=True)
    route = cl.Route(np.array([[-100.0, 0.0], [400.0, 0.0]]))
    derived = []
    for w, out in ((B, ob), (C, oc)):
        sc = {**w.sc, "pos_now": w.sc["pos"][t], "head_now": w.sc["heading"][t]}
        s0 = float(w.sc["pos"][t, 0, 0]) + 100.0 - 15.0
        S = s0 + np.outer(np.linspace(0.5, 1.5, 4), np.arange(1, cl.H_FRAMES + 1) * DT * 5.0)
        E = pp.ego_bodies(route, S, np.zeros_like(S))
        pens = [pen for _, pen, _ in pp.agent_penetrations(sc, E, out, {"clear_w": 0.0, "margin_m": 0.0})]
        derived.append((pp.lead_tracks(sc, route, s0, out), pens))
    (tb, pb), (tc, pc) = derived
    assert all(np.allclose(x, y, equal_nan=True) for x, y in zip(tb, tc))  # lead positions, speeds, sizes
    assert len(pb) == len(pc) > 0 and all(np.allclose(x, y) for x, y in zip(pb, pc))


def test_constant_proposals_over_a_hold_are_executed_exactly():
    s, d, v = cl.rollout(0.0, 0.0, 8.0, -1.0, 1.5, cl.H_FRAMES)
    for hold in (2, 5, 10):
        hs, hd, hv = cl.rollout(0.0, 0.0, 8.0, -1.0, 1.5, hold)
        assert np.array_equal(hs, s[:hold]) and np.array_equal(hd, d[:hold]) and np.array_equal(hv, v[:hold])


def test_idm_proposals_with_hold_are_evaluated_as_executed():
    goals, targets = np.array([2.0, 6.0, 10.0]), cl.LAT_TARGETS
    for hold in (2, 5, 10):
        cands, S, D, V_ = pp.idm_rollouts(0.0, 0.3, 7.0, goals, targets, None, hold)
        assert S.shape == (len(cands), cl.H_FRAMES)
        for i, (a, dt) in enumerate(cands):
            hs, hd, hv = cl.rollout(0.0, 0.3, 7.0, a, dt, hold)                 # what COMMIT_EXACT executes
            assert np.array_equal(S[i, :hold], hs) and np.array_equal(D[i, :hold], hd) and np.array_equal(V_[i, :hold], hv)
    c0, S0, *_ = pp.idm_rollouts(0.0, 0.3, 7.0, goals, targets, None)
    G, DT_ = np.meshgrid(goals, targets, indexing="ij")
    A0, Sf, _, _ = pp.idm_roll(0.0, 0.3, 7.0, G.ravel(), DT_.ravel(), None)  # the frozen full-horizon IDM
    assert [c[0] for c in c0] == [float(a) for a in A0] and np.array_equal(S0, Sf)


def test_one_frame_execution_drifts_from_the_evaluated_lateral_path():
    """Why exact commitment: one-frame rollouts recompute the lateral rate every frame."""
    hold = 10
    s, d, v = 0.0, 0.0, 8.0
    for _ in range(hold):
        (s,), (d,), (v,) = cl.rollout(s, d, v, 0.0, 1.5, 1)
    planned = cl.rollout(0.0, 0.0, 8.0, 0.0, 1.5, hold)[1][-1]
    assert abs(planned - d) > 0.1


def _probe(regime):
    out = cf.use_regime(regime)
    return cl.REPLAN, cl.COMMIT_EXACT, *out


def test_the_regime_setter_takes_effect_inside_a_spawned_worker():
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        assert ex.submit(_probe, {"delay_s": 2.0, "commit_s": 1.0}).result() == (10, True, 20, True)
        assert ex.submit(_probe, {"delay_s": 1.0, "commit_s": 0.5}).result() == (5, True, 10, True)
        assert ex.submit(_probe, None).result() == (cf.REPLAN_FROZEN, False, 0, False)   # a reused worker resets


def test_the_anchoring_check_switches_parts_off_and_the_map_keeps_all_three():
    assert _probe({"delay_s": 1.0, "commit_s": 0.5, "anchor": False, "exact": False}) == (5, False, 10, False)
    assert _probe({"delay_s": 0.0, "commit_s": 0.2, "exact": True}) == (2, True, 0, True)
    assert all(set(cf.regime_cfg({}, r)["regime"]) == {"delay_s", "commit_s"} for r in cf.REGIMES)
    _probe(None)


def test_regime_tests_weights():
    choice = {cf.regime_key(r): {"single": f"{cf.regime_key(r)} | s", "set": f"{cf.regime_key(r)} | x"} for r in cf.REGIMES}
    w = cf.trend_weights(choice)
    assert abs(sum(w.values())) < 1e-12 and max(w.values()) == 2.0 and min(w.values()) == -2.0
    i = cf.interaction_weights(choice)
    assert sorted(i.values()) == [-1.0] * 4 + [1.0] * 4
    assert i[f"{cf.regime_key(cf.REGIME_CORNER)} | x"] == 1.0
