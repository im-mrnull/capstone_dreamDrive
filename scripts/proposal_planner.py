"""Common proposal planner for the matched-planner headroom test (docs/wiki/headroom-test.md s8).

Planner v2/v3 (closed_loop_sim.plan) samples constant accelerations x lateral targets. This
planner adds PDM-Closed style proposals (Dauner et al., CoRL 2023): an IDM car-follower towards
several target speeds (fractions of the reference speed), following the nearest predicted agent
in the ego's lane, for each lateral target. Every proposal is checked and scored as planner v3
does (same penetration feasibility, soft penalty, clearance and comfort terms) plus a progress
term relative to the proposal that gets furthest; the chosen proposal's first command is
executed and the planner replans every 0.2 s as before.

The improvement applies to single trajectories and sets alike: the IDM leader and every
collision check take the worst case over the predicted futures, as planner v2 does.

With no IDM proposals and no progress term it is closed_loop_sim.plan exactly (`check`).

Usage:
    python scripts/proposal_planner.py check     # reproduces closed_loop_sim.plan, dev episodes
"""

from __future__ import annotations

import argparse

import numpy as np

import closed_loop_sim as cl

# idm_fracs  target speeds of the IDM proposals, as fractions of the reference speed (() = none)
# prog_w     weight of the progress shortfall: 1 - progress / best progress over the proposals
# const      "always": constant-acceleration proposals compete with the IDM ones;
#            "fallback": they are used only when no IDM proposal is feasible (PDM-Closed's
#            emergency manoeuvres)
# margin_m   extra buffer (m) on every predicted footprint (the caution control of
#            docs/wiki/headroom-test.md section 10)
PP_DEFAULTS = {"idm_fracs": (), "prog_w": 0.0, "const": "always", "margin_m": 0.0}
IDM_EGO = (1.5, 2.0, 1.2, 2.0)          # a_max, b_comf (m/s^2), time headway T (s), standstill gap s0 (m)
A_MIN, A_MAX = -6.0, 2.0                # command limits (planner v2's acceleration range)
EGO_FRONT = max(cl.EGO[0]) + cl.EGO[1]  # m ahead of the sensor
LANE_MARGIN = 0.3                       # m added to the half widths for "in the ego's lane"
WINDOW = (-10.0, 80.0)                  # route window around the ego for Frenet projection (m)


def frenet(route, s0: float, P: np.ndarray):
    """Arc length and lateral offset of points P (..., 2) near the ego (nan where P is not finite)."""
    lo, hi = np.searchsorted(route.s_grid, [s0 + WINDOW[0], s0 + WINDOW[1]])
    xy, tan, nrm, sg = route.xy[lo:hi + 1], route.tan[lo:hi + 1], route.nrm[lo:hi + 1], route.s_grid[lo:hi + 1]
    flat = P.reshape(-1, 2)
    ok = np.isfinite(flat[:, 0])
    s, d = np.full(len(flat), np.nan), np.full(len(flat), np.nan)
    if ok.any() and len(sg):
        q = flat[ok]
        i = np.argmin(((q[:, None] - xy[None]) ** 2).sum(-1), axis=1)
        s[ok] = sg[i] + ((q - xy[i]) * tan[i]).sum(-1)               # off the 0.25 m grid
        d[ok] = ((q - xy[i]) * nrm[i]).sum(-1)
    return s.reshape(P.shape[:-1]), d.reshape(P.shape[:-1])


def lead_tracks(sc, route, s0, preds):
    """Per agent and future, its Frenet track for frames 0..H (now + prediction) and its sizes:
    s, d (N, H+1), half length and half width (N,), N = agents x futures."""
    S, D, L, W = [], [], [], []
    ego = route.xy[min(int(round(s0 / 0.25)), len(route.s_grid) - 1)]
    for a_idx, samples, *_ in preds:
        if np.linalg.norm(sc["pos_now"][a_idx] - ego) > WINDOW[1]:
            continue
        full = np.concatenate([np.repeat(sc["pos_now"][a_idx][None, None], samples.shape[0], 0),
                               samples[:, :cl.H_FRAMES]], 1)                 # (K, H+1, 2)
        s, d = frenet(route, s0, full)
        fp = cl.footprint(sc["cls"][a_idx])
        S.append(s), D.append(d)
        L += [max(np.abs(fp[0])) + fp[1]] * len(s)
        W += [fp[1]] * len(s)
    if not S:
        return None
    return np.concatenate(S), np.concatenate(D), np.array(L), np.array(W)


def idm_roll(s0, d0, v0, G, DT_, tracks, k0: int = 0, n: int = cl.H_FRAMES):
    """Ego (s, d, v) for n frames from per-row start states at frame k0 under IDM towards goal
    speeds G with lateral targets DT_, following the nearest in-lane agent ahead (worst case over
    agents and futures); also the first command of each row."""
    a_max, b, T, gap0 = IDM_EGO
    C = len(G)
    s, d, v = np.broadcast_to(s0, C).astype(float), np.broadcast_to(d0, C).astype(float), np.broadcast_to(v0, C).astype(float)
    rate = (DT_ - d) / cl.LAT_TIME
    S, Dd, V, A0 = np.empty((C, n)), np.empty((C, n)), np.empty((C, n)), None
    for k in range(k0, k0 + n):
        free = np.maximum(a_max * (1.0 - (v / np.maximum(G, 0.1)) ** 4), -b)  # slow down comfortably
        acc = free
        if tracks is not None:
            ts, td, tl, tw = tracks
            sa, da = ts[:, k], td[:, k]                                        # (N,)
            vs = (ts[:, k + 1] - sa) / cl.DT
            inlane = np.abs(da[None] - d[:, None]) < cl.EGO[1] + tw[None] + LANE_MARGIN
            gap = sa[None] - s[:, None] - EGO_FRONT - tl[None]
            ahead = inlane & (sa[None] > s[:, None]) & np.isfinite(gap)
            gap = np.where(ahead, gap, np.inf)
            j = np.argmin(gap, axis=1)
            g = gap[np.arange(C), j]
            vl = np.where(np.isfinite(g), np.nan_to_num(vs[j]), v)
            star = gap0 + v * T + v * (v - vl) / (2.0 * np.sqrt(a_max * b))
            follow = np.where(g > 0.1, (np.maximum(star, 0.0) / np.maximum(g, 0.1)) ** 2, np.inf)
            acc = np.where(np.isfinite(g), free - a_max * follow, acc)
        acc = np.clip(np.nan_to_num(acc, nan=A_MIN, neginf=A_MIN), A_MIN, A_MAX)
        if A0 is None:
            A0 = acc.copy()
        nv = np.clip(v + acc * cl.DT, 0.0, cl.V_MAX)
        s = s + 0.5 * (v + nv) * cl.DT
        d = np.where(np.abs(DT_ - d) > np.abs(rate * cl.DT), d + rate * cl.DT, DT_)
        v = nv
        S[:, k - k0], Dd[:, k - k0], V[:, k - k0] = s, d, v
    return A0, S, Dd, V


def idm_rollouts(s0, d0, v0, goals, d_targets, tracks):
    """IDM proposals from the ego's state, one row per (goal, d_target): commands and (s, d, v)."""
    G, DT_ = np.meshgrid(goals, d_targets, indexing="ij")
    G, DT_ = G.ravel(), DT_.ravel()
    A0, S, Dd, V = idm_roll(s0, d0, v0, G, DT_, tracks)
    return [(float(a), float(dt)) for a, dt in zip(A0, DT_)], S, Dd, V


def proposals(sc, route, s0, d0, v0, v_ref, preds, cfg):
    """The candidate commands, their (s, d, v) rollouts and the number of constant ones."""
    cands = [(a, dt) for a in cfg["accels"] for dt in cl.LAT_TARGETS]
    rolls = [cl.rollout(s0, d0, v0, a, dt, cl.H_FRAMES) for a, dt in cands]
    n_const = len(cands)
    if cfg["idm_fracs"]:
        goals = np.array([f * v_ref for f in cfg["idm_fracs"]])
        c2, S, Dd, V = idm_rollouts(s0, d0, v0, goals, cl.LAT_TARGETS, lead_tracks(sc, route, s0, preds))
        cands += c2
        rolls += list(zip(S, Dd, V))
    return cands, rolls, n_const


def ego_bodies(route, S: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Ego circle centres (C, steps, 3, 2) at the planning offsets of trajectories S, D (C, H)."""
    idx = cl.PLAN_OFFSETS - 1
    xy, tan = route.pose(S[:, idx], D[:, idx])
    return cl.circles(xy, tan, cl.EGO)


def agent_penetrations(sc, E, preds, cfg):
    """For each predicted agent near any trajectory: (prediction tuple, penetration (C, K, steps),
    clearance shortfall (C, K) or None). Penetration as closed_loop_sim.plan, with margin_m
    added to every predicted footprint; the shortfall max(0, clear_m - gap) when clear_w > 0."""
    idx = cl.PLAN_OFFSETS - 1
    clear_on = cfg["clear_w"] > 0
    margin = cfg.get("margin_m", 0.0)
    ego_c = E.mean(axis=2)
    for pred in preds:
        a_idx, samples, band = pred[:3]
        fp = cl.footprint(sc["cls"][a_idx])
        P = samples[:, idx]
        ok = np.isfinite(P[..., 0])
        if not ok.any():
            continue
        full = np.concatenate([np.repeat(sc["pos_now"][a_idx][None, None], samples.shape[0], 0), samples], 1)
        back = np.maximum(idx + 1 - 3, 0)
        hd = full[:, idx + 1] - full[:, back]
        n = np.linalg.norm(hd, axis=-1, keepdims=True)
        hd = np.where(n > 0.5, hd / np.maximum(n, 1e-9), sc["head_now"][a_idx])
        reach = cl.EGO[1] + fp[1] + band[idx] + cl.EGO_REACH + max(np.abs(fp[0])) + 0.5 + margin
        if clear_on:
            reach = reach + cfg["clear_m"]
        cd = np.linalg.norm(ego_c[:, None] - np.where(ok[..., None], P, 1e9)[None], axis=-1)
        if not (cd < reach[None, None]).any():
            continue
        hd = np.where(np.isfinite(hd), hd, sc["head_now"][a_idx])
        A = cl.circles(np.where(ok[..., None], P, 1e9), hd, fp)
        dist = np.linalg.norm(E[:, None, :, :, None] - A[None, :, :, None, :, :], axis=-1)   # (C,K,steps,3,m)
        clr = None
        if clear_on:
            gap = dist.min(axis=(3, 4)) - (cl.EGO[1] + fp[1])
            clr = np.clip(cfg["clear_m"] - gap, 0, None).max(axis=2)
        need = cl.EGO[1] + fp[1] + band[idx] + cl.SAFETY_BUFFER + margin
        pen = ((need[None, None, :, None, None] - dist) / need[None, None, :, None, None]).max(axis=(3, 4))
        yield pred, pen, clr


def plan(sc, route, s0, d0, v0, v_ref, preds):
    """closed_loop_sim.plan with IDM proposals and a progress term (same interface)."""
    cfg = {**PP_DEFAULTS, **cl.PLAN_CFG}
    cands, rolls, n_const = proposals(sc, route, s0, d0, v0, v_ref, preds, cfg)
    S = np.stack([r[0] for r in rolls])
    E = ego_bodies(route, S, np.stack([r[1] for r in rolls]))
    v_end, s_end = [r[2][-1] for r in rolls], [r[0][-1] for r in rolls]
    hard = cl.PLAN_OFFSETS * cl.DT <= cfg["hard_s"] + 1e-9
    worst, soft, clear = np.zeros(len(cands)), np.zeros(len(cands)), np.zeros(len(cands))
    for _, pen, clr in agent_penetrations(sc, E, preds, cfg):
        pen = pen.max(axis=1)                                            # worst future: (C, steps)
        if clr is not None:
            clear += clr.max(axis=1) ** 2
        worst = np.maximum(worst, pen[:, hard].max(axis=1))
        if (~hard).any():
            soft = np.maximum(soft, np.clip(pen[:, ~hard], 0, None).max(axis=1))
    feasible = worst <= 0.0
    cost = np.array([((ve - v_ref) / 2.0) ** 2 + 0.05 * a * a + 0.3 * dt * dt + 0.5 * abs(dt - d0)
                     for (a, dt), ve in zip(cands, v_end)]) + cfg["soft_w"] * soft
    cost = cost + cl.v3_cost(cands, sc, clear)
    if cfg["prog_w"] > 0:
        prog = np.array(s_end) - s0
        best = prog[feasible].max() if feasible.any() else prog.max()
        cost = cost + cfg["prog_w"] * np.clip(1.0 - prog / max(best, 1.0), 0, None)
    sc["plan_fallback"] = not feasible.any()
    if cfg["const"] == "fallback" and feasible[n_const:].any():
        feasible[:n_const] = False
    if feasible.any():
        return cands[int(np.argmin(np.where(feasible, cost, np.inf)))]
    if cfg["fallback"] == "brake":
        return cl.ACCELS[0], min(cl.LAT_TARGETS, key=lambda x: abs(x - d0))
    return cands[int(np.lexsort((cost, worst))[0])]


def use(cfg: dict) -> None:
    """Install this planner in this process with cfg (planner v3 keys plus PP_DEFAULTS keys)."""
    cl.plan = plan
    for k in PP_DEFAULTS:
        cl.PLAN_CFG.pop(k, None)
    cl.PLAN_CFG.update(cfg)


def check(n_scenes: int = 6) -> None:
    """With no IDM proposals and no progress term the episodes equal closed_loop_sim.plan's."""
    import reactive_sim as rs
    v2_plan = cl.plan
    cfg, lam = rs._v2_cfg()
    scenes = rs._dev_scenes()[:n_scenes]
    fields = ("scene", "t_start", "persona", "score", "at_fault", "progress", "harsh_braking", "near_miss",
              "min_clearance_m", "jerk_rms")
    for n in ["CV (vel_k=5)", "K-mode set (10 modes)", "Oracle (true future, no band)"]:
        cl.plan = v2_plan
        a = cl.run_package((n, 0, scenes, lam, cfg, None))[3]
        cl.plan = plan
        b = cl.run_package((n, 0, scenes, lam, {**cfg, **PP_DEFAULTS}, None))[3]
        same = [tuple(x[f] for f in fields) == tuple(y[f] for f in fields) for x, y in zip(a, b)]
        print(f"{n:32s} {sum(same)}/{len(same)} identical", flush=True)
        assert all(same) and len(a) == len(b)
    cl.plan = v2_plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check",))
    ap.parse_args()
    check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
