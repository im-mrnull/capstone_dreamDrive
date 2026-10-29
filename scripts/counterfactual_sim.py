"""Counterfactual intent branches: a closed loop in which the future is genuinely uncertain.

Pre-registered in docs/wiki/headroom-test.md. In the non-reactive and reactive simulators a
perfect predictor gains nothing over constant velocity (dev: oracle 0.901 = CV 0.901;
reactive log oracle - CV = -0.001), so no predictor can show a safety advantage there.
Here the agents whose logged path enters the ego's corridor ahead get a hidden intent:

- TARGETS (from the logs only, before anything is driven): agents present at the episode
  start with a full 2 s history, moving at some point in the episode, whose logged position
  within the episode comes within the ego's corridor (route +- half ego width + half agent
  width + 0.5 m) between 2 m ahead of the ego and its reach (max(human progress, 6 m/s x 5 s)
  + 10 m). At most MAX_TARGETS per episode, earliest corridor entry first.
- each target keeps its LOGGED PATH; only its timing branches, LEAD seconds before its
  logged corridor entry (at the episode start if that is sooner), so the intent is hidden
  when the ego must decide (time warp: log time tau advances at rate r per frame). The
  branch is drawn independently with prior PRIOR from a hash of (scene, start, persona,
  draw seed, agent id), identical for every package:
    logged     r = 1: the log
    hesitate   slows towards a stop (implied deceleration <= b_comf of its class) until it
               is DOSE seconds behind its log, then returns to the logged pace (implied
               acceleration <= a_max), so it ends at least DOSE behind: a yielding /
               hesitating road user, a braking lead vehicle
    assertive  speeds up to RATE x its logged pace (implied acceleration <= a_max, speed
               <= the class cap) until it is DOSE seconds ahead of its log, then keeps the
               logged pace: a road user that goes first
  Beyond the scene's last frame an advanced agent continues at its final logged velocity.
  Branches do not react to the ego; non-target agents replay their logs.
- predictors see only the simulated past; once any target has left its log the models
  predict online from the simulated histories (reactive_sim.OnlinePredictor).
- ideal references (no learning, no bands):
    Branch oracle            the realised branch's future (perfect information)
    Ideal single (most likely)  the most probable branch still consistent with the
                             observed past (Bayesian posterior over the branches; a
                             branch is ruled out once it is CONSISTENT_M from it)
    Ideal single (mean)      the posterior-weighted mean of the consistent branches
                             (the L2-optimal single trajectory)
    Ideal set                every consistent branch with its posterior weight
  Non-target agents get their exact logged future in all of them.

With DOSE = 0 every branch is the log and every episode equals the non-reactive one (`check`).

Usage:
    python scripts/counterfactual_sim.py check        # dose 0 reproduces closed_loop_sim
    python scripts/counterfactual_sim.py eligibility  # targets per dev episode (logs only)
    python scripts/counterfactual_sim.py stage1       # gate G1 (after the rule is committed)
"""

from __future__ import annotations

import argparse
import hashlib

import numpy as np

import closed_loop_sim as cl
import reactive_sim as rs

BRANCHES = ("logged", "hesitate", "assertive")
PRIOR = (0.5, 0.25, 0.25)
DOSES = {"primary": (1.5, 1.3), "strong": (3.0, 1.6)}     # (DOSE s, assertive RATE)
LEAD = 20                           # frames: branches start 2 s before the logged corridor entry
DRAW_SEEDS = (0, 1, 2, 3, 4)
MAX_TARGETS = 3
CORRIDOR_MARGIN = 0.5               # m beyond the half widths
MIN_AHEAD = 2.0                     # m: corridor entries closer than this to the ego don't count
REACH_V = 6.0                       # m/s: ego reach for target selection (with +10 m)
REACH_EXTRA = 10.0
V_CAP = {"ped": 3.0, "two_wheeler": 16.7, "vehicle": 16.7}  # m/s, assertive speed cap
CONSISTENT_M = 0.3                  # m: a branch is ruled out once it is this far from the observed past
                                    # (annotation jitter: median max residual 0.21 m around a smooth 2 s path)
IDEAL = ("Branch oracle", "Ideal single (most likely)", "Ideal single (mean)", "Ideal set")


def targets(sc: dict, vlog: np.ndarray, t0: int, with_entry: bool = False) -> list:
    """Conflict-relevant agents of the episode starting at t0 (logs only); with_entry:
    (agent, first frame in the corridor) pairs."""
    route = sc["route"]
    s0 = sc["s_h"][t0]
    reach = max(sc["s_h"][t0 + cl.EP_LEN] - s0, REACH_V * cl.EP_LEN * cl.DT) + REACH_EXTRA
    found = []
    for a in np.where(sc["mask"][t0])[0]:
        if t0 < 19 or not sc["mask"][t0 - 19:t0 + 1, a].all():
            continue
        frames = [g for g in range(t0, min(t0 + cl.EP_LEN + 1, sc["T"])) if sc["mask"][g, a]]
        if not (vlog[frames, a] > rs.STANDING).any():
            continue
        hw = cl.EGO_RECT[1] / 2 + cl.rect_size(sc["cls"][a])[1] / 2 + CORRIDOR_MARGIN
        for g in frames:
            p = sc["pos"][g, a]
            i = int(np.argmin(np.linalg.norm(route.xy - p, axis=1)))
            if abs(float((p - route.xy[i]) @ route.nrm[i])) <= hw and s0 + MIN_AHEAD <= route.s_grid[i] <= s0 + reach:
                found.append((g, int(a)))
                break
    return [(a, g) if with_entry else a for g, a in sorted(found)[:MAX_TARGETS]]


def branch_draw(sid: str, t0: int, persona: str, draw_seed: int, agent_id: int) -> str:
    """Hidden branch of one target (identical for every package and dose)."""
    h = int(hashlib.sha256(f"{sid}|{t0}|{persona}|{draw_seed}|{agent_id}".encode()).hexdigest()[:16], 16)
    u = np.random.default_rng(h).random()
    return BRANCHES[int(np.searchsorted(np.cumsum(PRIOR), u, side="right"))]


def schedule(branch: str, t0: int, T: int, v_of, par, cap: float, dose: float, rate: float) -> np.ndarray:
    """(T,) log time tau at each frame (tau = g up to the branch start t0). v_of(tau) is the
    logged speed."""
    tau = np.arange(T, dtype=np.float64)
    if branch == "logged" or dose <= 0.0:
        return tau
    a_max, b_comf = par[0], par[1]
    D = dose / cl.DT
    r, phase = 1.0, "change"
    for g in range(t0 + 1, T):
        t = tau[g - 1]
        v = max(v_of(t), 1e-3)
        up, down = a_max * cl.DT / v, b_comf * cl.DT / v      # largest rate steps per frame
        if branch == "hesitate":
            lag = g - 1 - t
            if phase == "change" and lag >= D:
                phase = "resume"
            target = 0.0 if phase == "change" else 1.0
        else:
            ahead = t - (g - 1)
            if phase == "change" and ahead >= D:
                phase = "resume"
            target = min(rate, max(1.0, cap / v)) if phase == "change" else 1.0
        r = min(r + up, target) if target > r else max(r - down, target)
        tau[g] = t + r
    return tau


class CFWorld:
    """One episode: targets follow their branch schedules; everyone else replays the log.
    Writes the simulated positions into the episode's copies of sc["pos"/"mask"/"heading"]."""

    def __init__(self, sc: dict, log: dict, t0: int, persona: str, draw_seed: int, dose: tuple):
        self.sc, self.log, self.t0 = sc, log, t0
        T = sc["T"]
        entry = dict(targets(log["sc"], log["vlog"], t0, with_entry=True))
        self.targets = list(entry)
        self.t_div = {a: max(t0, g - LEAD) for a, g in entry.items()}
        self.branch = {a: branch_draw(sc["sid"], t0, persona, draw_seed, int(sc["agent_id"][a])) for a in self.targets}
        self.tau = {}                                   # (agent, branch) -> (T,) schedule
        for a in self.targets:
            grp = rs.idm_group(str(sc["cls"][a]))
            v_of = lambda t, a=a: log["vlog"][min(int(np.floor(t + 1e-9)), T - 1), a]   # noqa: E731
            for b in BRANCHES:
                self.tau[(a, b)] = schedule(b, self.t_div[a], T, v_of, rs.IDM[grp], V_CAP[grp], *dose)
        self.first_dev = np.full(sc["pos"].shape[1], np.inf)   # read by reactive_sim.OnlinePredictor

    def position(self, a: int, tau: float):
        """Logged position at log time tau (interpolated); beyond the scene's last frame an agent
        present there continues at its final velocity."""
        P, M, T = self.log["pos"], self.log["mask"], self.sc["T"]
        if tau > T - 1:
            if not (M[T - 1, a] and M[T - 3, a]):
                return np.full(2, np.nan), False
            return P[T - 1, a] + (tau - (T - 1)) * (P[T - 1, a] - P[T - 3, a]) / 2.0, True
        f0 = int(np.floor(tau + 1e-9))
        w = tau - f0
        if w < 1e-9 or f0 + 1 >= T:
            return P[f0, a], bool(M[f0, a])
        return (1 - w) * P[f0, a] + w * P[f0 + 1, a], bool(M[f0, a] and M[f0 + 1, a])

    def step(self, g: int, centre, tan, v_ego: float) -> None:
        for a in self.targets:
            t = self.tau[(a, self.branch[a])][g]
            p, m = self.position(a, t)
            self.sc["pos"][g, a], self.sc["mask"][g, a] = (p if m else self.log["pos"][min(g, self.sc["T"] - 1), a]), m
            self.sc["heading"][g, a] = self.log["heading"][min(int(np.rint(t)), self.sc["T"] - 1), a]
            if abs(t - g) > 1e-9:
                self.first_dev[a] = min(self.first_dev[a], g)

    def logged_future(self, a: int, t: int) -> np.ndarray:
        """(60,2) the log continued from the agent's log time at frame t (reactive_sim's log
        oracle; used only by the dose-0 check)."""
        tau = self.tau[(a, self.branch[a])][t] if a in self.branch else float(t)
        out = np.full((60, 2), np.nan)
        for k in range(1, 61):
            if tau + k > self.sc["T"] - 1:
                break
            p, m = self.position(a, tau + k)
            if m:
                out[k - 1] = p
        return out

    def branch_future(self, a: int, b: str, t: int) -> np.ndarray:
        """(60,2) target a's positions at frames t+1..t+60 under branch b (NaN where absent)."""
        out = np.full((60, 2), np.nan)
        tau = self.tau[(a, b)]
        for k in range(1, 61):
            g = t + k
            if g > self.sc["T"] - 1:
                break
            p, m = self.position(a, tau[g])
            if m:
                out[k - 1] = p
        return out

    def posterior(self, a: int, t: int) -> dict:
        """{branch: weight} over the branches consistent with target a's observed past up to t."""
        real = self.tau[(a, self.branch[a])]
        keep = {}
        for b, p in zip(BRANCHES, PRIOR):
            tb = self.tau[(a, b)]
            ok = all(np.linalg.norm(np.nan_to_num(self.position(a, tb[g])[0] - self.position(a, real[g])[0])) < CONSISTENT_M
                     for g in range(self.t0 + 1, t + 1))
            if ok:
                keep[b] = p
        z = sum(keep.values())
        return {b: p / z for b, p in keep.items()}


class IdealPredictor:
    """The ideal references of the module docstring (closed_loop_sim.Predictor interface;
    weighted=True appends per-future weights for the probability-aware planner)."""

    def __init__(self, name: str, weighted: bool = False):
        self.name, self.kind, self.weighted, self.seed = name, "ref", weighted, 0

    def at(self, sc: dict, t: int, lam: float = 1.0):
        w = sc["world"]
        out = []
        for a in np.where(sc["mask"][t])[0]:
            if a in w.branch:
                if self.name == "Branch oracle":
                    futs, wt = [w.branch_future(a, w.branch[a], t)], [1.0]
                else:
                    post = w.posterior(a, t)
                    futs = [w.branch_future(a, b, t) for b in post]
                    wt = list(post.values())
                    if self.name == "Ideal single (most likely)":
                        i = int(np.argmax(wt))
                        futs, wt = [futs[i]], [1.0]
                    elif self.name == "Ideal single (mean)":
                        F, W = np.stack(futs), np.array(wt)[:, None, None]
                        ok = np.isfinite(F)
                        num = np.where(ok, F * W, 0.0).sum(0)
                        den = np.where(ok, W, 0.0).sum(0)
                        futs, wt = [np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)], [1.0]
                samples = np.stack(futs)
            else:
                fut = sc["pos"][t + 1:t + 61, a]
                m = sc["mask"][t + 1:t + 61, a]
                f = np.full((60, 2), np.nan)
                f[:len(fut)][m] = fut[m]
                samples, wt = f[None], [1.0]
            item = (a, samples, np.zeros(60))
            out.append(item + ((np.asarray(wt, dtype=np.float64),) if self.weighted else ()))
        return out


def eligible_episodes(scene_ids) -> list:
    """(scene, start) pairs with at least one target (logs only)."""
    out = []
    for sid in scene_ids:
        log = rs.log_of(sid)
        for t0 in cl.EP_STARTS:
            if cl.episode_ok(log["sc"], t0) and targets(log["sc"], log["vlog"], t0):
                out.append((sid, t0))
    return out


def run_package_cf(args) -> tuple:
    """args = (package, seed, scene ids, band scale, planner cfg, {scene: fold} or None,
    dose name, draw seed, weighted). Eligible episodes only (at least one target)."""
    name, seed, scene_ids, lam, cfg, folds, dose, draw_seed, weighted = args
    cfg = dict(cfg)
    cl.set_instrument(cfg.pop("instrument", {}))
    cl.PLAN_CFG.update({**cl.V3_DEFAULTS, **cfg})
    dose_v = (0.0, 1.0) if dose == "none" else DOSES[dose]
    predictors, eps = {}, []
    for sid, t0 in eligible_episodes(scene_ids):
        fold = None if folds is None else folds[sid]
        if fold not in predictors:
            predictors[fold] = IdealPredictor(name, weighted) if name in IDEAL else \
                rs.OnlinePredictor(cl.Predictor(name, seed, fold), fold, weighted)
        log = rs.log_of(sid)
        for persona, vmin in cl.PERSONAS.items():
            sc = rs.episode_scene(log)
            world = CFWorld(sc, log, t0, persona, draw_seed, dose_v)
            sc["world"], sc["react"] = world, world.step
            r = cl.run_episode(sc, t0, vmin, predictors[fold], lam)
            tg = world.targets
            r.update({"persona": persona, "fold": fold, "dose": dose, "draw_seed": draw_seed,
                      "n_targets": len(tg), "branches": [world.branch[a] for a in tg],
                      "n_deviated": int(np.isfinite(world.first_dev).sum())})
            eps.append(r)
    return name, seed, lam, eps, 0


STAGE1_JSON = rs.BENCH / "headroom_stage1.json"
RULE_DOC = "docs/wiki/headroom-test.md"
STAGE1_PACKAGES = ("Human driver (log replay)", "No prediction (ignore agents)", "CV (vel_k=5)",
                   "Branch oracle", "Ideal single (most likely)", "Ideal single (mean)", "Ideal set")
G1 = {"score": 0.02, "at_fault": 0.02}          # branch oracle over CV (docs/wiki/headroom-test.md s5)


def _git(*args) -> str:
    import subprocess
    return subprocess.run(["git", *args], cwd=rs.REPO, capture_output=True, text=True).stdout.strip()


def summary(eps: list) -> dict:
    s = cl.summarise(eps)
    tg = [e for e in eps if e["n_targets"]]
    s.update({"near_miss_moving_rate": float(np.mean([e["near_miss_moving"] for e in eps])),
              "jerk_rms": float(np.mean([e["jerk_rms"] for e in eps])),
              "branched_episode_share": float(np.mean([any(b != "logged" for b in e["branches"]) for e in tg]))})
    return s


def stage1(workers: int) -> None:
    """Perfect-information headroom on the dev scenes (planner v2), gate G1 per dose."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    ref = {"rule_doc": RULE_DOC, "rule_commit": _git("log", "-1", "--format=%H", "--", RULE_DOC),
           "rule_doc_clean": _git("status", "--porcelain", "--", RULE_DOC, "scripts/counterfactual_sim.py") == ""}
    assert ref["rule_commit"] and ref["rule_doc_clean"], "commit the pre-registration first"
    cfg, lam = rs._v2_cfg()
    scenes = rs._dev_scenes()
    jobs = [(n, 0, scenes, lam, cfg, None, d, s, False) for d in DOSES for s in DRAW_SEEDS for n in STAGE1_PACKAGES]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(run_package_cf, jobs))
    eps = {}
    for j, r in zip(jobs, res):
        eps.setdefault((j[6], j[0]), []).extend(r[3])
    out = {d: {n: summary(eps[(d, n)]) for n in STAGE1_PACKAGES} for d in DOSES}
    gate = {}
    for d in DOSES:
        o, c = out[d]["Branch oracle"], out[d]["CV (vel_k=5)"]
        ds, da = o["score"] - c["score"], c["at_fault_rate"] - o["at_fault_rate"]
        gate[d] = {"score_gain": ds, "at_fault_reduction": da,
                   "passes": bool(ds >= G1["score"] or da >= G1["at_fault"])}
    chosen = "primary" if gate["primary"]["passes"] else "strong" if gate["strong"]["passes"] else None
    payload = {**ref, "scenes": "32 development scenes (frozen val + test)", "draw_seeds": list(DRAW_SEEDS),
               "doses": DOSES, "prior": dict(zip(BRANCHES, PRIOR)), "planner_v2_cfg": cfg, "band": lam,
               "packages": out, "g1": gate, "dose_for_later_stages": chosen,
               "when": datetime.now().isoformat(timespec="seconds")}
    STAGE1_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for d in DOSES:
        for n in sorted(out[d], key=lambda n: -out[d][n]["score"]):
            v = out[d][n]
            print(f"{d:8s} {n:32s} score {v['score']:.3f}  at-fault {v['at_fault_rate']:.3f}  vru "
                  f"{v['at_fault_vru_rate']:.3f}  near {v['near_miss_rate']:.2f}  harsh {v['harsh_braking_rate']:.2f}  "
                  f"progress {v['progress']:.3f}")
        g = gate[d]
        print(f"G1 {d}: score {g['score_gain']:+.3f}  at-fault {g['at_fault_reduction']:+.3f}  "
              f"{'PASS' if g['passes'] else 'fail'}")
    print("dose for later stages:", chosen, "| saved:", STAGE1_JSON)


def check(n_scenes: int = 6) -> None:
    """Dose 0 must reproduce the non-reactive episodes field for field (eligible episodes)."""
    cfg, lam = rs._v2_cfg()
    scenes = rs._dev_scenes()[:n_scenes]
    fields = ("scene", "t_start", "persona", "score", "at_fault", "progress", "harsh_braking", "near_miss",
              "min_clearance_m")
    elig = set(eligible_episodes(scenes))
    for n in ["CV (vel_k=5)", "ST-GAT (MHA)", "K-mode set (10 modes)", "AgentFormer set (10 samples)",
              "Oracle (true future, no band)", "Human driver (log replay)"]:
        a = [e for e in cl.run_package((n, 0, scenes, lam, cfg, None))[3] if (e["scene"], e["t_start"]) in elig]
        b = run_package_cf((n, 0, scenes, lam, cfg, None, "none", 0, False))[3]
        same = [tuple(x[f] for f in fields) == tuple(y[f] for f in fields) for x, y in zip(a, b)]
        print(f"{n:32s} {sum(same)}/{len(same)} identical", flush=True)
        assert all(same) and len(a) == len(b)
    a = [e for e in cl.run_package(("Oracle (true future, no band)", 0, scenes, lam, cfg, None))[3]
         if (e["scene"], e["t_start"]) in elig]
    for n in IDEAL:
        b = run_package_cf((n, 0, scenes, lam, cfg, None, "none", 0, False))[3]
        same = [tuple(x[f] for f in fields) == tuple(y[f] for f in fields) for x, y in zip(a, b)]
        print(f"{n:32s} {sum(same)}/{len(same)} identical to the oracle", flush=True)
        assert all(same) and len(a) == len(b)


def eligibility() -> None:
    scenes = rs._dev_scenes()
    n_ok = sum(cl.episode_ok(rs.log_of(s)["sc"], t0) for s in scenes for t0 in cl.EP_STARTS)
    el = eligible_episodes(scenes)
    cls, n_t = {}, []
    for sid, t0 in el:
        log = rs.log_of(sid)
        tg = targets(log["sc"], log["vlog"], t0)
        n_t.append(len(tg))
        for a in tg:
            c = str(log["sc"]["cls"][a])
            cls[c] = cls.get(c, 0) + 1
    print(f"dev: {len(el)}/{n_ok} episodes eligible, {np.mean(n_t):.2f} targets each; "
          f"episodes per package per dose = {len(el)} x {len(cl.PERSONAS)} personas x {len(DRAW_SEEDS)} draws "
          f"= {len(el) * len(cl.PERSONAS) * len(DRAW_SEEDS)}")
    print("targets by class:", dict(sorted(cls.items(), key=lambda x: -x[1])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "eligibility", "stage1"))
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    if args.cmd == "check":
        check()
    elif args.cmd == "eligibility":
        eligibility()
    else:
        stage1(args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
