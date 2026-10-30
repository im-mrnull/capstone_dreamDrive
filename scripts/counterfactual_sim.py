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
    python scripts/counterfactual_sim.py tune         # matched-planner G2: proposal planner, draws 5-9
    python scripts/counterfactual_sim.py matched      # matched-planner G2: public cells, draws 10-19
    python scripts/counterfactual_sim.py confirm-tune # confirmation: safety margin, dev draws 5-9
    python scripts/counterfactual_sim.py confirm      # confirmation: public cells, 73 primary scenes
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


# --------------------------------------------------------------------------
# Matched-planner G2 (docs/wiki/headroom-test.md section 8)
# --------------------------------------------------------------------------
MATCHED_TUNE_JSON = rs.BENCH / "headroom_matched_tune.json"
MATCHED_JSON = rs.BENCH / "headroom_matched.json"
TUNE_DRAWS = (5, 6, 7, 8, 9)
EVAL_DRAWS = tuple(range(10, 20))
LATENCY_DRAWS = (10, 11)
MATCHED = ("Branch oracle", "Ideal single (mean)", "Ideal single (most likely)", "Ideal set")
TUNE_ON = ("Branch oracle", "Ideal single (mean)", "Ideal set")    # neither predictor type favoured
G2M = {"score": 0.01, "at_fault": 0.01, "alpha": 0.05, "replicates": 2000}
LATENCY_BUDGET_MS = {"median": 50.0, "p95": 100.0}
PLAN_V2 = cl.plan


def matched_dose() -> str:
    import json
    dose = json.loads(STAGE1_JSON.read_text(encoding="utf-8"))["dose_for_later_stages"]
    assert dose is not None, "G1 failed at both doses: the test stops"
    return dose


def proposal_grid() -> list[dict]:
    """12 settings of the proposal planner: competing or fallback-only constant proposals x
    clearance weight x progress weight; fine accelerations, jerk 0.2 and IDM targets fixed."""
    from itertools import product
    from closed_loop_v3 import FINE_ACCELS
    base, _ = rs._v2_cfg()
    return [{**base, "accels": FINE_ACCELS, "clear_m": 1.0, "jerk_w": 0.2, "lat_w": 0.0,
             "idm_fracs": (0.2, 0.4, 0.6, 0.8, 1.0), "const": c, "clear_w": cw, "prog_w": pw}
            for c, cw, pw in product(("always", "fallback"), (0.0, 2.0), (0.0, 2.0, 5.0))]


def use_planner(planner: str, cfg: dict) -> None:
    """Install planner "v2" or "proposal" in this process (pool workers are reused)."""
    import proposal_planner as pp
    if planner == "proposal":
        pp.use(dict(cfg))
    else:
        cl.plan = PLAN_V2


def _matched_task(job) -> tuple:
    """Pool task: (planner, run_package_cf args)."""
    planner, args = job
    use_planner(planner, args[4])
    return run_package_cf(args)


def per_scene(eps: list) -> dict:
    """{scene: [episodes, score sum, at-fault sum]}: sufficient for the paired scene bootstrap."""
    out = {}
    for e in eps:
        r = out.setdefault(e["scene"], [0, 0.0, 0])
        r[0] += 1
        r[1] += e["score"]
        r[2] += int(e["at_fault"])
    return out


def cells_of(jobs, res) -> dict:
    """{planner: {predictor: summary + per_scene}} pooled over draws."""
    eps = {}
    for (planner, args), r in zip(jobs, res):
        eps.setdefault(planner, {}).setdefault(args[0], []).extend(r[3])
    return {p: {n: {**summary(e), "per_scene": per_scene(e)} for n, e in row.items()} for p, row in eps.items()}


def latency(task_setup, jobs) -> dict:
    """Per-replan planning time (ms), single process: median and p95 per predictor."""
    import time
    times = {}
    for planner, args in jobs:
        task_setup(planner, args[4])
        inner, bucket = cl.plan, times.setdefault(args[0], [])

        def timed(*a, _inner=inner, _bucket=bucket):
            t = time.perf_counter()
            out = _inner(*a)
            _bucket.append(1000.0 * (time.perf_counter() - t))
            return out
        cl.plan = timed
        run_package_cf(args)
        cl.plan = inner
    return {n: {"median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95)), "n_replans": len(v)}
            for n, v in times.items()}


def latency_jobs(planner, scenes, lam, cfg, dose, weighted) -> list:
    return [(planner, (n, 0, scenes, lam, cfg, None, dose, d, weighted))
            for n in ("Ideal set", "Ideal single (mean)") for d in LATENCY_DRAWS]


def within_budget(lat: dict) -> bool:
    return all(v["median_ms"] <= LATENCY_BUDGET_MS["median"] and v["p95_ms"] <= LATENCY_BUDGET_MS["p95"]
               for v in lat.values())


def preregistered() -> dict:
    ref = {"rule_doc": RULE_DOC, "rule_commit": _git("log", "-1", "--format=%H", "--", RULE_DOC),
           "rule_clean": _git("status", "--porcelain", "--", RULE_DOC, "scripts/counterfactual_sim.py",
                              "scripts/proposal_planner.py") == ""}
    assert ref["rule_commit"] and ref["rule_clean"], "commit the pre-registration first"
    return ref


def tune_rows(grid, res) -> list:
    """Per setting: TUNE_ON summaries and their mean score (res in grid x TUNE_ON x draws order)."""
    rows, k = [], len(TUNE_ON) * len(TUNE_DRAWS)
    for i, cfg in enumerate(grid):
        part = res[i * k:(i + 1) * k]
        st = {n: summary([e for r in part if r[0] == n for e in r[3]]) for n in TUNE_ON}
        rows.append({"cfg": cfg, "mean_score": float(np.mean([v["score"] for v in st.values()])), "packages": st})
    return rows


def tune(workers: int) -> None:
    """Pick the proposal planner's setting on draws 5-9: highest mean score over TUNE_ON."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    ref, dose = preregistered(), matched_dose()
    _, lam = rs._v2_cfg()
    scenes, grid = rs._dev_scenes(), proposal_grid()
    jobs = [("proposal", (n, 0, scenes, lam, cfg, None, dose, d, False))
            for cfg in grid for n in TUNE_ON for d in TUNE_DRAWS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_matched_task, jobs))
    rows = tune_rows(grid, res)
    best = max(range(len(rows)), key=lambda i: rows[i]["mean_score"])
    payload = {**ref, "dose": dose, "draws": list(TUNE_DRAWS), "tune_on": list(TUNE_ON),
               "proposal": {"grid": rows, "chosen": rows[best]["cfg"]},
               "when": datetime.now().isoformat(timespec="seconds")}
    MATCHED_TUNE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for r in rows:
        c = r["cfg"]
        print(f"const {c['const']:8s} clear {c['clear_w']:.0f} prog {c['prog_w']:.0f}: mean {r['mean_score']:.3f}  "
              + "  ".join(f"{n}: {v['score']:.3f}/{v['at_fault_rate']:.3f}" for n, v in r["packages"].items()))
    print("chosen:", {k: rows[best]["cfg"][k] for k in ("const", "clear_w", "prog_w")}, "| saved:", MATCHED_TUNE_JSON)


def matched(workers: int) -> None:
    """The public cells (planner v2, proposal planner) x MATCHED predictors on draws 10-19."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    ref, dose = preregistered(), matched_dose()
    tuned = json.loads(MATCHED_TUNE_JSON.read_text(encoding="utf-8"))
    assert tuned["rule_commit"] == ref["rule_commit"], "tuning ran under a different rule commit"
    v2, lam = rs._v2_cfg()
    cfgs = {"v2": v2, "proposal": tuned["proposal"]["chosen"]}
    scenes = rs._dev_scenes()
    jobs = [(p, (n, 0, scenes, lam, cfgs[p], None, dose, d, False)) for p in cfgs for n in MATCHED for d in EVAL_DRAWS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_matched_task, jobs))
    cells = cells_of(jobs, res)
    lat = {p: latency(use_planner, latency_jobs(p, scenes, lam, cfgs[p], dose, False)) for p in cfgs}
    payload = {**ref, "dose": dose, "draws": list(EVAL_DRAWS), "planner_cfgs": cfgs, "band": lam,
               "cells": cells, "latency": lat, "latency_budget_ms": LATENCY_BUDGET_MS,
               "within_budget": {p: within_budget(v) for p, v in lat.items()},
               "when": datetime.now().isoformat(timespec="seconds")}
    MATCHED_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_cells(cells, lat)
    print("saved:", MATCHED_JSON)


def print_cells(cells: dict, lat: dict) -> None:
    for p, row in cells.items():
        for n, v in row.items():
            print(f"{p:12s} {n:28s} score {v['score']:.3f}  at-fault {v['at_fault_rate']:.4f}  vru "
                  f"{v['at_fault_vru_rate']:.4f}  near {v['near_miss_rate']:.3f}  harsh {v['harsh_braking_rate']:.3f}  "
                  f"progress {v['progress']:.3f}  jerk {v['jerk_rms']:.2f}")
        for n, v in lat.get(p, {}).items():
            print(f"{p:12s} latency {n:20s} median {v['median_ms']:.1f} ms  p95 {v['p95_ms']:.1f} ms")


def matched_gate(cells: dict, eligible: list, seed: int = 0) -> dict:
    """G2-matched: some eligible planner's Ideal set beats the best single cell over all
    eligible planners (best score / lowest at-fault; the oracle is not a single predictor here)
    by the threshold, with a Holm-adjusted one-sided paired scene-bootstrap p < alpha over the
    2 x len(eligible) tests (draws pooled per scene)."""
    singles = [(p, n) for p in eligible for n in ("Ideal single (mean)", "Ideal single (most likely)")]
    best = {"score": max(singles, key=lambda c: cells[c[0]][c[1]]["score"]),
            "at_fault": min(singles, key=lambda c: cells[c[0]][c[1]]["at_fault_rate"])}
    scenes = sorted(cells[eligible[0]]["Ideal set"]["per_scene"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(scenes), (G2M["replicates"], len(scenes)))
    counts = np.stack([np.bincount(r, minlength=len(scenes)) for r in idx]).astype(float)   # (B, scenes)

    def sums(p, n, col):
        ps = cells[p][n]["per_scene"]
        return np.array([ps[s][col] for s in scenes], float), np.array([ps[s][0] for s in scenes], float)
    tests = []
    for p in eligible:
        for metric, col, sign in (("score", 1, 1.0), ("at_fault", 2, -1.0)):
            bp, bn = best[metric]
            xs, xn = sums(p, "Ideal set", col)
            ys, yn = sums(bp, bn, col)
            gain = sign * (xs.sum() / xn.sum() - ys.sum() / yn.sum())                   # > 0: the set is better
            reps = sign * ((counts @ xs) / (counts @ xn) - (counts @ ys) / (counts @ yn))
            tests.append({"planner": p, "metric": metric, "vs": [bp, bn], "gain": float(gain),
                          "p": float((np.sum(reps <= 0) + 1) / (len(reps) + 1)),
                          "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]})
    run, m = 0.0, len(tests)
    for r, i in enumerate(np.argsort([t["p"] for t in tests])):
        run = max(run, min(1.0, (m - r) * tests[i]["p"]))
        tests[i]["p_holm"] = run
    for t in tests:
        t["passes"] = bool(t["gain"] >= G2M[t["metric"]] and t["p_holm"] < G2M["alpha"])
    return {"best_single": {k: list(v) for k, v in best.items()}, "eligible": list(eligible), "tests": tests,
            "passes": any(t["passes"] for t in tests)}


# --------------------------------------------------------------------------
# Confirmation on the primary scenes (docs/wiki/headroom-test.md section 10)
# --------------------------------------------------------------------------
CONFIRM_TUNE_JSON = rs.BENCH / "headroom_confirm_tune.json"
CONFIRM_JSON = rs.BENCH / "headroom_confirm.json"
CONFIRM_DRAWS = tuple(range(10))
CONFIRM_LATENCY_DRAWS = (0, 1)
MARGINS = (0.0, 0.25, 0.5, 0.75, 1.0)
# cells: "planner | predictor"; the contingency cells come from the confidential runner
SINGLE_CELLS = ("proposal | Ideal single (mean)", "proposal | Ideal single (most likely)",
                "proposal+margin | Ideal single (mean)", "contingency | Ideal single (mean)")
SET_CELLS = ("proposal | Ideal set", "contingency | Ideal set")
PAIRS = {"H2": ("contingency | Ideal set", "contingency | Ideal single (mean)"),
         "H3": ("contingency | Ideal set", "proposal | Ideal set"),
         "H4": ("proposal | Ideal set", "proposal+margin | Ideal single (mean)")}


def primary_scenes() -> list:
    import closed_loop_cv as cv
    return sorted(cv.primary_scenes(cv.fold_of_scene()))


def frozen_proposal() -> dict:
    import json
    return json.loads(MATCHED_TUNE_JSON.read_text(encoding="utf-8"))["proposal"]["chosen"]


def confirm_tune(workers: int) -> None:
    """The caution control: the proposal planner's safety margin for the Ideal single (mean),
    best score on dev draws 5-9."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    ref, dose = preregistered(), matched_dose()
    _, lam = rs._v2_cfg()
    base, scenes = frozen_proposal(), rs._dev_scenes()
    jobs = [("proposal", ("Ideal single (mean)", 0, scenes, lam, {**base, "margin_m": m}, None, dose, d, False))
            for m in MARGINS for d in TUNE_DRAWS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_matched_task, jobs))
    rows = []
    for m in MARGINS:
        st = summary([e for (_, a), r in zip(jobs, res) if a[4]["margin_m"] == m for e in r[3]])
        rows.append({"margin_m": m, "summary": st})
        print(f"margin {m:.2f} m: score {st['score']:.3f}  at-fault {st['at_fault_rate']:.4f}  "
              f"harsh {st['harsh_braking_rate']:.3f}  near {st['near_miss_rate']:.3f}  progress {st['progress']:.3f}")
    best = max(rows, key=lambda r: r["summary"]["score"])["margin_m"]
    payload = {**ref, "dose": dose, "draws": list(TUNE_DRAWS), "rows": rows, "chosen_margin_m": best,
               "when": datetime.now().isoformat(timespec="seconds")}
    CONFIRM_TUNE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("chosen margin:", best, "| saved:", CONFIRM_TUNE_JSON)


def confirm(workers: int) -> None:
    """The public cells on the 73 primary scenes, draws 0-9."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    ref, dose = preregistered(), matched_dose()
    tuned = json.loads(CONFIRM_TUNE_JSON.read_text(encoding="utf-8"))
    assert tuned["rule_commit"] == ref["rule_commit"], "tuning ran under a different rule commit"
    _, lam = rs._v2_cfg()
    base, scenes = frozen_proposal(), primary_scenes()
    cfgs = {"proposal": base, "proposal+margin": {**base, "margin_m": tuned["chosen_margin_m"]}}
    preds = {"proposal": ("Branch oracle", "Ideal single (mean)", "Ideal single (most likely)", "Ideal set"),
             "proposal+margin": ("Ideal single (mean)",)}
    jobs = [(p, (n, 0, scenes, lam, cfgs[p], None, dose, d, False)) for p in cfgs for n in preds[p] for d in CONFIRM_DRAWS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_matched_task, [("proposal", a) for _, a in jobs]))
    cells = {f"{p} | {n}": v for p, row in cells_of(jobs, res).items() for n, v in row.items()}
    lat = {p: latency(use_planner, [("proposal", (n, 0, scenes, lam, cfgs[p], None, dose, d, False))
                                    for n in ("Ideal set", "Ideal single (mean)") for d in CONFIRM_LATENCY_DRAWS])
           for p in cfgs}
    payload = {**ref, "dose": dose, "scenes": "73 primary scenes", "draws": list(CONFIRM_DRAWS), "planner_cfgs": cfgs,
               "band": lam, "cells": cells, "latency": lat, "latency_budget_ms": LATENCY_BUDGET_MS,
               "within_budget": {p: within_budget(v) for p, v in lat.items()},
               "when": datetime.now().isoformat(timespec="seconds")}
    CONFIRM_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for n, v in cells.items():
        print(f"{n:40s} score {v['score']:.4f}  at-fault {v['at_fault_rate']:.4f}  vru {v['at_fault_vru_rate']:.4f}  "
              f"near {v['near_miss_rate']:.3f}  harsh {v['harsh_braking_rate']:.3f}  progress {v['progress']:.3f}")
    for p, row in lat.items():
        for n, v in row.items():
            print(f"{p:16s} latency {n:20s} median {v['median_ms']:.1f} ms  p95 {v['p95_ms']:.1f} ms")
    print("saved:", CONFIRM_JSON)


def paired_boot(x: dict, y: dict, metric: str, scenes: list, counts: np.ndarray) -> dict:
    """One-sided paired scene bootstrap of x better than y (score up / at-fault down)."""
    col, sign = {"score": (1, 1.0), "at_fault": (2, -1.0)}[metric]
    xs, xn = (np.array([x["per_scene"][s][c] for s in scenes], float) for c in (col, 0))
    ys, yn = (np.array([y["per_scene"][s][c] for s in scenes], float) for c in (col, 0))
    reps = sign * ((counts @ xs) / (counts @ xn) - (counts @ ys) / (counts @ yn))
    return {"gain": float(sign * (xs.sum() / xn.sum() - ys.sum() / yn.sum())),
            "p": float((np.sum(reps <= 0) + 1) / (len(reps) + 1)),
            "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]}


def confirm_tests(cells: dict, eligible_planners: set, seed: int = 0) -> dict:
    """H1 (best set cell vs best single cell, per metric) and H2-H4, 2 metrics each, Holm over
    all; cells of planners outside the latency budget cannot be compared. H1 reading rule:
    meaningful (threshold and Holm p < alpha) / small but real (Holm p < alpha) / none."""
    ok = lambda c: c in cells and c.split(" | ")[0] in eligible_planners     # noqa: E731
    singles, sets = [c for c in SINGLE_CELLS if ok(c)], [c for c in SET_CELLS if ok(c)]
    scenes = sorted(cells[singles[0]]["per_scene"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(scenes), (G2M["replicates"], len(scenes)))
    counts = np.stack([np.bincount(r, minlength=len(scenes)) for r in idx]).astype(float)
    key = {"score": lambda c: cells[c]["score"], "at_fault": lambda c: -cells[c]["at_fault_rate"]}
    tests = []
    for m in ("score", "at_fault"):
        x, y = max(sets, key=key[m]), max(singles, key=key[m])
        tests.append({"test": "H1", "metric": m, "x": x, "y": y, **paired_boot(cells[x], cells[y], m, scenes, counts)})
    for h, (x, y) in PAIRS.items():
        if ok(x) and ok(y):
            for m in ("score", "at_fault"):
                tests.append({"test": h, "metric": m, "x": x, "y": y, **paired_boot(cells[x], cells[y], m, scenes, counts)})
    run, n = 0.0, len(tests)
    for r, i in enumerate(np.argsort([t["p"] for t in tests])):
        run = max(run, min(1.0, (n - r) * tests[i]["p"]))
        tests[i]["p_holm"] = run
    for t in tests:
        t["significant"] = bool(t["p_holm"] < G2M["alpha"])
        t["meets_threshold"] = bool(t["gain"] >= G2M[t["metric"]])
    h1 = [t for t in tests if t["test"] == "H1"]
    reading = ("meaningful benefit" if any(t["significant"] and t["meets_threshold"] for t in h1) else
               "small but real benefit" if any(t["significant"] for t in h1) else "no demonstrated benefit")
    return {"eligible_planners": sorted(eligible_planners), "tests": tests, "h1_reading": reading}


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
    ap.add_argument("cmd", choices=("check", "eligibility", "stage1", "tune", "matched", "confirm-tune", "confirm"))
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    if args.cmd == "check":
        check()
    elif args.cmd == "eligibility":
        eligibility()
    elif args.cmd == "tune":
        tune(args.workers)
    elif args.cmd == "matched":
        matched(args.workers)
    elif args.cmd == "confirm-tune":
        confirm_tune(args.workers)
    elif args.cmd == "confirm":
        confirm(args.workers)
    else:
        stage1(args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
