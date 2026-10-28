"""Reactive closed loop: surrounding agents that yield to the ego.

Pre-registered in docs/wiki/reactive-test.md. The non-reactive simulator
(closed_loop_sim.py) replays every agent's log, so a perfect predictor gains nothing
over constant velocity (planner v3 development: oracle 0.901 = CV 0.901 on dev).
Here a share of the agents react to the ego:

- every agent follows its LOGGED PATH. A non-reactive agent replays its log exactly;
- a reactive agent (drawn with probability p_react per episode, identically for every
  package) runs IDM along its path with the ego as its leader whenever the ego's
  footprint (+0.3 m) lies on the path ahead (the next 4 s / 40 m of its log). It
  perceives the ego 0.4 s late. IDM's interaction term is subtracted from the
  acceleration that tracks the logged speed, so an unconstrained agent keeps to its
  log; once delayed it continues its log from where it is (time warp: log time
  tau advances at speed / logged speed), so the lag persists. No lateral evasion,
  no reactions between agents.
- predictions are recomputed online from the simulated histories once any agent has
  left its log (OnlinePredictor); until then the precomputed predictions are used,
  which is exact because the inputs are identical.
- the oracle reference becomes the "log oracle": each agent's logged future continued
  from its current log time (intent, not reactions).

With p_react = 0 every episode equals the non-reactive one (checked: `check`).

Usage:
    python scripts/reactive_sim.py check            # p_react = 0 reproduces closed_loop_sim
    python scripts/reactive_sim.py verify-online    # online models reproduce the saved predictions
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import closed_loop_sim as cl
from conformal_wrapper import cv_predict

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "outputs" / "benchmark_v2"

P_REACT = 0.5                       # primary share of reactive agents (1.0: sensitivity)
SIM_SEEDS = (0, 1, 2)
DELAY = 4                           # frames (0.4 s) between the ego's motion and the agent's reaction
AHEAD_FRAMES = 40                   # logged frames of the agent's path searched for the ego (4 s)
AHEAD_M = 40.0
MARGIN = 0.3                        # m added to the corridor half-width
NEAR_M = 60.0                       # agents farther than this from the ego cannot be constrained
STANDING = 0.3                      # m/s: logged speeds below this count as standing
MIN_INTERACTION = 0.05              # m/s^2: smaller IDM interaction terms are ignored
PED_LIKE = {"Pedestrian", "Child", "Animal"}
TWO_WHEELERS = {"MotorcyleRider", "ScooterRider", "BicycleRider", "Motorcycle", "Scooter", "Bicycle",
                "BicycleGroup"}
# IDM parameters, fixed a priori: a_max, b_comf, b_max (m/s^2), time headway T (s), gap s0 (m)
IDM = {"vehicle": (1.5, 2.0, 6.0, 1.0, 1.0),
       "two_wheeler": (2.0, 2.5, 6.0, 0.8, 0.5),
       "ped": (1.0, 1.5, 3.0, 0.5, 0.5)}


def idm_group(cls: str) -> str:
    return "ped" if cls in PED_LIKE else "two_wheeler" if cls in TWO_WHEELERS else "vehicle"


def reactive_draw(sid: str, t_start: int, persona: str, sim_seed: int, n_agents: int, p: float) -> np.ndarray:
    """Which agents react, from a hash of the episode and sim seed (identical for every
    package; nested in p, so the p = 1 reactive set contains the p = 0.5 one)."""
    h = int(hashlib.sha256(f"{sid}|{t_start}|{persona}|{sim_seed}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(h).random(n_agents) < p


def logged_speed(pos: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(T,A) speed from the displacement over +-2 frames (annotation jitter is ~cm per frame)."""
    T = pos.shape[0]
    v = np.zeros(mask.shape)
    for t in range(T):
        lo, hi = max(0, t - 2), min(T - 1, t + 2)
        ok = mask[lo] & mask[hi] & (hi > lo)
        v[t, ok] = np.linalg.norm(pos[hi, ok] - pos[lo, ok], axis=-1) / ((hi - lo) * cl.DT)
    return v


def seg_dist(p: np.ndarray, centre: np.ndarray, tan: np.ndarray, half_len: float) -> np.ndarray:
    """Distance from points p (n,2) to the segment centre +- half_len * tan."""
    u = np.clip((p - centre) @ tan, -half_len, half_len)
    return np.linalg.norm(p - (centre + u[:, None] * tan[None]), axis=1)


class World:
    """One episode's agents: simulated positions in sc["pos"/"mask"/"heading"] (copies)."""

    def __init__(self, sc: dict, log: dict, t_start: int, react: np.ndarray):
        self.sc, self.log, self.react = sc, log, react
        A = log["pos"].shape[1]
        self.tau = np.full(A, float(t_start))         # log time each agent is at
        self.vlog = log["vlog"]
        self.v = self.vlog[t_start].copy()
        self.first_dev = np.full(A, np.inf)           # first frame an agent is off its log
        self.tau_hist = {t_start: self.tau.copy()}
        self.ego = {}
        grp = [idm_group(str(c)) for c in sc["cls"]]
        self.par = np.array([IDM[g] for g in grp])     # (A,5)
        self.size = np.array([cl.rect_size(c) for c in sc["cls"]])   # (A,2) length, width

    def ego_at(self, k: int):
        """Ego footprint centre, heading and speed at frame k (the human's pose before the episode)."""
        if k in self.ego:
            return self.ego[k]
        k = int(np.clip(k, 0, self.sc["T"] - 1))
        xy, tan = self.sc["route"].pose(np.array(self.sc["s_h"][k]), np.array(0.0))
        return xy + cl.EGO_CENTRE * tan, tan, float(self.sc["v_h"][k])

    def gap_to_ego(self, a: int, t0: float, centre, tan):
        """(gap along the agent's logged path to the ego's footprint, path direction there) or None."""
        P, M, T = self.log["pos"], self.log["mask"], self.sc["T"]
        f = int(np.floor(t0))
        cur = self.position(a, t0)[0]
        if np.linalg.norm(cur - centre) > NEAR_M:
            return None
        ks = np.arange(f + 1, min(f + 1 + AHEAD_FRAMES, T))
        ks = ks[M[ks, a]]
        if ks.size == 0:
            return None
        pts = np.vstack([cur[None], P[ks, a]])
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        keep = arc <= AHEAD_M
        hw = self.size[a, 1] / 2 + cl.EGO_RECT[1] / 2 + MARGIN
        d = seg_dist(pts[keep], centre, tan, cl.EGO_RECT[0] / 2)
        hit = np.where(d < hw)[0]
        if hit.size == 0:
            return None
        i = int(hit[0])
        j = min(i + 1, len(pts) - 1)
        step = pts[j] - pts[max(j - 1, 0)]
        n = np.linalg.norm(step)
        direction = step / n if n > 1e-6 else tan
        return float(arc[i] - self.size[a, 0] / 2), direction

    def position(self, a: int, tau: float):
        P, M, T = self.log["pos"], self.log["mask"], self.sc["T"]
        f0 = int(np.floor(tau + 1e-9))
        w = tau - f0
        if w < 1e-9 or f0 + 1 >= T:
            return P[min(f0, T - 1), a], bool(M[min(f0, T - 1), a])
        return (1 - w) * P[f0, a] + w * P[f0 + 1, a], bool(M[f0, a] and M[f0 + 1, a])

    def step(self, g: int, centre, tan, v_ego: float) -> None:
        """Move every agent to frame g (called by run_episode before the contact check)."""
        self.ego[g] = (centre, tan, v_ego)
        self.tau[~self.react] = g                                # non-reactive agents replay their log
        ec, et, ev = self.ego_at(g - 1 - DELAY)
        for a in np.where(self.react)[0]:
            t0 = self.tau[a]
            lagfree = t0 == g - 1
            if lagfree and not (self.log["mask"][g - 1, a] or self.log["mask"][g, a]):
                self.tau[a] = g
                continue
            a_max, b_comf, b_max, T_h, s0 = self.par[a]
            vl = self.vlog[int(np.floor(t0 + 1e-9)), a]
            res = self.gap_to_ego(a, t0, ec, et)
            inter = 0.0
            gap = None
            if res is not None:
                gap, direction = res
                vc = self.v[a]
                dv = vc - ev * float(et @ direction)
                s_star = s0 + max(0.0, vc * T_h + vc * dv / (2.0 * np.sqrt(a_max * b_comf)))
                inter = a_max * (s_star / max(gap, 0.1)) ** 2
            if lagfree and inter < MIN_INTERACTION:
                self.tau[a], self.v[a] = g, self.vlog[g - 1, a]
                continue                                         # replays its log exactly
            acc_free = np.clip((vl - self.v[a]) / cl.DT, -b_max, a_max)
            acc = np.clip(acc_free - inter, -b_max, a_max)
            v_new = max(0.0, self.v[a] + acc * cl.DT)
            if vl > STANDING:
                rho = min(v_new / vl, 1.0)
            else:                                                # logged as standing: keep its log
                rho = 0.0 if (gap is not None and gap < s0 + 0.5) else 1.0
                v_new = 0.0 if rho == 0.0 else vl
            t_new = min(t0 + rho, float(g))
            self.tau[a], self.v[a] = t_new, v_new
            if t_new < g:
                self.first_dev[a] = min(self.first_dev[a], g)
            p, m = self.position(a, t_new)
            self.sc["pos"][g, a], self.sc["mask"][g, a] = p, m
            self.sc["heading"][g, a] = self.log["heading"][int(np.rint(t_new)), a]
        self.tau_hist[g] = self.tau.copy()

    def logged_future(self, a: int, t: int) -> np.ndarray:
        """(60,2) the agent's log continued from its log time at frame t (NaN where absent)."""
        tau = self.tau_hist.get(t, np.full(len(self.tau), float(t)))[a]
        out = np.full((60, 2), np.nan)
        for k in range(1, 61):
            if tau + k > self.sc["T"] - 1:
                break
            p, m = self.position(a, tau + k)
            if m:
                out[k - 1] = p
        return out


# --------------------------------------------------------------------------
# Online prediction from the simulated histories
# --------------------------------------------------------------------------
def _batch(obs: np.ndarray):
    """(M,20,2) world-frame histories -> a one-sequence dd_data.Batch (closed_loop_windows layout)."""
    import torch
    from dd_data import Batch
    o = torch.from_numpy(np.ascontiguousarray(obs.transpose(0, 2, 1)[None]).astype(np.float32))
    rel = torch.zeros_like(o)
    rel[..., 1:] = o[..., 1:] - o[..., :-1]
    M = obs.shape[0]
    z = torch.zeros((1, M, 2, 60))
    return Batch(obs_abs=o, pred_abs=z, obs_rel=rel, pred_rel=z, agent_mask=torch.ones((1, M), dtype=torch.bool),
                 fut_mask=torch.zeros((1, M, 60), dtype=torch.bool), agent_type=np.full((1, M), "", dtype=object))


@lru_cache(maxsize=None)
def model_fn(key: str, seed: int, fold: int | None):
    """fn(obs (M,20,2), rng_seed) -> {"pred": (M,60,2), "samples": (M,K,60,2), "probs": (M,K)} on CPU."""
    if key == "cv":
        return lambda obs, _s: {"pred": obs[:, -1:, :] + cv_predict(obs, 5)}
    import torch
    torch.set_num_threads(1)
    from safety_scorecard import _load_script
    cpu = torch.device("cpu")
    ckpt = BENCH if fold is None else cl.CV_DIR / f"fold{fold}"
    if key in ("stgat", "transformer", "mfp", "mftf"):
        dp = _load_script("dump_predictions.py")
        fn = {"stgat": dp.load_stgat, "transformer": dp.load_transformer, "mfp": dp.load_mfp,
              "mftf": dp.load_mftf}[key](cpu, seed, ckpt)

        def run(obs, _s):
            b = _batch(obs)
            with torch.no_grad():
                out = {"pred": fn(b)[0].permute(0, 2, 1).double().numpy()}
                if hasattr(fn, "samples_fn"):
                    out["samples"] = fn.samples_fn(b)[0].permute(0, 1, 3, 2).double().numpy()
                    out["probs"] = fn.probs_fn(b)[0].double().numpy()
            return out
        return run
    if key == "agentformer":
        from dd_agentformer import predict_seq, seq_to_af_data
        lat = _load_script("safety_latency.py")
        ck = torch.load(ckpt / f"agentformer_ckpt_seed{seed}.pt", map_location=cpu, weights_only=False)
        model, cfg = lat.build_agentformer(ck, cpu)

        def run(obs, rng_seed):
            torch.manual_seed(rng_seed)
            M = obs.shape[0]
            data = seq_to_af_data(obs.transpose(0, 2, 1), np.zeros((M, 2, 60)), np.zeros((M, 60), bool),
                                  cfg.traj_scale, stride=cfg.stride)
            with torch.no_grad():
                ml, samp = predict_seq(model, data, nz=cfg.nz, sample_k=10,
                                       traj_scale=cfg.traj_scale, stride=cfg.stride)
            return {"pred": ml.double().numpy(), "samples": samp.double().numpy()}
        return run
    raise ValueError(f"no online model for {key}")


class OnlinePredictor:
    """Wraps a closed_loop_sim predictor (or one returning per-future weights): the saved
    predictions while every agent is on its log, online model outputs afterwards."""

    def __init__(self, base, fold: int | None, weighted: bool = False):
        self.base, self.fold, self.weighted = base, fold, weighted
        self.name, self.kind = base.name, base.kind
        self.key = None if base.kind == "ref" else cl.ALL_PACKAGES[base.name][1]
        self.n_online = 0

    def at(self, sc: dict, t: int, lam: float = 1.0):
        w = sc.get("world")
        if w is not None and self.name == "Oracle (true future, no band)":
            return [(a, w.logged_future(a, t)[None], np.zeros(60)) + ((np.ones(1),) if self.weighted else ())
                    for a in np.where(sc["mask"][t])[0]]
        if w is None or self.kind == "ref" or w.first_dev.min() > t:
            return self.base.at(sc, t, lam)
        self.n_online += 1
        present = np.where(sc["mask"][t])[0]
        full = [a for a in present if t >= 19 and sc["mask"][t - 19:t + 1, a].all()]
        res = {}
        if full:
            obs = sc["pos"][t - 19:t + 1, full].transpose(1, 0, 2)
            seed = int(hashlib.sha256(f"{sc['sid']}|{t}".encode()).hexdigest()[:8], 16)
            res = model_fn(self.key, self.base.seed, self.fold)(obs, seed)
        row = {a: i for i, a in enumerate(full)}
        out = []
        for a in present:
            cls = sc["cls"][a]
            if a in row:
                i = row[a]
                if self.kind == "set":
                    samples = res["samples"][i, :10]
                    wt = res["probs"][i, :10] / res["probs"][i, :10].sum() if "probs" in res \
                        else np.full(len(samples), 1.0 / len(samples))
                else:
                    samples, wt = res["pred"][i][None], np.ones(1)
                item = (a, samples, lam * cl.band_row(self.base.band, cls))
            else:
                item = (a, cl.cv_extrapolate(sc, t, a)[None], lam * cl.band_row(self.base.cv_band, cls))
                wt = np.ones(1)
            out.append(item + ((wt,) if self.weighted else ()))
        return out


# --------------------------------------------------------------------------
# Running packages
# --------------------------------------------------------------------------
@lru_cache(maxsize=4)
def log_of(sid: str) -> dict:
    sc = cl.load_scene(sid)
    return {"sc": sc, "pos": sc["pos"], "mask": sc["mask"], "heading": sc["heading"],
            "vlog": logged_speed(sc["pos"], sc["mask"])}


def episode_scene(log: dict) -> dict:
    sc = dict(log["sc"])
    sc["pos"], sc["mask"], sc["heading"] = log["pos"].copy(), log["mask"].copy(), log["heading"].copy()
    return sc


def run_package_reactive(args) -> tuple:
    """args = (package, seed, scene ids, band scale, planner cfg, {scene: fold} or None,
    p_react, sim seed, weighted). Same episodes as closed_loop_sim.run_package."""
    name, seed, scene_ids, lam, cfg, folds, p_react, sim_seed, weighted = args
    cfg = dict(cfg)
    cl.set_instrument(cfg.pop("instrument", {}))
    cl.PLAN_CFG.update({**cl.V3_DEFAULTS, **cfg})
    predictors, eps, excluded = {}, [], 0
    for sid in scene_ids:
        fold = None if folds is None else folds[sid]
        if fold not in predictors:
            predictors[fold] = OnlinePredictor(cl.Predictor(name, seed, fold), fold, weighted)
        log = log_of(sid)
        for t0 in cl.EP_STARTS:
            if not cl.episode_ok(log["sc"], t0):
                excluded += len(cl.PERSONAS)
                continue
            for persona, vmin in cl.PERSONAS.items():
                sc = episode_scene(log)
                react = reactive_draw(sid, t0, persona, sim_seed, sc["pos"].shape[1], p_react)
                world = World(sc, log, t0, react)
                sc["world"], sc["react"] = world, world.step
                r = cl.run_episode(sc, t0, vmin, predictors[fold], lam)
                r.update({"persona": persona, "fold": fold, "sim_seed": sim_seed,
                          "n_reactive": int(react.sum()), "n_deviated": int(np.isfinite(world.first_dev).sum())})
                eps.append(r)
    return name, seed, lam, eps, excluded


def _dev_scenes() -> list:
    from dd_split import load_splits
    m = load_splits(REPO / "splits" / "scene_splits.json")
    return sorted(s for s, v in m.items() if v in ("val", "test"))


def _v2_cfg():
    g = json.loads((BENCH / "closed_loop_dev_grid.json").read_text(encoding="utf-8"))["chosen"]
    return {k: g[k] for k in ("hard_s", "soft_w", "fallback")}, float(g["band"])


def check(n_scenes: int = 6) -> None:
    """p_react = 0 must reproduce the non-reactive episodes field for field."""
    cfg, lam = _v2_cfg()
    scenes = _dev_scenes()[:n_scenes]
    names = ["CV (vel_k=5)", "ST-GAT (MHA)", "K-mode set (10 modes)", "AgentFormer set (10 samples)",
             "Oracle (true future, no band)", "Human driver (log replay)"]
    fields = ("scene", "t_start", "persona", "score", "at_fault", "progress", "harsh_braking", "near_miss",
              "min_clearance_m")
    for n in names:
        a = cl.run_package((n, 0, scenes, lam, cfg, None))[3]
        b = run_package_reactive((n, 0, scenes, lam, cfg, None, 0.0, 0, False))[3]
        same = [tuple(x[f] for f in fields) == tuple(y[f] for f in fields) for x, y in zip(a, b)]
        print(f"{n:32s} {sum(same)}/{len(same)} identical  deviated agents {sum(e['n_deviated'] for e in b)}")
        assert all(same) and len(a) == len(b)


def verify_online(n_frames: int = 40) -> None:
    """On unchanged logs the online models must reproduce the saved dense predictions
    (saved samples are float16: ~0.06-0.12 m resolution at these world coordinates, so the
    online output is compared after the same rounding). AgentFormer's samples are random:
    its deterministic most-likely output must match, and its sets' mean spread must agree."""
    rng = np.random.default_rng(0)
    scenes = _dev_scenes()
    frames = [(scenes[rng.integers(len(scenes))], int(rng.choice(list(range(19, 98, 2))))) for _ in range(n_frames)]
    for name in ("CV (vel_k=5)", "ST-GAT (MHA)", "K-mode (top-1)", "K-mode set (10 modes)", "MF transformer (top-1)",
                 "MF transformer set (10 modes)", "AgentFormer (most-likely)", "AgentFormer set (10 samples)"):
        base = cl.Predictor(name, 0, None)
        key = cl.ALL_PACKAGES[name][1]
        fn = model_fn(key, 0, None)
        errs, spread = [], [[], []]
        for sid, t in frames:
            sc = log_of(sid)["sc"]
            full = [a for a in np.where(sc["mask"][t])[0] if sc["mask"][t - 19:t + 1, a].all()]
            if not full:
                continue
            res = fn(sc["pos"][t - 19:t + 1, full].transpose(1, 0, 2), 0)
            for j, a in enumerate(full):
                i = base.rows.get((sid, t, int(sc["agent_id"][a])))
                if i is None:
                    continue
                saved = base.p[i]
                if base.kind == "single":
                    errs.append(np.nanmax(np.abs(res["pred"][j] - saved[0])))
                elif key == "agentformer":
                    for k, x in enumerate((res["samples"][j], saved)):
                        spread[k].append(np.linalg.norm(x[:, -1] - x[:, -1].mean(0), axis=-1).mean())
                else:
                    online = res["samples"][j, :saved.shape[0]].astype(np.float16).astype(np.float64)
                    errs.append(np.nanmax(np.abs(online - saved)))
        msg = f"{name:32s} "
        if errs:
            msg += f"max |online - saved| {max(errs):.4f} m over {len(errs)} agents"
        if spread[0]:
            msg += (f"mean endpoint spread online {np.mean(spread[0]):.2f} m vs saved {np.mean(spread[1]):.2f} m "
                    f"over {len(spread[0])} agents (random samples)")
        print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "verify-online"))
    args = ap.parse_args()
    if args.cmd == "check":
        check()
    else:
        verify_online()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
