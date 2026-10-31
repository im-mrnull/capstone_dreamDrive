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
    python scripts/counterfactual_sim.py draw-provenance     # regime map gate A: every result's scenes and draws
    python scripts/counterfactual_sim.py regime-resolution   # regime map: the log-only delay gate
    python scripts/counterfactual_sim.py regime-pilot        # regime map: throughput, invariance, identity
    python scripts/counterfactual_sim.py regime-anchor-check # regime map: what the baseline cell changes, and why
    python scripts/counterfactual_sim.py regime-calibration  # regime map: the whole pipeline on synthetic data
    python scripts/counterfactual_sim.py regime-tune         # regime map: margins, dev draws 60-64
    python scripts/counterfactual_sim.py regime-map          # regime map: public cells, primary scenes
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


def targets(sc: dict, vlog: np.ndarray, t0: int, with_entry: bool = False, near: float = 0.0,
            cap: int | None = MAX_TARGETS) -> list:
    """Conflict-relevant agents of the episode starting at t0 (logs only); with_entry:
    (agent, first frame in the corridor) pairs. near widens the corridor (lateral branches);
    cap=None keeps every candidate."""
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
        hw = cl.EGO_RECT[1] / 2 + cl.rect_size(sc["cls"][a])[1] / 2 + CORRIDOR_MARGIN + near
        for g in frames:
            p = sc["pos"][g, a]
            i = int(np.argmin(np.linalg.norm(route.xy - p, axis=1)))
            if abs(float((p - route.xy[i]) @ route.nrm[i])) <= hw and s0 + MIN_AHEAD <= route.s_grid[i] <= s0 + reach:
                found.append((g, int(a)))
                break
    return [(a, g) if with_entry else a for g, a in sorted(found)[:cap]]


def branch_draw(sid: str, t0: int, persona: str, draw_seed: int, agent_id: int,
                branches: tuple = BRANCHES, prior: tuple = PRIOR) -> str:
    """Hidden branch of one target (identical for every package and dose)."""
    h = int(hashlib.sha256(f"{sid}|{t0}|{persona}|{draw_seed}|{agent_id}".encode()).hexdigest()[:16], 16)
    u = np.random.default_rng(h).random()
    return branches[int(np.searchsorted(np.cumsum(prior), u, side="right"))]


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
    Writes the simulated positions into the episode's copies of sc["pos"/"mask"/"heading"].
    The regime map (docs/wiki/headroom-test.md section 14): delay (frames) lets the posterior
    see the observed past only up to t - delay; anchor starts every branch's future at the
    target's current position (its own timing increments from there)."""
    delay, anchor = 0, False

    def __init__(self, sc: dict, log: dict, t0: int, persona: str, draw_seed: int, dose: tuple,
                 delay: int = 0, anchor: bool = False):
        self.sc, self.log, self.t0 = sc, log, t0
        self.delay, self.anchor = delay, anchor
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
        """(60,2) target a's positions at frames t+1..t+60 under branch b (NaN where absent);
        anchored: log time tau_real[t] + tau_b[t+k] - tau_b[t] (the identity for the realised branch)."""
        out = np.full((60, 2), np.nan)
        tau = self.tau[(a, b)]
        shift = self.tau[(a, self.branch[a])][t] - tau[t] if self.anchor else 0.0
        for k in range(1, 61):
            g = t + k
            if g > self.sc["T"] - 1:
                break
            p, m = self.position(a, tau[g] + shift)
            if m:
                out[k - 1] = p
        return out

    def posterior(self, a: int, t: int) -> dict:
        """{branch: weight} over the branches consistent with target a's observed past up to
        t - delay."""
        real = self.tau[(a, self.branch[a])]
        keep = {}
        for b, p in zip(BRANCHES, PRIOR):
            tb = self.tau[(a, b)]
            ok = all(np.linalg.norm(np.nan_to_num(self.position(a, tb[g])[0] - self.position(a, real[g])[0])) < CONSISTENT_M
                     for g in range(self.t0 + 1, max(self.t0, t - self.delay) + 1))
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


# --------------------------------------------------------------------------
# Lateral intent branches (docs/wiki/headroom-test.md section 12)
# --------------------------------------------------------------------------
PATH_BRANCHES = ("logged", "stay out", "enter")
PATH_PRIOR = (0.5, 0.25, 0.25)
PRIOR_VARIANTS = {"": PATH_PRIOR, "-p60": (0.6, 0.2, 0.2), "-p40": (0.4, 0.3, 0.3)}   # dev sensitivity only
PATH_SCALES = {"path-primary": 1.0, "path-strong": 1.5, "path-none": 0.0}             # scale on LAT_LIMITS
PATH_DOSES = {d + v: (s, p) for d, s in PATH_SCALES.items() for v, p in PRIOR_VARIANTS.items()}
LAT_LIMITS = {"ped": (1.0, 1.0), "two_wheeler": (1.5, 2.0), "vehicle": (1.0, 1.5)}   # lateral m/s, m/s^2
CONTEXT_LABELS = {"same": {"stay out": "hold lane", "enter": "cut in"},
                  "crossing": {"stay out": "wait", "enter": "cross"}}
NEAR_M = 2.0                        # m: lateral targets come within the corridor + NEAR_M
PATH_LEAD = 10                      # frames: lateral branches start 1 s before the corridor approach
FRAME_WINDOW = 40                   # route grid points (10 m): per-frame search window of the lateral frame
EDGE_M = 0.3                        # m: a stay-out target stops this far outside the corridor
AWAY_MPS = 0.1                      # m/s: a target moving away from the centreline faster has no enter branch
OVERLAP_S = 0.5                     # s: a branch overlapping a logged agent longer than this excludes the target
DEVIATES_M = 0.05                   # m: frames where a branch is this far off its log count for overlaps
DENSITY_M = 30.0                    # m: radius for the reported scene density
_PATH_CACHE: dict = {}


def route_lateral(route, P: np.ndarray):
    """Nearest route grid index and signed lateral offset of each point of P (T, 2)."""
    idx = np.zeros(len(P), dtype=int)
    d = np.full(len(P), np.nan)
    ok = np.isfinite(P[:, 0])
    if ok.any():
        q = P[ok]
        i = np.argmin(((q[:, None] - route.xy[None]) ** 2).sum(-1), axis=1)
        idx[ok], d[ok] = i, ((q - route.xy[i]) * route.nrm[i]).sum(-1)
    return idx, d


def lateral_frame(route, P: np.ndarray, M: np.ndarray, half: int = 5):
    """A smooth lateral frame along an agent's logged track: its arc length on the route
    (off the 0.25 m grid with the tangent) averaged over +-half frames, the route normal
    interpolated there, and the signed lateral offset of the log in that frame. The nearest
    grid point alone jumps on curves, which would shake a branch held metres off its log."""
    T = len(P)
    idx, prev = np.zeros(T, dtype=int), None
    for g in np.where(M)[0]:                     # continuous: search near the previous frame's index
        lo, hi = (0, len(route.s_grid)) if prev is None else (max(prev - FRAME_WINDOW, 0), prev + FRAME_WINDOW + 1)
        idx[g] = lo + int(np.argmin(((route.xy[lo:hi] - P[g]) ** 2).sum(-1)))
        prev = idx[g]
    s = np.full(T, np.nan)
    s[M] = route.s_grid[idx[M]] + ((P[M] - route.xy[idx[M]]) * route.tan[idx[M]]).sum(-1)
    sm = s.copy()
    for g in np.where(M)[0]:
        w = s[max(0, g - half):g + half + 1]
        sm[g] = np.nanmean(w)
    sm = np.where(M, np.clip(sm, 0.0, route.s_grid[-1]), 0.0)
    xy = np.stack([np.interp(sm, route.s_grid, route.xy[:, j]) for j in range(2)], 1)
    nrm = np.stack([np.interp(sm, route.s_grid, route.nrm[:, j]) for j in range(2)], 1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    d = np.where(M, ((P - xy) * nrm).sum(-1), np.nan)
    return idx, nrm, d, sm


def _frame_point(route, s: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Route point at arc length s plus d along the interpolated normal."""
    xy = np.stack([np.interp(s, route.s_grid, route.xy[:, j]) for j in range(2)], 1)
    nrm = np.stack([np.interp(s, route.s_grid, route.nrm[:, j]) for j in range(2)], 1)
    return xy + d[:, None] * nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)


def retimed_hold(route, P: np.ndarray, M: np.ndarray, s_log: np.ndarray, d_log: np.ndarray, d_b: np.ndarray,
                 t_div: int) -> np.ndarray:
    """A same-direction hold-lane branch that keeps the log's step length: where it no longer
    follows the log's lateral motion it advances along the route by the rest of the step,
    sqrt(|log step|^2 - (its lateral step)^2), so a logged swerve does not become a speed dip."""
    s_b, extra = s_log.copy(), 0.0                          # extra: progress beyond the log (kept over gaps)
    for g in range(t_div + 1, len(P)):
        if M[g] and M[g - 1]:
            step2 = float(((P[g] - P[g - 1]) ** 2).sum())
            extra += (np.sqrt(max(step2 - (d_b[g] - d_b[g - 1]) ** 2, 0.0))
                      - np.sqrt(max(step2 - (d_log[g] - d_log[g - 1]) ** 2, 0.0)))
        if M[g]:
            s_b[g] = s_log[g] + extra                       # = the log wherever the lateral steps agree
    s_b = np.clip(s_b, 0.0, route.s_grid[-1])
    out = P.copy()
    ok = M & np.isfinite(d_b) & np.isfinite(d_log)
    out[ok] = P[ok] + _frame_point(route, s_b[ok], d_b[ok]) - _frame_point(route, s_log[ok], d_log[ok])
    return out


def _stop_speed(e: float, a: float) -> float:
    """Largest speed from which braking by a * DT per frame stops within e metres."""
    h = a * cl.DT / 2.0
    return -h + np.sqrt(h * h + 2.0 * a * max(e, 0.0))


def _toward(d: float, w: float, target: float, vmax: float, a: float):
    """One frame towards target: speed <= vmax, |acceleration| <= a, braking curve on arrival."""
    e = target - d
    w_des = np.sign(e) * min(vmax, _stop_speed(abs(e), a))
    w += float(np.clip(w_des - w, -a * cl.DT, a * cl.DT))
    nd = d + w * cl.DT
    return (target, 0.0) if (target - nd) * e <= 0.0 else (nd, w)


def context_of(heading: np.ndarray, tangent: np.ndarray) -> str:
    """Direction of travel relative to the ego route: same (< 45 deg), crossing, oncoming (> 135 deg)."""
    c = float(heading @ tangent)
    return "same" if c > np.cos(np.pi / 4) else "oncoming" if c < -np.cos(np.pi / 4) else "crossing"


def path_lateral(branch: str, context: str, d_log: np.ndarray, ok: np.ndarray, t_div: int, edge: float,
                 vmax: float, a: float) -> np.ndarray:
    """(T,) lateral offset of a target under a lateral-intent branch (the log before t_div).
    No branch reverses the target's lateral motion:
    stay out  follows the log's lateral steps until it must brake to stop at the corridor
              edge + EDGE_M (at its current offset if it is already closer); from then on it
              brakes (<= a, never speeding up) and holds where it stops, never backing out and
              no longer following the log's lateral swings
    enter     same direction: cut in to the ego centreline and stay on it; crossing: cross the
              corridor to the far edge and hold; speed <= max(vmax, its own lateral speed),
              |acceleration| <= a. A target moving away from the centreline (> AWAY_MPS) at
              t_div has no enter branch (it equals the log)."""
    d = d_log.copy()
    if branch == "logged" or vmax <= 0.0:
        return d
    d0 = float(d_log[t_div])
    side = 1.0 if d0 >= 0.0 else -1.0
    k = 5 if ok[t_div - 5] else 1
    w0 = float((d_log[t_div] - d_log[t_div - k]) / (k * cl.DT)) if ok[t_div - k] else 0.0
    cur, u = d0, -side * w0                        # u: speed towards the centreline
    if branch == "enter":
        if side * w0 > AWAY_MPS:
            return d
        target = 0.0 if context == "same" else -side * edge
        vcap = max(vmax, abs(w0))
        w = float(np.clip(w0, -vcap, vcap))
        for g in range(t_div + 1, len(d)):
            cur, w = _toward(cur, w, target, vcap, a)
            d[g] = cur
        return d
    line = edge if side * d0 >= edge else side * d0
    holding = False
    for g in range(t_div + 1, len(d)):
        u_log = -side * (d_log[g] - d_log[g - 1]) / cl.DT if ok[g] and ok[g - 1] else 0.0
        cap = _stop_speed(side * cur - line, a)
        holding |= u_log > cap                                   # braking starts: from now on it holds
        u = max(0.0, min(u, max(cap, u - a * cl.DT))) if holding else u_log   # never speeds up, brakes <= a
        cur -= side * u * cl.DT
        d[g] = cur
    return d


def _overlap_s(sc: dict, a: int, Pb: np.ndarray, Hb: np.ndarray, frames) -> tuple:
    """Longest cumulative time (s) the branch footprint overlaps one logged agent where the
    log's own footprint does not (frames where the branch is off its log), and that agent."""
    size = cl.rect_size(sc["cls"][a])
    reach = np.hypot(*size) / 2
    count = {}
    for g in frames:
        if np.linalg.norm(Pb[g] - sc["pos"][g, a]) <= DEVIATES_M:
            continue
        mine = cl.rect(Pb[g], Hb[g], size)
        log_mine = cl.rect(sc["pos"][g, a], sc["heading"][g, a], size)
        for o in np.where(sc["mask"][g])[0]:
            if o == a:
                continue
            so = cl.rect_size(sc["cls"][o])
            if np.linalg.norm(sc["pos"][g, o] - Pb[g]) > reach + np.hypot(*so) / 2:
                continue
            other = cl.rect(sc["pos"][g, o], sc["heading"][g, o], so)
            if cl.rect_gap(mine, other) < 0.0 <= cl.rect_gap(log_mine, other):
                count[int(o)] = count.get(int(o), 0) + 1
    if not count:
        return 0.0, None
    o = max(count, key=count.get)
    return count[o] * cl.DT, o


def path_plan(log: dict, t0: int, scale: float) -> dict:
    """Targets and branch paths of one episode (logs only; identical for every package; cached).
    Candidates: targets() with the corridor widened by NEAR_M, earliest first. Oncoming ones
    are not branched; one whose stay-out or enter branch overlaps a logged agent for more than
    OVERLAP_S is excluded (logs only). The first MAX_TARGETS remaining are the targets."""
    sc, route = log["sc"], log["sc"]["route"]
    key = (sc["sid"], t0, scale)
    if key in _PATH_CACHE:
        return _PATH_CACHE[key]
    end = min(t0 + cl.EP_LEN, sc["T"] - 1)
    plan = {"targets": [], "t_div": {}, "context": {}, "P": {}, "H": {}, "records": []}
    for a, g_in in targets(sc, log["vlog"], t0, with_entry=True, near=NEAR_M, cap=None):
        if len(plan["targets"]) >= MAX_TARGETS:
            break
        t_div = max(t0, g_in - PATH_LEAD)
        P, M = sc["pos"][:, a], sc["mask"][:, a]
        idx, nrm, d_log, s_log = lateral_frame(route, P, M)
        ctx = context_of(sc["heading"][t_div, a], route.tan[idx[t_div]])
        rec = {"agent": int(a), "agent_id": int(sc["agent_id"][a]), "cls": str(sc["cls"][a]),
               "group": rs.idm_group(str(sc["cls"][a])), "context": ctx, "t_div": int(t_div),
               "density": int((np.linalg.norm(sc["pos"][t_div][sc["mask"][t_div]] - P[t_div], axis=1)
                               <= DENSITY_M).sum() - 1)}
        if ctx == "oncoming":
            plan["records"].append({**rec, "status": "oncoming"})
            continue
        edge = cl.EGO_RECT[1] / 2 + cl.rect_size(sc["cls"][a])[1] / 2 + CORRIDOR_MARGIN + EDGE_M
        vmax, acc = (scale * x for x in LAT_LIMITS[rec["group"]])
        paths, worst = {}, (0.0, None, None)
        for b in PATH_BRANCHES:
            d_b = path_lateral(b, ctx, d_log, M, t_div, edge, vmax, acc)
            dd = np.where(M, np.nan_to_num(d_b - d_log), 0.0)
            Pb = retimed_hold(route, P, M, s_log, d_log, d_b, t_div) if b == "stay out" and ctx == "same" and dd.any()                 else P + dd[:, None] * nrm
            Hb = sc["heading"][:, a] if not dd.any() else cl.agent_headings(Pb[:, None], M[:, None], route)[:, 0]
            paths[b] = (Pb, Hb)
            if b != "logged" and dd.any():
                t_ov, o = _overlap_s(sc, a, Pb, Hb, range(t_div, end + 1))
                if t_ov > worst[0]:
                    worst = (t_ov, b, o)
        rec["distinct"] = 1 + len(distinct_branches({b: paths[b][0] for b in PATH_BRANCHES}, M, t_div, end))
        rec["overlap_s"], rec["overlap_branch"] = worst[0], worst[1]
        rec["overlap_with"] = None if worst[2] is None else str(sc["cls"][worst[2]])
        if worst[0] > OVERLAP_S:
            plan["records"].append({**rec, "status": "excluded (overlap)"})
            continue
        plan["records"].append({**rec, "status": "target"})
        plan["targets"].append(a)
        plan["t_div"][a], plan["context"][a] = t_div, ctx
        for b, (Pb, Hb) in paths.items():
            plan["P"][(a, b)], plan["H"][(a, b)] = Pb, Hb
    _PATH_CACHE[key] = plan
    return plan


class PathWorld(CFWorld):
    """One episode with lateral intent: targets keep their logged along-route timing; their
    lateral offset from the ego route follows the branch (PATH_BRANCHES, path_plan). Same
    interface as CFWorld (step / branch_future / posterior). force: every target takes that
    branch (diagnostics only)."""

    def __init__(self, sc: dict, log: dict, t0: int, persona: str, draw_seed: int, dose: tuple,
                 force: str | None = None):
        self.sc, self.log, self.t0 = sc, log, t0
        scale, self.prior = dose
        plan = path_plan(log, t0, scale)
        self.targets, self.t_div, self.context = list(plan["targets"]), plan["t_div"], plan["context"]
        self.P, self.H, self.records = plan["P"], plan["H"], plan["records"]
        self.branch = {a: force or branch_draw(sc["sid"], t0, persona, draw_seed, int(sc["agent_id"][a]),
                                               PATH_BRANCHES, self.prior) for a in self.targets}
        self.first_dev = np.full(sc["pos"].shape[1], np.inf)
        self.ego = []                                   # (frame, centre, speed): diagnostics

    def step(self, g: int, centre, tan, v_ego: float) -> None:
        self.ego.append((g, np.array(centre, dtype=float), float(v_ego)))
        for a in self.targets:
            b = self.branch[a]
            self.sc["pos"][g, a], self.sc["heading"][g, a] = self.P[(a, b)][g], self.H[(a, b)][g]
            if np.linalg.norm(self.P[(a, b)][g] - self.log["pos"][g, a]) > 1e-9:
                self.first_dev[a] = min(self.first_dev[a], g)

    def logged_future(self, a: int, t: int) -> np.ndarray:
        """(60,2) the realised future (reactive_sim's oracle; used by the zero-limit check)."""
        return self.branch_future(a, self.branch.get(a, "logged"), t)

    def branch_future(self, a: int, b: str, t: int) -> np.ndarray:
        out = np.full((60, 2), np.nan)
        g = np.arange(t + 1, min(t + 61, self.sc["T"]))
        m = self.log["mask"][g, a]
        P = self.P[(a, b)] if a in self.branch else self.log["pos"][:, a]
        out[:len(g)][m] = P[g][m]
        return out

    def posterior(self, a: int, t: int) -> dict:
        real = self.P[(a, self.branch[a])][self.t0 + 1:t + 1]
        keep = {b: p for b, p in zip(PATH_BRANCHES, self.prior)
                if not len(real) or np.linalg.norm(np.nan_to_num(self.P[(a, b)][self.t0 + 1:t + 1] - real),
                                                   axis=1).max() < CONSISTENT_M}
        z = sum(keep.values())
        return {b: p / z for b, p in keep.items()}

def eligible_episodes(scene_ids, path_scale: float | None = None) -> list:
    """(scene, start) pairs with at least one target (logs only); path_scale: lateral branches."""
    out = []
    for sid in scene_ids:
        log = rs.log_of(sid)
        for t0 in cl.EP_STARTS:
            if not cl.episode_ok(log["sc"], t0):
                continue
            if (targets(log["sc"], log["vlog"], t0) if path_scale is None else path_plan(log, t0, path_scale)["targets"]):
                out.append((sid, t0))
    return out


REPLAN_FROZEN = cl.REPLAN


def use_regime(regime: dict | None) -> tuple:
    """Install a regime of the regime map (docs/wiki/headroom-test.md section 14) in this
    process and return the world's (delay frames, anchor): the commitment interval sets
    cl.REPLAN with exact commitment. None: the frozen simulator. Every task sets it, because
    pool workers are reused. Only the anchoring check switches parts off ("anchor", "exact",
    "hold": False); the map always has all three."""
    if regime is None:
        cl.REPLAN, cl.COMMIT_EXACT = REPLAN_FROZEN, False
        return 0, False
    cl.REPLAN, cl.COMMIT_EXACT = int(round(regime["commit_s"] / cl.DT)), regime.get("exact", True)
    return int(round(regime["delay_s"] / cl.DT)), regime.get("anchor", True)


def run_package_cf(args) -> tuple:
    """args = (package, seed, scene ids, band scale, planner cfg, {scene: fold} or None,
    dose name, draw seed, weighted). Eligible episodes only (at least one target). cfg may
    carry "regime" ({"delay_s", "commit_s"}): the commitment interval is also the planner's
    hold_s."""
    name, seed, scene_ids, lam, cfg, folds, dose, draw_seed, weighted = args
    cfg = dict(cfg)
    cl.set_instrument(cfg.pop("instrument", {}))
    regime = cfg.pop("regime", None)
    delay, anchor = use_regime(regime)
    if regime is not None:
        cfg["hold_s"] = regime["commit_s"] if regime.get("hold", True) else 0.0
    cl.PLAN_CFG.update({**cl.V3_DEFAULTS, **cfg})
    path = dose in PATH_DOSES
    assert regime is None or not path, "the regime map uses the timing world"
    dose_v = (0.0, 1.0) if dose == "none" else PATH_DOSES[dose] if path else DOSES[dose]
    predictors, eps = {}, []
    for sid, t0 in eligible_episodes(scene_ids, dose_v[0] if path else None):
        fold = None if folds is None else folds[sid]
        if fold not in predictors:
            predictors[fold] = IdealPredictor(name, weighted) if name in IDEAL else \
                rs.OnlinePredictor(cl.Predictor(name, seed, fold), fold, weighted)
        log = rs.log_of(sid)
        for persona, vmin in cl.PERSONAS.items():
            sc = rs.episode_scene(log)
            world = PathWorld(sc, log, t0, persona, draw_seed, dose_v) if path else \
                CFWorld(sc, log, t0, persona, draw_seed, dose_v, delay, anchor)
            sc["world"], sc["react"] = world, world.step
            r = cl.run_episode(sc, t0, vmin, predictors[fold], lam)
            tg = world.targets
            r.update({"persona": persona, "fold": fold, "dose": dose, "draw_seed": draw_seed,
                      "n_targets": len(tg), "branches": [world.branch[a] for a in tg],
                      "n_deviated": int(np.isfinite(world.first_dev).sum())})
            if path:
                r["targets"] = [[str(sc["cls"][a]), world.context[a], world.branch[a]] for a in tg]
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
    elig = set(eligible_episodes(scenes, 0.0))              # lateral branches with zero limits
    for n in ["CV (vel_k=5)", "Oracle (true future, no band)"]:
        a = [e for e in cl.run_package((n, 0, scenes, lam, cfg, None))[3] if (e["scene"], e["t_start"]) in elig]
        b = run_package_cf((n, 0, scenes, lam, cfg, None, "path-none", 0, False))[3]
        same = [tuple(x[f] for f in fields) == tuple(y[f] for f in fields) for x, y in zip(a, b)]
        print(f"path-none {n:22s} {sum(same)}/{len(same)} identical", flush=True)
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


# --------------------------------------------------------------------------
# Lateral intent: scenario validity report (go/no-go before pre-registration; dev scenes)
# --------------------------------------------------------------------------
VALIDITY_JSON = rs.BENCH / "lateral_validity.json"
CELL_M = 1.0                        # m: occupancy grid
GATES = {"eligible_episodes": 80, "implausible_excess": 0.10, "separation_4s_m": 1.5, "decision_sensitive": 0.20}


def _footprint_cells(P: np.ndarray, H: np.ndarray, size) -> np.ndarray:
    """(n, 2) integer occupancy cells covered by rectangles at P (n, 2) with headings H (n, 2)."""
    L, W = size
    u, v = np.meshgrid(np.linspace(-L / 2, L / 2, max(2, int(np.ceil(L / 0.5)) + 1)),
                       np.linspace(-W / 2, W / 2, max(2, int(np.ceil(W / 0.5)) + 1)))
    u, v = u.ravel(), v.ravel()
    nrm = np.stack([-H[:, 1], H[:, 0]], 1)
    pts = P[:, None] + u[None, :, None] * H[:, None] + v[None, :, None] * nrm[:, None]
    return np.floor(pts.reshape(-1, 2) / CELL_M).astype(int)


def distinct_branches(P: dict, M: np.ndarray, lo: int, hi: int) -> list:
    """Branches that get more than CONSISTENT_M from the log within frames lo..hi (real alternatives)."""
    g = np.arange(lo, hi + 1)
    g = g[M[g]]
    return [b for b in PATH_BRANCHES[1:]
            if len(g) and np.linalg.norm(np.nan_to_num(P[b][g] - P["logged"][g]), axis=1).max() > CONSISTENT_M]


def _occupancy(sc: dict):
    """Per frame: {cell: agents} occupied at that frame; and per cell the agents that traversed
    it at any time in the scene ("ego" for the logged ego)."""
    T = sc["T"]
    now = [dict() for _ in range(T)]
    seen: dict = {}
    for a in range(sc["pos"].shape[1]):
        f = np.where(sc["mask"][:, a])[0]
        if not len(f):
            continue
        size = cl.rect_size(sc["cls"][a])
        for g in f:
            for c in map(tuple, _footprint_cells(sc["pos"][g:g + 1, a], sc["heading"][g:g + 1, a], size)):
                now[g].setdefault(c, set()).add(int(a))
                seen.setdefault(c, set()).add(int(a))
    for c in map(tuple, _footprint_cells(sc["ego_raw"], sc["ego_head_raw"], cl.EGO_RECT)):
        seen.setdefault(c, set()).add("ego")
    return now, seen


def _manoeuvre(acc: float, lat: float) -> str:
    return "lateral" if abs(lat) >= 0.5 else "brake" if acc <= -1.0 else "accelerate" if acc >= 0.5 else "maintain"


def _forced_task(job) -> dict:
    """Pool task: one dev episode (nominal persona) driven by the proposal planner with the Branch
    oracle, once per forced branch. Per 1 s window: the mean commanded acceleration, the largest
    lateral target, the ego displacement over the window (window 1: from the common start, so
    the end position) and the manoeuvre class. Decision-sensitive between two real
    alternatives if a window differs by > 1 m/s^2, > 0.5 m lateral target, > 1 m displacement
    or class. Reported for the first second (the pre-registered quantity) and any window."""
    sid, t0, scale, cfg = job
    use_planner("proposal", cfg)
    cl.set_instrument({})
    cl.PLAN_CFG.update({**cl.V3_DEFAULTS, **cfg})
    _, lam = rs._v2_cfg()
    log = rs.log_of(sid)
    out = {}
    for b in PATH_BRANCHES:
        sc = rs.episode_scene(log)
        world = PathWorld(sc, log, t0, "nominal", 0, (scale, PATH_PRIOR), force=b)
        sc["world"], sc["react"] = world, world.step
        cmds, inner = [], cl.plan

        def rec(*args, _inner=inner, _cmds=cmds):
            res = _inner(*args)
            _cmds.append(res)
            return res
        cl.plan = rec
        cl.run_episode(sc, t0, cl.PERSONAS["nominal"], IdealPredictor("Branch oracle"), lam)
        cl.plan = inner
        per_s = int(1.0 / (cl.REPLAN * cl.DT))                     # replans per second
        ego = {g: c for g, c, _ in world.ego}
        start = ego[t0 + 1] - (ego[t0 + 2] - ego[t0 + 1])          # the ego centre at t0 (common to all runs)
        win = []
        for k in range(cl.EP_LEN // 10):
            cw = cmds[k * per_s:(k + 1) * per_s]
            acc = float(np.mean([c[0] for c in cw]))
            lat = float(max((c[1] for c in cw), key=abs))
            p0 = start if k == 0 else ego[t0 + 10 * k]
            win.append({"acc": acc, "lat": lat, "disp": (ego[t0 + 10 * (k + 1)] - p0).tolist(), "class": _manoeuvre(acc, lat)})
        out[b] = win
    end = min(t0 + cl.EP_LEN, log["sc"]["T"] - 1)
    distinct = sorted({b for a in world.targets
                       for b in distinct_branches({q: world.P[(a, q)] for q in PATH_BRANCHES}, log["mask"][:, a], t0, end)},
                      key=PATH_BRANCHES.index)
    pairs = [("logged", b) for b in distinct] + ([tuple(distinct)] if len(distinct) == 2 else [])
    first = {"accel": False, "lateral": False, "position": False, "manoeuvre": False}
    anyw = dict(first)
    for p, q in pairs:
        for k, (x, y) in enumerate(zip(out[p], out[q])):
            f = {"accel": abs(x["acc"] - y["acc"]) > 1.0, "lateral": abs(x["lat"] - y["lat"]) > 0.5,
                 "position": float(np.linalg.norm(np.subtract(x["disp"], y["disp"]))) > 1.0,
                 "manoeuvre": x["class"] != y["class"]}
            for d, v in f.items():
                anyw[d] |= v
                if k == 0:
                    first[d] |= v
    return {"scene": sid, "t0": t0, "has_alternative": bool(distinct),
            "first_second": {**first, "any": any(first.values())}, "any_window": {**anyw, "any": any(anyw.values())}}


def _target_diagnostics(log: dict, t0: int, plan: dict, occ, scale: float) -> list:
    """Per target: plausibility excess over its own log (scorecard limits, heading jumps,
    lateral reversals, lateral speed / acceleration over the class limits, each on 0.5 s
    steps), occupancy labels of off-log branch frames, separation, observability and the
    invalid-mean-trajectory flags."""
    import multifuture_scorecard as ms
    sc, route = log["sc"], log["sc"]["route"]
    now, seen = occ
    end = min(t0 + cl.EP_LEN, sc["T"] - 1)
    rows = []
    for a in plan["targets"]:
        c, ctx, t_div = str(sc["cls"][a]), plan["context"][a], plan["t_div"][a]
        vmax, acc_max = (scale * x for x in LAT_LIMITS[rs.idm_group(c)])
        M = sc["mask"][:, a]
        P = {b: plan["P"][(a, b)] for b in PATH_BRANCHES}
        H = {b: plan["H"][(a, b)] for b in PATH_BRANCHES}
        distinct = distinct_branches(P, M, t_div, end)
        row = {"cls": c, "group": rs.idm_group(c), "context": ctx, "distinct": distinct, "branch": {}}
        full = t0 >= 9 and M[t0 - 9:end + 1].all()

        def heading_jump(Hb):
            h = Hb[t_div:end + 1]
            cosang = (h[2:] * h[:-2]).sum(-1)
            return bool((cosang < np.cos(np.pi / 4)).any()) if len(h) > 2 else False

        def lateral(Pb):
            _, d = route_lateral(route, np.where(M[:, None], Pb, np.nan)[t_div:end + 1])
            w = np.diff(d[::5]) / 0.5
            return d, w, np.diff(w) / 0.5

        def reversal(Pb):
            w = lateral(Pb)[1]
            w = w[np.abs(w) > 0.2]
            return bool(len(w) and (w > 0).any() and (w < 0).any())

        def over_limit(Pb):
            _, w, acc = lateral(Pb)
            return bool((np.nan_to_num(np.abs(w)) > vmax + 0.05).any() or (np.nan_to_num(np.abs(acc)) > 1.1 * acc_max).any())
        base = {"implausible": ms.implausible(P["logged"][t0 - 9:end + 1][None, None], np.array([c]))[0] > 0 if full else None,
                "heading_jump": heading_jump(H["logged"]), "reversal": reversal(P["logged"]), "over_limit": over_limit(P["logged"])}
        for b in distinct:
            off = [g for g in range(t_div, end + 1) if M[g] and np.linalg.norm(P[b][g] - P["logged"][g]) > DEVIATES_M]
            labels = {"occupied": 0, "traversed by others": 0, "own track only": 0, "unknown": 0}
            for g in off:
                cell = tuple(np.floor(P[b][g] / CELL_M).astype(int))
                who = seen.get(cell, set())
                labels["occupied" if now[g].get(cell, set()) - {int(a)} else "traversed by others" if who - {int(a)}
                       else "own track only" if who else "unknown"] += 1
            sep = np.linalg.norm(P[b] - P["logged"], axis=1)
            split = [g for g in range(t_div, end + 1) if M[g] and sep[g] > CONSISTENT_M]
            ex = {"implausible_excess": (bool(ms.implausible(P[b][t0 - 9:end + 1][None, None], np.array([c]))[0] > 0)
                                         and not base["implausible"]) if full else None,
                  "heading_jump_excess": heading_jump(H[b]) and not base["heading_jump"],
                  "reversal_excess": reversal(P[b]) and not base["reversal"],
                  "lateral_limit_excess": over_limit(P[b]) and not base["over_limit"]}
            ex["any_excess"] = any(bool(v) for v in ex.values())
            raw = {"implausible": bool(ms.implausible(P[b][t0 - 9:end + 1][None, None], np.array([c]))[0] > 0) if full else False,
                   "heading_jump": heading_jump(H[b]), "reversal": reversal(P[b]), "over_limit": over_limit(P[b])}
            ex["branch_any_flag"], ex["log_any_flag"] = any(raw.values()), any(bool(x) for x in base.values())
            row["branch"][b] = {**ex, "occupancy": labels, "split_after_s": (split[0] - t_div) * cl.DT if split else None}
        for k in (2, 4):
            for anchor, g in (("", t0 + int(k / cl.DT)), ("div_", t_div + int(k / cl.DT))):
                if len(distinct) and g < sc["T"] and M[g]:
                    pts = np.stack([P[b][g] for b in ["logged"] + distinct])
                    dd = np.linalg.norm(pts[:, None] - pts[None], axis=-1)[np.triu_indices(len(pts), 1)]
                    row[f"sep_{anchor}{k}s_pairs"] = dd.tolist()
        hw = cl.EGO_RECT[1] / 2 + cl.rect_size(c)[1] / 2 + CORRIDOR_MARGIN
        flags, far = [], []
        for g in range(t0, end + 1):
            if not M[g]:
                continue
            pts = np.stack([P[b][g] for b in PATH_BRANCHES])
            mean = sum(p * P[b][g] for b, p in zip(PATH_BRANCHES, PATH_PRIOR))
            gap = float(np.linalg.norm(pts - mean, axis=1).min())
            _, dm = route_lateral(route, mean[None])
            _, db = route_lateral(route, pts)
            others = seen.get(tuple(np.floor(mean / CELL_M).astype(int)), set()) - {int(a)}
            invalid_place = (abs(dm[0]) <= hw and not (np.abs(db) <= hw).any()) or not others
            far.append(gap > 1.0)
            flags.append((g, gap > 1.0 and invalid_place))
        row["mean_far_rate"] = float(np.mean(far)) if far else 0.0
        row["imtr_steps"] = [int(f) for _, f in flags]
        row["imtr_first3s"] = any(f for g, f in flags if g <= t0 + 30)
        rows.append(row)
    return rows


def path_validity(workers: int) -> dict:
    """The scenario validity report of docs/wiki/headroom-test.md section 12 (dev scenes only,
    log-only plus forced-branch oracle runs; no G1 quantity is computed). Gates as fixed in the
    plan; where its wording allows several readings, every reading is reported."""
    import json
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    scenes, cfg = rs._dev_scenes(), frozen_proposal()
    out = {"gates": GATES, "simulator_sha": _git("rev-parse", "HEAD"), "dirty": _git("status", "--porcelain") != "",
           "when": datetime.now().isoformat(timespec="seconds")}
    for dose in ("path-primary", "path-strong"):
        scale = PATH_SCALES[dose]
        recs, rows, el, lost = [], [], [], 0
        for sid in scenes:
            log = rs.log_of(sid)
            occ = None
            for t0 in cl.EP_STARTS:
                if not cl.episode_ok(log["sc"], t0):
                    continue
                plan = path_plan(log, t0, scale)
                recs += plan["records"]
                if not plan["targets"]:
                    lost += any(r["status"] == "excluded (overlap)" for r in plan["records"])
                    continue
                el.append((sid, t0))
                occ = occ or _occupancy(log["sc"])
                rows += [{**r, "episode": f"{sid}|{t0}"} for r in _target_diagnostics(log, t0, plan, occ, scale)]
        assert el == eligible_episodes(scenes, scale)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            forced = list(ex.map(_forced_task, [(sid, t0, scale, cfg) for sid, t0 in el]))
        st = Counter(r["status"] for r in recs)
        cand = [r for r in recs if r["status"] != "oncoming"]
        excl = [r for r in cand if r["status"] != "target"]

        def share(xs):
            xs = [x for x in xs if x is not None]
            return {"rate": float(np.mean(xs)) if xs else None, "n": len(xs)}
        plaus, occ_t, obs = {}, {}, {}
        crit = ("implausible_excess", "heading_jump_excess", "reversal_excess", "lateral_limit_excess", "any_excess",
                "branch_any_flag", "log_any_flag")
        for b in PATH_BRANCHES[1:]:
            for ctx in ("same", "crossing", "all"):
                br = [r["branch"][b] for r in rows if b in r["branch"] and ctx in (r["context"], "all")]
                lab = f"{CONTEXT_LABELS[ctx][b]} ({ctx})" if ctx != "all" else f"{b} (all)"
                plaus[lab] = {k: share([x[k] for x in br]) for k in crit}
                tot = Counter()
                for x in br:
                    tot.update(x["occupancy"])
                n = sum(tot.values())
                occ_t[lab] = {k: (v / n if n else None) for k, v in tot.items()} | {"frames": n}
                sp = [x["split_after_s"] for x in br]
                obs[lab] = {"n": len(sp), "median_split_s": float(np.median(sp)) if sp else None,
                            **{f"ambiguous_ge_{k}s": float(np.mean([s >= k for s in sp])) if sp else None for k in (1, 2, 3)}}
        alt = [f for f in forced if f["has_alternative"]]
        for v in plaus.values():
            if v["branch_any_flag"]["rate"] is not None:
                v["added_points (branch rate - log rate)"] = v["branch_any_flag"]["rate"] - v["log_any_flag"]["rate"]
        sep = {}
        for k in (2, 4):
            have = [r for r in rows if f"sep_div_{k}s_pairs" in r]
            sep[f"{k}s after branch start"] = {
                "median_of_target_max_m": float(np.median([max(r[f"sep_div_{k}s_pairs"]) for r in have])),
                "median_over_all_pairs_m": float(np.median([d for r in have for d in r[f"sep_div_{k}s_pairs"]])),
                "n_targets": len(have)}
            have = [r for r in rows if f"sep_{k}s_pairs" in r]
            sep[f"{k}s"] = {"median_of_target_max_m": float(np.median([max(r[f"sep_{k}s_pairs"]) for r in have])),
                            "median_over_all_pairs_m": float(np.median([d for r in have for d in r[f"sep_{k}s_pairs"]])),
                            "median_of_target_min_m": float(np.median([min(r[f"sep_{k}s_pairs"]) for r in have])),
                            "n_targets": len(have)}
        ds = {w: {**{d: float(np.mean([f[w][d] for f in forced])) for d in ("accel", "lateral", "position", "manoeuvre", "any")},
                  "with_alternative": float(np.mean([f[w]["any"] for f in alt])) if alt else None}
              for w in ("first_second", "any_window")}
        row = {
            "eligible_episodes": len(el), "episodes_lost_to_overlap_rule": lost,
            "candidates": dict(st),
            "excluded_overlap_share": len(excl) / len(cand) if cand else None,
            "excluded_by_group_context": dict(Counter(f"{r['group']} / {r['context']}" for r in excl)),
            "candidates_by_group_context": dict(Counter(f"{r['group']} / {r['context']}" for r in cand)),
            "density_included_vs_excluded": [float(np.mean([r["density"] for r in cand if r["status"] == "target"])),
                                             float(np.mean([r["density"] for r in excl])) if excl else None],
            "overlap_with": dict(Counter(r["overlap_with"] for r in excl)),
            "targets_by_group_context_alternatives": dict(Counter(f"{r['group']} / {r['context']} / {len(r['distinct'])} real alternatives"
                                                                  for r in rows)),
            "targets_without_real_alternative": float(np.mean([not r["distinct"] for r in rows])),
            "plausibility": plaus, "occupancy": occ_t, "observability": obs, "separation": sep,
            "imtr": {"timestep_rate": float(np.mean([f for r in rows for f in r["imtr_steps"]])),
                     "episode_rate_first3s": float(np.mean([any(r["imtr_first3s"] for r in rows if r["episode"] == e)
                                                            for e in {r["episode"] for r in rows}])),
                     "target_rate_first3s": float(np.mean([r["imtr_first3s"] for r in rows])),
                     "mean_far_1m_rate": float(np.mean([r["mean_far_rate"] for r in rows]))},
            "decision_sensitive": {**ds, "n": len(forced), "n_with_alternative": len(alt)}}
        ctx_types = {k: v["any_excess"] for k, v in plaus.items() if "(all)" not in k}
        row["go"] = {
            "eligible": len(el) >= GATES["eligible_episodes"],
            "plausibility (any criterion, every context branch type)":
                max(v["rate"] or 0.0 for v in ctx_types.values()) <= GATES["implausible_excess"],
            "plausibility (any criterion, context branch types with n >= 20)":
                max((v["rate"] or 0.0) for v in ctx_types.values() if v["n"] >= 20) <= GATES["implausible_excess"],
            "separation (median over all pairs)": sep["4s"]["median_over_all_pairs_m"] >= GATES["separation_4s_m"],
            "separation (median of each target's largest pair)": sep["4s"]["median_of_target_max_m"] >= GATES["separation_4s_m"],
            "decision_sensitive (first second, pre-registered)": ds["first_second"]["any"] >= GATES["decision_sensitive"],
            "decision_sensitive (any window, disclosed deviation)": ds["any_window"]["any"] >= GATES["decision_sensitive"]}
        out[dose] = row
        print(dose, json.dumps({k: row[k] for k in ("eligible_episodes", "episodes_lost_to_overlap_rule", "candidates",
                                                    "excluded_overlap_share", "targets_without_real_alternative",
                                                    "separation", "imtr", "decision_sensitive", "go")}, indent=1), flush=True)
        print(" plausibility:", json.dumps({k: v["any_excess"] for k, v in plaus.items()}), flush=True)
        print(" occupancy:", json.dumps(occ_t), flush=True)
        print(" observability:", json.dumps(obs), flush=True)
    VALIDITY_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved:", VALIDITY_JSON)
    return out


# --------------------------------------------------------------------------
# Lateral intent: statistics, runners (docs/wiki/headroom-test.md section 12)
# --------------------------------------------------------------------------
LAT_STAGE1_JSON = rs.BENCH / "lateral_stage1.json"
LAT_TUNE_JSON = rs.BENCH / "lateral_tune.json"
LAT_CONFIRM_JSON = rs.BENCH / "lateral_confirm.json"
LAT_CALIBRATION_JSON = rs.BENCH / "lateral_stats_calibration.json"
CONFIRM_1B_JSON = rs.BENCH / "headroom_confirm_1b.json"
LAT_G1_DRAWS = (0, 1, 2, 3, 4)
LAT_TUNE_DRAWS = (5, 6, 7, 8, 9)
LAT_EVAL_DRAWS = tuple(range(10, 20))
LAT_LATENCY_DRAWS = (10, 11)
DRAWS_1B = tuple(range(20, 30))
FLIPS_1B = 200_000                          # sign flips for the Part 1b p-values (Monte Carlo SE ~0.0003 at p 0.0125)
NB = "contingency (no branching)"           # the confidential planner's matched no-branching control
LAT_SINGLES = ("proposal | Ideal single (mean)", "proposal | Ideal single (most likely)",
               "proposal+margin | Ideal single (mean)", "contingency | Ideal single (mean)")
LAT_SETS = ("proposal | Ideal set", "contingency | Ideal set", f"{NB} | Ideal set")
LAT_PAIRS = {"H2": ("contingency | Ideal set", "contingency | Ideal single (mean)"),
             "H3": ("contingency | Ideal set", f"{NB} | Ideal set"),
             "H4": ("proposal | Ideal set", "proposal+margin | Ideal single (mean)")}
LAT_INTERACTIONS = {"H5": (("contingency | Ideal set", "contingency | Ideal single (mean)"),
                           ("proposal | Ideal set", "proposal | Ideal single (mean)"))}
LAT_SECONDARY_PAIRS = {"H3 vs the proposal worst-case set": ("contingency | Ideal set", "proposal | Ideal set")}
LAT_SECONDARY_INTERACTIONS = {"H5 with the matched no-branching set": (
    ("contingency | Ideal set", "contingency | Ideal single (mean)"), (f"{NB} | Ideal set", "proposal | Ideal single (mean)"))}
PAIRS_1B = {"contingency vs proposal": ("contingency | Ideal single (mean)", "proposal | Ideal single (mean)"),
            "contingency vs margin": ("contingency | Ideal single (mean)", "proposal+margin | Ideal single (mean)")}


def boot_counts(n_scenes: int, seed: int = 0, replicates: int | None = None) -> np.ndarray:
    """(B, scenes) multiplicities of a scene-cluster bootstrap: each replicate draws scenes with
    replacement and keeps every episode of every cell of each drawn scene (for the CIs)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_scenes, (replicates or G2M["replicates"], n_scenes))
    return np.stack([np.bincount(r, minlength=n_scenes) for r in idx]).astype(float)


def flip_signs(n_scenes: int, seed: int = 0, replicates: int | None = None) -> np.ndarray:
    """(B, scenes) random scene signs for the sign-flip randomisation tests (the p-values)."""
    rng = np.random.default_rng(seed + 7919)
    return rng.choice((-1.0, 1.0), (replicates or G2M["replicates"], n_scenes))


def _cell_arrays(cells: dict, names, metric: str, scenes: list):
    """Per-scene sums (cells, scenes), episode counts and the sign that makes larger better."""
    col, sign = {"score": (1, 1.0), "at_fault": (2, -1.0)}[metric]
    X = np.array([[cells[c]["per_scene"][s][col] for s in scenes] for c in names], float)
    N = np.array([[cells[c]["per_scene"][s][0] for s in scenes] for c in names], float)
    return X, N, sign


def contrast_test(cells: dict, plus, minus, metric: str, scenes: list, counts: np.ndarray, signs: np.ndarray) -> dict:
    """sum(plus) - sum(minus) of cell rates (a pair: plus=[x], minus=[y]; an interaction
    (a - b) - (c - d): plus=[a, d], minus=[b, c]); positive = better. Cells share their
    episodes, so the per-scene contrast c_s and count n_s give the estimate sum(c) / sum(n).
    p: one-sided scene sign-flip randomisation test (exact when each scene's contrast is
    symmetric about 0 under the null). CI: scene-cluster bootstrap percentile."""
    X, N, sign = _cell_arrays(cells, [*plus, *minus], metric, scenes)
    w = np.array([1.0] * len(plus) + [-1.0] * len(minus))[:, None]
    c, n = sign * (w * X).sum(0), N[0]
    est = float(c.sum() / n.sum())
    flipped = (signs @ c) / n.sum()
    boot = (counts @ c) / (counts @ n)
    return {"gain": est, "p": float((np.sum(flipped >= est) + 1) / (len(flipped) + 1)),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def max_vs_max(cells: dict, sets, singles, metric: str, scenes: list, counts: np.ndarray, signs: np.ndarray) -> dict:
    """H1: best set minus best single (highest score / lowest at-fault, chosen per metric).
    The null is composite (best set <= best single), and "all cells equal" is not its least
    favourable point when some singles are worse. p is therefore a union bound over which
    single is truly best: for each single i, the null puts every set at single i and the other
    singles out of contention; each cell's scene deviations, with its systematic gap removed,
    are sign-flipped jointly per scene (valid with unequal cell noise), and
    t_i = max over sets - single i; p = max_i share(t_i >= observed gain). Conservative when
    singles tie. CI: scene-cluster bootstrap percentile, selection redone per replicate.
    Positive = sets better."""
    names = list(sets) + list(singles)
    X, N, sign = _cell_arrays(cells, names, metric, scenes)
    k, n = len(sets), N[0]
    est = sign * X.sum(1) / n.sum()
    gain = float(est[:k].max() - est[k:].max())
    D = sign * (X - X.mean(0))
    D = D - D.sum(1, keepdims=True) / n.sum() * n[None]                        # gap-free deviations
    fl = (signs @ D.T) / n.sum()                                               # (B, cells)
    p_i = [float((np.sum(fl[:, :k].max(1) - fl[:, k + i] >= gain) + 1) / (len(fl) + 1)) for i in range(len(singles))]
    reps = sign * (counts @ X.T) / (counts @ N.T)
    boot = reps[:, :k].max(1) - reps[:, k:].max(1)
    win = {c: float(np.mean(reps[:, :k].argmax(1) == i)) for i, c in enumerate(sets)}
    win.update({c: float(np.mean(reps[:, k:].argmax(1) == i)) for i, c in enumerate(singles)})
    return {"x": names[int(est[:k].argmax())], "y": names[k + int(est[k:].argmax())], "gain": gain,
            "p": max(p_i), "p_by_single": dict(zip(singles, p_i)),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "replicate_winners": win}


def holm(tests: list) -> None:
    """Adds p_holm to every test (in place)."""
    run, n = 0.0, len(tests)
    for r, i in enumerate(np.argsort([t["p"] for t in tests], kind="stable")):
        run = max(run, min(1.0, (n - r) * tests[i]["p"]))
        tests[i]["p_holm"] = run


def lateral_tests(cells: dict, eligible_planners: set, dev_ranks: dict | None = None, seed: int = 0) -> dict:
    """H1 (max-vs-max, joint sign-flip), H2-H4 (pairs) and H5 (interaction) by scene sign-flip;
    score and at-fault each; Holm over all (10 when every planner is eligible). Secondary,
    outside the family: H1 with the cells fixed by the dev ranks, H3 against the proposal
    worst-case set, H5 with the matched no-branching set. H1 reading rule as section 10."""
    ok = lambda c: c in cells and c.split(" | ")[0] in eligible_planners     # noqa: E731
    singles, sets = [c for c in LAT_SINGLES if ok(c)], [c for c in LAT_SETS if ok(c)]
    scenes = sorted(cells[singles[0]]["per_scene"])
    assert all(cells[c]["per_scene"][s][0] == cells[singles[0]]["per_scene"][s][0]
               for c in singles + sets for s in scenes), "cells must share their episodes"
    counts, signs = boot_counts(len(scenes), seed), flip_signs(len(scenes), seed)
    pair = lambda x, y, m: contrast_test(cells, [x], [y], m, scenes, counts, signs)           # noqa: E731
    inter = lambda a, b, m: contrast_test(cells, [a[0], b[1]], [a[1], b[0]], m, scenes, counts, signs)  # noqa: E731
    tests = [{"test": "H1", "metric": m, **max_vs_max(cells, sets, singles, m, scenes, counts, signs)}
             for m in ("score", "at_fault")]
    for h, (x, y) in LAT_PAIRS.items():
        if ok(x) and ok(y):
            tests += [{"test": h, "metric": m, "x": x, "y": y, **pair(x, y, m)} for m in ("score", "at_fault")]
    for h, (a, b) in LAT_INTERACTIONS.items():
        if all(ok(c) for c in (*a, *b)):
            tests += [{"test": h, "metric": m, "x": list(a), "y": list(b), **inter(a, b, m)} for m in ("score", "at_fault")]
    holm(tests)
    for t in tests:
        t["significant"] = bool(t["p_holm"] < G2M["alpha"])
        t["meets_threshold"] = bool(t["gain"] >= G2M[t["metric"]])
    h1 = [t for t in tests if t["test"] == "H1"]
    reading = ("meaningful benefit" if any(t["significant"] and t["meets_threshold"] for t in h1) else
               "small but real benefit" if any(t["significant"] for t in h1) else "no demonstrated benefit")
    secondary = []
    for h, (x, y) in LAT_SECONDARY_PAIRS.items():
        if ok(x) and ok(y):
            secondary += [{"test": h, "metric": m, "x": x, "y": y, **pair(x, y, m)} for m in ("score", "at_fault")]
    for h, (a, b) in LAT_SECONDARY_INTERACTIONS.items():
        if all(ok(c) for c in (*a, *b)):
            secondary += [{"test": h, "metric": m, "x": list(a), "y": list(b), **inter(a, b, m)} for m in ("score", "at_fault")]
    for m, (x, y) in (dev_ranks or {}).items():
        if ok(x) and ok(y):
            secondary.append({"test": "H1 with the dev-ranked cells", "metric": m, "x": x, "y": y, **pair(x, y, m)})
    return {"eligible_planners": sorted(eligible_planners), "family_size": len(tests), "tests": tests,
            "h1_reading": reading, "secondary": secondary}


def tests_1b(cells: dict, seed: int = 0) -> dict:
    """Part 1b: contingency single vs proposal single and vs margin single, at-fault (primary)
    and score, one-sided scene sign-flip tests, Holm over 4. Confirmed best single setup: both
    at-fault tests significant; better than the plain planner only: only that at-fault test."""
    scenes = sorted(cells["proposal | Ideal single (mean)"]["per_scene"])
    assert all(cells[c]["per_scene"][s][0] == cells["proposal | Ideal single (mean)"]["per_scene"][s][0]
               for pair in PAIRS_1B.values() for c in pair for s in scenes), "cells must share their episodes"
    counts, signs = boot_counts(len(scenes), seed), flip_signs(len(scenes), seed, FLIPS_1B)
    tests = [{"test": h, "metric": m, "x": x, "y": y, **contrast_test(cells, [x], [y], m, scenes, counts, signs)}
             for h, (x, y) in PAIRS_1B.items() for m in ("at_fault", "score")]
    holm(tests)
    sensitivity = []                         # secondary: the at-fault contrast without its largest-contributing scene
    for h, (x, y) in PAIRS_1B.items():
        X, _, sign = _cell_arrays(cells, [x, y], "at_fault", scenes)
        top = scenes[int(np.argmax(sign * (X[0] - X[1])))]
        rest = [s for s in scenes if s != top]
        sensitivity.append({"test": h, "metric": "at_fault", "removed_scene": top, **contrast_test(
            cells, [x], [y], "at_fault", rest, boot_counts(len(rest), seed), flip_signs(len(rest), seed, FLIPS_1B))})
    sig = {t["test"]: t["p_holm"] < G2M["alpha"] for t in tests if t["metric"] == "at_fault"}
    reading = ("confirmed best single setup" if all(sig.values()) else
               "better than the plain planner only" if sig["contingency vs proposal"] else "not confirmed")
    return {"tests": tests, "reading": reading, "flips": FLIPS_1B, "sensitivity_largest_scene_removed": sensitivity}


def repro(cfgs: dict, scene_ids) -> dict:
    """Reproducibility fields: simulator commit, split-file and scene-data hashes, config hash."""
    import hashlib
    import json
    h = hashlib.sha256()
    for sid in sorted(scene_ids):
        h.update(hashlib.sha256((cl.SCENES / f"{sid}.npz").read_bytes()).digest())
    h.update(hashlib.sha256(cl.EGO_CSV.read_bytes()).digest())
    return {"simulator_sha": _git("rev-parse", "HEAD"),
            "split_sha256": hashlib.sha256((rs.REPO / "splits" / "scene_splits.json").read_bytes()).hexdigest(),
            "scene_data_sha256": h.hexdigest(),
            "config_sha256": hashlib.sha256(json.dumps(cfgs, sort_keys=True, default=str).encode()).hexdigest()}


def episode_records(jobs, res) -> dict:
    """{cell: [[scene, start, persona, draw, targets (class, context, branch) or branches, score, at-fault]]}."""
    out = {}
    for (planner, args), r in zip(jobs, res):
        out.setdefault(f"{planner} | {args[0]}", []).extend(
            [e["scene"], e["t_start"], e["persona"], e["draw_seed"], e.get("targets", e["branches"]),
             round(e["score"], 6), bool(e["at_fault"])] for e in r[3])
    return out


def _lat_task(job) -> tuple:
    """Pool task: (planner, run_package_cf args) with planner "proposal" (any margin via cfg)."""
    planner, args = job
    use_planner("proposal" if planner.startswith("proposal") else planner, args[4])
    return run_package_cf(args)


def lateral_dose() -> str:
    import json
    dose = json.loads(LAT_STAGE1_JSON.read_text(encoding="utf-8"))["dose_for_later_stages"]
    assert dose is not None, "G1 failed at both lateral doses: the test stops"
    return dose


def path_stage1(workers: int) -> None:
    """G1 for lateral intent (dev draws 0-4, proposal planner): Branch oracle vs CV per dose."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    ref, cfg = preregistered(), frozen_proposal()
    _, lam = rs._v2_cfg()
    scenes = rs._dev_scenes()
    doses = ("path-primary", "path-strong")
    jobs = [("proposal", (n, 0, scenes, lam, cfg, None, d, s, False))
            for d in doses for s in LAT_G1_DRAWS for n in ("CV (vel_k=5)", "Branch oracle")]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_lat_task, jobs))
    out = {d: {} for d in doses}
    for (_, a), r in zip(jobs, res):
        out[a[6]].setdefault(a[0], []).extend(r[3])
    out = {d: {n: summary(e) for n, e in row.items()} for d, row in out.items()}
    gate = {}
    for d in doses:
        o, c = out[d]["Branch oracle"], out[d]["CV (vel_k=5)"]
        ds, da = o["score"] - c["score"], c["at_fault_rate"] - o["at_fault_rate"]
        gate[d] = {"score_gain": ds, "at_fault_reduction": da, "passes": bool(ds >= G1["score"] or da >= G1["at_fault"])}
    chosen = "path-primary" if gate["path-primary"]["passes"] else "path-strong" if gate["path-strong"]["passes"] else None
    payload = {**ref, **repro({"proposal": cfg}, scenes), "draws": list(LAT_G1_DRAWS), "planner_cfg": cfg,
               "packages": out, "g1": gate, "dose_for_later_stages": chosen,
               "when": datetime.now().isoformat(timespec="seconds")}
    LAT_STAGE1_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for d in doses:
        for n, v in out[d].items():
            print(f"{d:13s} {n:14s} score {v['score']:.3f}  at-fault {v['at_fault_rate']:.4f}  near {v['near_miss_rate']:.3f}  "
                  f"harsh {v['harsh_braking_rate']:.3f}  progress {v['progress']:.3f}")
        g = gate[d]
        print(f"G1 {d}: score {g['score_gain']:+.4f}  at-fault {g['at_fault_reduction']:+.4f}  {'PASS' if g['passes'] else 'fail'}")
    print("dose for later stages:", chosen, "| saved:", LAT_STAGE1_JSON)


def path_tune(workers: int) -> None:
    """Dev draws 5-9 at the G1 dose: the margin for the Ideal single (mean) (best score), then the
    public cells at the three priors (sensitivity; 50/25/25 also gives the dev ranks)."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    ref, dose = preregistered(), lateral_dose()
    _, lam = rs._v2_cfg()
    base, scenes = frozen_proposal(), rs._dev_scenes()
    jobs = [("proposal", ("Ideal single (mean)", 0, scenes, lam, {**base, "margin_m": m}, None, dose, d, False))
            for m in MARGINS for d in LAT_TUNE_DRAWS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_lat_task, jobs))
        rows = [{"margin_m": m, "summary": summary([e for (_, a), r in zip(jobs, res) if a[4]["margin_m"] == m for e in r[3]])}
                for m in MARGINS]
        margin = max(rows, key=lambda r: r["summary"]["score"])["margin_m"]
        cfgs = {"proposal": base, "proposal+margin": {**base, "margin_m": margin}}
        cells_of_prior = {("proposal", "Ideal single (mean)"), ("proposal", "Ideal single (most likely)"),
                          ("proposal", "Ideal set"), ("proposal+margin", "Ideal single (mean)")}
        sjobs = [(p, (n, 0, scenes, lam, cfgs[p], None, dose + v, d, False))
                 for v in PRIOR_VARIANTS for p, n in sorted(cells_of_prior) for d in LAT_TUNE_DRAWS]
        sres = list(ex.map(_lat_task, sjobs))
    sens = {}
    for v in PRIOR_VARIANTS:
        part = [(j, r) for j, r in zip(sjobs, sres) if j[1][6] == dose + v]
        cells = cells_of([j for j, _ in part], [r for _, r in part])
        sens["/".join(str(int(round(100 * p))) for p in PRIOR_VARIANTS[v])] = \
            {f"{p} | {n}": {k: c[k] for k in ("score", "at_fault_rate", "harsh_braking_rate", "near_miss_rate", "progress")}
             for p, row in cells.items() for n, c in row.items()}
    payload = {**ref, **repro(cfgs, scenes), "dose": dose, "draws": list(LAT_TUNE_DRAWS),
               "margin_rows": rows, "chosen_margin_m": margin, "prior_sensitivity": sens,
               "when": datetime.now().isoformat(timespec="seconds")}
    LAT_TUNE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for r in rows:
        s = r["summary"]
        print(f"margin {r['margin_m']:.2f} m: score {s['score']:.3f}  at-fault {s['at_fault_rate']:.4f}  "
              f"harsh {s['harsh_braking_rate']:.3f}  near {s['near_miss_rate']:.3f}  progress {s['progress']:.3f}")
    for pr, row in sens.items():
        print(f"prior {pr}: " + "  ".join(f"{c}: {v['score']:.3f}/{v['at_fault_rate']:.4f}" for c, v in row.items()))
    print("chosen margin:", margin, "| saved:", LAT_TUNE_JSON)


def path_confirm(workers: int) -> None:
    """The public lateral cells on the 73 primary scenes, draws 10-19 (latency on 10-11)."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    ref, dose = preregistered(), lateral_dose()
    tuned = json.loads(LAT_TUNE_JSON.read_text(encoding="utf-8"))
    assert tuned["rule_commit"] == ref["rule_commit"], "tuning ran under a different rule commit"
    _, lam = rs._v2_cfg()
    base, scenes = frozen_proposal(), primary_scenes()
    cfgs = {"proposal": base, "proposal+margin": {**base, "margin_m": tuned["chosen_margin_m"]}}
    preds = {"proposal": ("Branch oracle", "Ideal single (mean)", "Ideal single (most likely)", "Ideal set"),
             "proposal+margin": ("Ideal single (mean)",)}
    jobs = [(p, (n, 0, scenes, lam, cfgs[p], None, dose, d, False)) for p in cfgs for n in preds[p] for d in LAT_EVAL_DRAWS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_lat_task, jobs))
    cells = {f"{p} | {n}": v for p, row in cells_of(jobs, res).items() for n, v in row.items()}
    lat = {p: latency(use_planner, [("proposal", (n, 0, scenes, lam, cfgs[p], None, dose, d, False))
                                    for n in ("Ideal set", "Ideal single (mean)") for d in LAT_LATENCY_DRAWS])
           for p in cfgs}
    payload = {**ref, **repro(cfgs, scenes), "dose": dose, "scenes": "73 primary scenes", "draws": list(LAT_EVAL_DRAWS),
               "planner_cfgs": cfgs, "band": lam, "cells": cells, "latency": lat, "latency_budget_ms": LATENCY_BUDGET_MS,
               "within_budget": {p: within_budget(v) for p, v in lat.items()}, "episodes": episode_records(jobs, res),
               "when": datetime.now().isoformat(timespec="seconds")}
    LAT_CONFIRM_JSON.write_text(json.dumps(payload), encoding="utf-8")
    print_cells({"": cells}, {})
    print("saved:", LAT_CONFIRM_JSON)


def confirm_1b(workers: int) -> None:
    """Part 1b public cells: timing branches (G1 dose), primary scenes, fresh draws 20-29;
    proposal and proposal + the section 10 margin, Ideal single (mean)."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    ref, dose = preregistered(), matched_dose()
    margin = json.loads(CONFIRM_TUNE_JSON.read_text(encoding="utf-8"))["chosen_margin_m"]
    _, lam = rs._v2_cfg()
    base, scenes = frozen_proposal(), primary_scenes()
    cfgs = {"proposal": base, "proposal+margin": {**base, "margin_m": margin}}
    jobs = [(p, ("Ideal single (mean)", 0, scenes, lam, cfgs[p], None, dose, d, False)) for p in cfgs for d in DRAWS_1B]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_lat_task, jobs))
    cells = {f"{p} | {n}": v for p, row in cells_of(jobs, res).items() for n, v in row.items()}
    payload = {**ref, **repro(cfgs, scenes), "dose": dose, "scenes": "73 primary scenes", "draws": list(DRAWS_1B),
               "planner_cfgs": cfgs, "cells": cells, "episodes": episode_records(jobs, res),
               "when": datetime.now().isoformat(timespec="seconds")}
    CONFIRM_1B_JSON.write_text(json.dumps(payload), encoding="utf-8")
    print_cells({"": cells}, {})
    print("saved:", CONFIRM_1B_JSON)


def calibration_criteria(scen: dict) -> dict:
    """H1 over every null (full and composite: in a composite null only H1 is null); H2-H5 and
    the family-wise rate over the full nulls only (there every cell is equal)."""
    full = {k: v for k, v in scen.items() if k.startswith("null")}
    h1n = {k: v for k, v in scen.items() if "null" in k}
    h1 = max(v["rate_raw"][f"H1 {m}"] for v in h1n.values() for m in ("score", "at_fault"))
    per_test = {t: max(v["rate_raw"].get(t, 0.0) for v in full.values()) for t in next(iter(full.values()))["rate_raw"]}
    per_test.update({f"H1 {m}": max(v["rate_raw"][f"H1 {m}"] for v in h1n.values()) for m in ("score", "at_fault")})
    winner = max([x for v in h1n.values() for x in v["h1_significant_set_winner"].values()] or [0.0])
    power = scen["power, +0.01 score on the contingency set"]["rate_holm"]["H1 score"]
    return {"h1_false_positive_max": h1, "false_positive_le_0.06": h1 <= 0.06, "per_test_false_positive_max": per_test,
            "tests_above_0.06": sorted(t for t, v in per_test.items() if v > 0.06),
            "familywise_max_full_nulls": max(v["familywise_rate_holm"] for v in full.values()),
            "max_significant_share_of_any_set_winner": winner, "no_systematic_winner": winner <= 0.06,
            "power_score_holm": power, "power_ge_0.80": power >= 0.80,
            "power_score_holm_other_scenarios": {k: v["rate_holm"]["H1 score"] for k, v in scen.items()
                                                 if k.startswith("power")}}


def stats_calibration(n_datasets: int = 1000) -> dict:
    """The procedure's calibration on synthetic data that keeps the real scene structure.
    Source: the public primary-scene cells of section 10 (timing branches, already reported):
    per scene its episode count, the proposal single's score / at-fault mean, and the four
    proposal cells' deviations from their per-scene average (each cell's overall mean removed).
    Each synthetic cell: the scene mean plus one of that scene's deviations, with a random sign
    (symmetric nulls) or as observed (the asymmetric null: real deviation shapes), one cell x3;
    all cells are equal in truth. Runs lateral_tests (Holm over 10) on each dataset with its
    own seed. Null: per-test, H1 and family-wise rejection rates, H1 winners; power: +0.01
    score added to the contingency set."""
    import json
    src = json.loads(CONFIRM_JSON.read_text(encoding="utf-8"))["cells"]
    names = ("proposal | Branch oracle", "proposal | Ideal single (mean)", "proposal | Ideal single (most likely)",
             "proposal | Ideal set")
    scenes = sorted(src[names[1]]["per_scene"])
    S = len(scenes)
    n = np.array([src[names[1]]["per_scene"][s][0] for s in scenes], float)
    m = {k: np.array([[src[c]["per_scene"][s][col] for s in scenes] for c in names], float) / n
         for k, col in (("score", 1), ("at_fault", 2))}
    dev = {k: (v - v.mean(0)) - ((v - v.mean(0)) * n).sum(1, keepdims=True) / n.sum() for k, v in m.items()}
    base = {k: m[k][1] for k in m}
    cell_names = list(LAT_SINGLES) + list(LAT_SETS)
    eligible = {"proposal", "proposal+margin", "contingency", NB}
    out = {"source": str(CONFIRM_JSON.name), "n_datasets": n_datasets, "scenes": S,
           "episodes_per_scene_median": float(np.median(n)), "p_value_method": "scene sign-flip randomisation",
           "scenarios": {}}
    best = "proposal+margin | Ideal single (mean)"
    G = lambda sg, ag: {c: (-sg, ag) for c in LAT_SINGLES if c != best}            # noqa: E731
    scen_list = (  # name, 3x-noise cell, +score on the contingency set, symmetric, {cell: (score, at-fault) shift}
        ("null, noisy set", "proposal | Ideal set", 0.0, True, {}),
        ("null, noisy single", best, 0.0, True, {}),
        ("null, asymmetric (real deviation shapes), noisy set", "proposal | Ideal set", 0.0, False, {}),
        ("composite null, other singles -0.002 score", None, 0.0, True, G(0.002, 0.0)),
        ("composite null, other singles -0.005 score", None, 0.0, True, G(0.005, 0.0)),
        ("composite null, other singles -0.01 score", None, 0.0, True, G(0.01, 0.0)),
        ("composite null, other singles -0.03 score", None, 0.0, True, G(0.03, 0.0)),
        ("composite null, other singles +0.2 pt at-fault", None, 0.0, True, G(0.0, 0.002)),
        ("composite null, other singles +0.4 pt at-fault", None, 0.0, True, G(0.0, 0.004)),
        ("composite null, -0.005 score, noisy best single", best, 0.0, True, G(0.005, 0.0)),
        ("power, +0.01 score on the contingency set", "proposal | Ideal set", 0.01, True, {}),
        ("power, +0.01 score, no noisy cell", None, 0.01, True, {}),
        ("power, +0.01 score, other singles -0.005", None, 0.01, True, G(0.005, 0.0)))
    for scen, noisy, effect, symmetric, shift in scen_list:
        rng = np.random.default_rng(12345)
        raw, hol, fw, sig_x, sig_y = {}, {}, 0, {}, {}
        for i in range(n_datasets):
            cells = {}
            for c in cell_names:
                f = 3.0 if c == noisy else 1.0
                pick = rng.integers(0, len(names), S)
                sgn = rng.choice((-1.0, 1.0), S) if symmetric else np.ones(S)
                ds, da = shift.get(c, (0.0, 0.0))
                sc_m = base["score"] + ds + f * sgn * dev["score"][pick, np.arange(S)]           # not clipped:
                af_m = base["at_fault"] + da + f * sgn * dev["at_fault"][pick, np.arange(S)]     # exact null
                if c == "contingency | Ideal set":
                    sc_m = sc_m + effect
                cells[c] = {"per_scene": {s: [n[j], n[j] * sc_m[j], n[j] * af_m[j]] for j, s in enumerate(scenes)}}
            r = lateral_tests(cells, eligible, seed=i)
            fw += any(t["p_holm"] < G2M["alpha"] for t in r["tests"])
            for t in r["tests"]:
                k = f"{t['test']} {t['metric']}"
                raw[k] = raw.get(k, 0) + (t["p"] < G2M["alpha"])
                hol[k] = hol.get(k, 0) + (t["p_holm"] < G2M["alpha"])
                if t["test"] == "H1" and t["p_holm"] < G2M["alpha"]:
                    sig_x[t["x"]] = sig_x.get(t["x"], 0) + 1
                    sig_y[t["y"]] = sig_y.get(t["y"], 0) + 1
        out["scenarios"][scen] = {"noisy_cell": noisy, "effect": effect, "symmetric": symmetric,
                                  "shift": {c: list(v) for c, v in shift.items()},
                                  "rate_raw": {k: v / n_datasets for k, v in raw.items()},
                                  "rate_holm": {k: v / n_datasets for k, v in hol.items()},
                                  "familywise_rate_holm": fw / n_datasets,
                                  "h1_significant_set_winner": {k: v / n_datasets for k, v in sig_x.items()},
                                  "h1_significant_single_loser": {k: v / n_datasets for k, v in sig_y.items()}}
        print(scen, json.dumps(out["scenarios"][scen]), flush=True)
    out["criteria"] = calibration_criteria(out["scenarios"])
    LAT_CALIBRATION_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("criteria:", out["criteria"], "| saved:", LAT_CALIBRATION_JSON)
    return out


# --------------------------------------------------------------------------
# Regime map: recognition delay x commitment interval (docs/wiki/headroom-test.md section 14)
# --------------------------------------------------------------------------
PROVENANCE_JSON = rs.BENCH / "draw_provenance.json"
REGIME_RESOLUTION_JSON = rs.BENCH / "regime_resolution.json"
REGIME_PILOT_JSON = rs.BENCH / "regime_pilot.json"
REGIME_ANCHOR_JSON = rs.BENCH / "regime_anchor_check.json"
REGIME_CALIBRATION_JSON = rs.BENCH / "regime_calibration.json"
REGIME_TUNE_JSON = rs.BENCH / "regime_tune.json"
REGIME_SAMPLE_JSON = rs.BENCH / "regime_sample_size.json"
REGIME_MAP_JSON = rs.BENCH / "regime_map.json"
REGIME_DELAYS = (0.0, 1.0, 2.0)          # s: recognition delay (rows: the futures resolve fast / medium / slow)
REGIME_COMMITS = (0.2, 0.5, 1.0)         # s: commitment interval (columns; 0.2 s is our existing 5 Hz baseline)
REGIMES = tuple((d, c) for d in REGIME_DELAYS for c in REGIME_COMMITS)
REGIME_CORNER = (REGIME_DELAYS[-1], REGIME_COMMITS[-1])
REGIME_MARGINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
REGIME_MARGINS_EDGE = (2.5, 3.0)         # added when a dev optimum is the grid's top
REGIME_DEV_DRAWS = tuple(range(60, 65))
REGIME_EVAL_DRAWS = {10: tuple(range(30, 40)), 20: tuple(range(30, 50)), 30: tuple(range(30, 60))}
REGIME_ENGINEERING_DRAWS = {"resolution": tuple(range(90, 95)), "pilot": (95, 96), "smoke test": (97, 98)}   # dev, before the rule
# cells "planner | predictor"; "+margin": the margin tuned on dev for that planner, predictor and regime
REGIME_SINGLES = ("proposal | Ideal single (mean)", "proposal+margin | Ideal single (mean)",
                  "proposal+margin | Ideal single (most likely)", "contingency | Ideal single (mean)",
                  "contingency+margin | Ideal single (mean)")
REGIME_SETS = ("proposal | Ideal set", "proposal+margin | Ideal set", "contingency | Ideal set",
               "contingency+margin | Ideal set")
REGIME_ALPHA, REGIME_FLIPS, REGIME_CALIB_FLIPS = 0.05, FLIPS_1B, 2000
REGIME_EFFECT = 0.01                     # the meaningful at-fault advantage: 1.0 percentage point
REGIME_POWER = 0.80
REGIME_FPR = 0.06                        # bound on the upper 95% Clopper-Pearson limit of every null rejection rate


def regime_key(r: tuple) -> str:
    return f"delay {r[0]:g} s, commit {r[1]:g} s"


def regime_cfg(base: dict, r: tuple, margin: float = 0.0) -> dict:
    return {**base, "margin_m": margin, "regime": {"delay_s": r[0], "commit_s": r[1]}}


# Gate A: draw provenance --------------------------------------------------
PUBLIC_RESULTS = ("headroom_*.json", "lateral_*.json", "regime_*.json")
# result files that record no scenes or no draws: (scene set, draws or None = recorded, world, runner)
UNRECORDED = {
    "headroom_stage1.json": ("dev", None, None, "stage1(): rs._dev_scenes()"),
    "headroom_matched_tune.json": ("dev", None, None, "tune(): rs._dev_scenes()"),
    "headroom_confirm_tune.json": ("dev", None, None, "confirm_tune(): rs._dev_scenes()"),
    "lateral_validity.json": ("dev", (), "path-primary/path-strong",
                              "path_validity(): rs._dev_scenes(), forced branches, no branch draws"),
    "lateral_gpu_benchmark.json": ("dev", (99,), "path-primary",
                                   "session benchmark script: rs._dev_scenes()[:8], path-primary, draw 99"),
    "lateral_stats_calibration.json": ("synthetic", (), None, "stats_calibration(): synthetic datasets, no episodes"),
    "regime_calibration.json": ("synthetic", (), None, "regime_calibration(): synthetic datasets, no episodes")}
REGIME_OWN = ("regime_",)                    # this experiment's own outputs (callers may add theirs)
CONFIRMATION_RESULTS = ("headroom_confirm.json",)   # the timing-world confirmation (reconciliation check)


def _scene_ids(o) -> set:
    """Scene ids listed anywhere in a result: per-scene cells and episode records."""
    out = set()
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "per_scene" and isinstance(v, dict):
                out |= set(v)
            elif k == "episodes" and isinstance(v, dict):
                out |= {e[0] for rows in v.values() if isinstance(rows, list) for e in rows if isinstance(e, list) and e}
            else:
                out |= _scene_ids(v)
    elif isinstance(o, list):
        for x in o:
            out |= _scene_ids(x)
    return out


def draw_provenance(patterns=PUBLIC_RESULTS, unrecorded: dict | None = None, own=REGIME_OWN,
                    confirmations=CONFIRMATION_RESULTS) -> dict:
    """Gate A: per result file, the scene set and branch draws it drove, from the file itself
    (per-scene cells, episode records, recorded draws) or, where it records none, from the
    runner that wrote it; the draws reserved by runners that never ran; the reconciliation
    checks and which of the proposed draws are free."""
    import hashlib
    import json
    unrecorded = {**UNRECORDED, **(unrecorded or {})}
    dev, primary = set(rs._dev_scenes()), set(primary_scenes())
    sha = lambda b: hashlib.sha256(b).hexdigest()                                   # noqa: E731
    rows = []
    for path in sorted({p for pat in patterns for p in rs.BENCH.glob(pat)}):
        if path == PROVENANCE_JSON:
            continue
        raw = path.read_bytes()
        d = json.loads(raw)
        ids, note = _scene_ids(d), unrecorded.get(path.name)
        if ids:
            scene_set = "dev" if ids <= dev else "primary" if ids <= primary else "mixed"
            scene_src = f"file: {len(ids)} scene ids in its cells / episode records"
        elif note:
            scene_set, scene_src = note[0], f"runner: {note[3]}"
        elif isinstance(d.get("scenes"), str):
            scene_set = "primary" if "primary" in d["scenes"] else "dev" if "development" in d["scenes"] else d["scenes"]
            scene_src = f"file: {d['scenes']!r}"
        else:
            scene_set, scene_src = "not recorded", "-"
        draws, draws_src = d.get("draws", d.get("draw_seeds")), "file"
        if draws is None:
            draws, draws_src = (list(note[1]), f"runner: {note[3]}") if note and note[1] is not None else (None, "not recorded")
        world = d.get("dose") or ("/".join(d["doses"]) if isinstance(d.get("doses"), dict) else None) or (note and note[2]) or "-"
        cfg = next((d[k] for k in ("planner_cfgs", "planner_cfg", "planner_v2_cfg") if k in d), None)
        rows.append({"file": path.name, "status": "run", "world": world, "scene_set": scene_set,
                     "scene_ids_source": scene_src, "scene_ids_sha256": sha(json.dumps(sorted(ids)).encode()) if ids else None,
                     "draws": draws, "draws_source": draws_src, "rule_commit": d.get("rule_commit"),
                     "simulator_sha": d.get("simulator_sha"),
                     "config_sha256": d.get("config_sha256") or (
                         f"{sha(json.dumps(cfg, sort_keys=True, default=str).encode())} (computed from the recorded planner config)"
                         if cfg is not None else "not recorded"),
                     "file_sha256": sha(raw), "when": d.get("when")})
    for name, scene_set, draws, out in (("lateral G1, path-stage1", "dev", LAT_G1_DRAWS, LAT_STAGE1_JSON),
                                        ("lateral tuning, path-tune", "dev", LAT_TUNE_DRAWS, LAT_TUNE_JSON),
                                        ("lateral confirmation, path-confirm", "primary", LAT_EVAL_DRAWS, LAT_CONFIRM_JSON)):
        if not out.exists():
            rows.append({"file": out.name, "status": f"pre-registered, not run ({name})", "world": "lateral",
                         "scene_set": scene_set, "scene_ids_source": "runner", "draws": list(draws),
                         "draws_source": "runner constant", "rule_commit": None, "config_sha256": "not run",
                         "file_sha256": None, "when": None})
    used = {}
    for r in rows:
        if r["status"] == "run" and r["draws"] and not r["file"].startswith(own):
            used.setdefault(r["scene_set"], set()).update(r["draws"])
    run_with = lambda ss, dr: sorted(r["file"] for r in rows if r["status"] == "run" and r["scene_set"] == ss      # noqa: E731
                                     and set(r["draws"] or ()) & set(dr) and not r["file"].startswith(own))
    proposed = {"dev, regime tuning": ("dev", REGIME_DEV_DRAWS), "primary, regime evaluation": ("primary", range(30, 60)),
                "dev, regime engineering": ("dev", tuple(range(90, 99)))}
    checks = {"timing_world_confirmation_draws": {r["file"]: (r["scene_set"], r["draws"]) for r in rows
                                                  if r["file"] in confirmations},
              "primary_draws_10_19_driven_by": run_with("primary", range(10, 20)),
              "proposed_draws_already_used_by": {k: run_with(ss, dr) for k, (ss, dr) in proposed.items()}}
    return {"rows": rows, "used_draws": {k: sorted(v) for k, v in used.items()}, "checks": checks,
            "files_scanned": len([r for r in rows if r["status"] == "run"])}


def provenance_table(prov: dict) -> str:
    """The markdown table of a draw_provenance() result."""
    def span(d):
        if not d:
            return "-" if d is None else "none"
        d = sorted(d)
        return f"{d[0]}-{d[-1]}" if d == list(range(d[0], d[-1] + 1)) and len(d) > 1 else ", ".join(map(str, d))
    lines = ["| File | Status | World | Scenes | Draws | Rule commit | Config | Sources |", "|---|---|---|---|---|---|---|---|"]
    for r in prov["rows"]:
        cfg = r["config_sha256"]
        cfg = cfg if cfg in ("not recorded", "not run") else cfg[:12] + (" (computed)" if "computed" in cfg else "")
        lines.append(f"| `{r['file']}` | {r['status']} | {r['world']} | {r['scene_set']} | {span(r['draws'])} | "
                     f"{(r['rule_commit'] or '-')[:9]} | {cfg} | scenes: {r['scene_ids_source']}; draws: {r['draws_source']} |")
    return "\n".join(lines)


def assert_draws_free(scene_set: str, draws, patterns=PUBLIC_RESULTS, unrecorded: dict | None = None,
                      own=REGIME_OWN) -> None:
    """No result outside this experiment's own has driven these draws on this scene set."""
    prov = draw_provenance(patterns, unrecorded, own)
    clash = sorted(r["file"] for r in prov["rows"] if r["status"] == "run" and r["scene_set"] == scene_set
                   and set(r["draws"] or ()) & set(draws) and not r["file"].startswith(own))
    assert not clash, f"draws {list(draws)} on the {scene_set} scenes were already driven by {clash}"


def draw_provenance_cmd() -> dict:
    import json
    prov = draw_provenance()
    PROVENANCE_JSON.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(provenance_table(prov))
    print("used draws (excluding this experiment):", prov["used_draws"])
    print("checks:", json.dumps(prov["checks"], indent=1))
    print("saved:", PROVENANCE_JSON)
    return prov


# The resolution gate (log-only) -----------------------------------------
def _resolution_task(job) -> list:
    """One dev scene, no ego driving (branches don't react to it): per eligible episode, persona,
    draw and delay, each target's time from its branch start until the Ideal set (the
    predictor's own output) holds one future; the share of 0.2 s replans with >= 2 futures in
    the episode; at delay 0 the largest anchoring shift of a live alternative future."""
    sid, draws, dose = job
    log = rs.log_of(sid)
    pred, rows = IdealPredictor("Ideal set"), []
    for _, t0 in eligible_episodes([sid]):
        for persona in cl.PERSONAS:
            for draw in draws:
                for delay_s in REGIME_DELAYS:
                    sc = rs.episode_scene(log)
                    w = CFWorld(sc, log, t0, persona, draw, DOSES[dose], int(round(delay_s / cl.DT)), True)
                    sc["world"] = w
                    t_end = min(sc["T"] - 1, t0 + cl.EP_LEN + 30)
                    n_fut = {a: {} for a in w.targets}
                    shift = {a: 0.0 for a in w.targets}
                    for g in range(t0, t_end + 1):
                        if g > t0:
                            w.step(g, None, None, 0.0)
                        for a, samples, *_ in pred.at(sc, g):
                            if a in n_fut:
                                n_fut[a][g] = len(samples)
                                if delay_s == 0.0 and g < t0 + cl.EP_LEN and len(samples) > 1:
                                    for b in w.posterior(a, g):
                                        if b != w.branch[a]:
                                            w.anchor = False
                                            f0 = w.branch_future(a, b, g)
                                            w.anchor = True
                                            dd = np.linalg.norm(w.branch_future(a, b, g) - f0, axis=1)
                                            if np.isfinite(dd).any():
                                                shift[a] = max(shift[a], float(np.nanmax(dd)))
                    for a in w.targets:
                        start = max(t0, w.t_div[a])
                        res = next((g for g in range(start, t_end + 1) if n_fut[a].get(g) == 1), None)
                        reps = [n_fut[a].get(g, 0) for g in range(t0, t0 + cl.EP_LEN, 2)]
                        rows.append({"scene": sid, "t0": t0, "persona": persona, "draw": draw, "agent": int(a),
                                     "branch": w.branch[a], "delay_s": delay_s, "branch_start": int(w.t_div[a]),
                                     "resolution_s": None if res is None else round((res - w.t_div[a]) * cl.DT, 6),
                                     "ambiguous_replans": sum(n >= 2 for n in reps), "replans": len(reps),
                                     "anchor_max_shift_m": shift[a] if delay_s == 0.0 else None})
    return rows


def regime_resolution(workers: int) -> dict:
    """The hard implementation gate: at every delay the median resolution time (censored = never
    within the window) is >= the delay-0 median + delay - 0.1 s; also the paired per-target shift.
    Dev scenes, strong dose, engineering draws 90-94."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    dose, draws = matched_dose(), REGIME_ENGINEERING_DRAWS["resolution"]
    scenes = rs._dev_scenes()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for part in ex.map(_resolution_task, [(s, draws, dose) for s in scenes]) for r in part]
    key = lambda r: (r["scene"], r["t0"], r["persona"], r["draw"], r["agent"])          # noqa: E731
    at0 = {key(r): r["resolution_s"] for r in rows if r["delay_s"] == 0.0}
    out = {}
    for d in REGIME_DELAYS:
        rr = [r for r in rows if r["delay_s"] == d]
        t = np.array([np.inf if r["resolution_s"] is None else r["resolution_s"] for r in rr])
        paired = [r["resolution_s"] - at0[key(r)] for r in rr if r["resolution_s"] is not None and at0[key(r)] is not None]
        out[f"{d:g}"] = {"targets": len(rr), "resolved_share": float(np.isfinite(t).mean()),
                         "median_resolution_s": float(np.median(t)),
                         "quartiles_s": [float(np.percentile(t, 25)), float(np.percentile(t, 75))],
                         "paired_shift_median_s": float(np.median(paired)) if paired else None,
                         "paired_shift_min_s": float(np.min(paired)) if paired else None,
                         "ambiguous_replan_share": float(sum(r["ambiguous_replans"] for r in rr) / sum(r["replans"] for r in rr)),
                         "by_branch_median_s": {b: float(np.median([np.inf if r["resolution_s"] is None else r["resolution_s"]
                                                                    for r in rr if r["branch"] == b])) for b in BRANCHES}}
    m0 = out["0"]["median_resolution_s"]
    gate = {f"{d:g}": bool(out[f"{d:g}"]["median_resolution_s"] >= m0 + d - 0.1) for d in REGIME_DELAYS}
    shifts = [r["anchor_max_shift_m"] for r in rows if r["anchor_max_shift_m"] is not None]
    payload = {"label": "engineering, log-only (no ego driving), before the pre-registration",
               "simulator_sha": _git("rev-parse", "HEAD"), "dirty": _git("status", "--porcelain", "--", "scripts") != "",
               "dose": dose, "scenes": "32 development scenes", "draws": list(draws), "delays_s": list(REGIME_DELAYS),
               "per_delay": out, "gate": gate, "passes": all(gate.values()),
               "anchor_shift_at_delay_0_m": {"median": float(np.median(shifts)), "p95": float(np.percentile(shifts, 95)),
                                             "max": float(np.max(shifts))},
               "rows": rows, "when": datetime.now().isoformat(timespec="seconds")}
    REGIME_RESOLUTION_JSON.write_text(json.dumps(payload), encoding="utf-8")
    for d, v in out.items():
        print(f"delay {d} s: median {v['median_resolution_s']:.2f} s, IQR {v['quartiles_s']}, resolved {v['resolved_share']:.1%}, "
              f"paired shift median {v['paired_shift_median_s']} min {v['paired_shift_min_s']}, ambiguous replans "
              f"{v['ambiguous_replan_share']:.1%}, by branch {v['by_branch_median_s']}")
    print("anchor shift at delay 0 (m):", payload["anchor_shift_at_delay_0_m"])
    print("gate:", gate, "PASS" if payload["passes"] else "FAIL", "| saved:", REGIME_RESOLUTION_JSON)
    return payload


# Throughput pilot (engineering) -----------------------------------------
def _peak_mb() -> float:
    """This process's peak working set in MB (Windows; ru_maxrss elsewhere)."""
    import sys
    if sys.platform != "win32":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                   [(f, ctypes.c_size_t) for f in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                   "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                                   "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    c = Counters()
    c.cb = ctypes.sizeof(Counters)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2 ** 20


def _measured(task, job) -> tuple:
    """(task(job), process CPU seconds, peak working set MB) in a pool worker."""
    import time
    c = time.process_time()
    r = task(job)
    return r, time.process_time() - c, _peak_mb()


def _pilot_task(job) -> tuple:
    return _measured(_lat_task, job)


def episodes_equal(a: list, b: list) -> bool:
    fields = ("scene", "t_start", "persona", "draw_seed", "score", "at_fault", "progress", "harsh_braking", "near_miss",
              "min_clearance_m", "jerk_rms", "n_replans")
    return len(a) == len(b) and all(tuple(x[f] for f in fields) == tuple(y[f] for f in fields) for x, y in zip(a, b))


def pilot_summary(jobs, res, per_cell: dict) -> dict:
    """Per (planner, commitment interval): episodes per CPU second and peak memory; the replan check."""
    out = {}
    for (planner, args), (r, cpu, mb) in zip(jobs, res):
        reg = args[4].get("regime")
        k = f"{planner} | commit {reg['commit_s']:g} s" if reg else f"{planner} | frozen"
        o = out.setdefault(k, {"episodes": 0, "cpu_s": 0.0, "peak_mb": 0.0, "replans_ok": True})
        o["episodes"] += len(r[3])
        o["cpu_s"] += cpu
        o["peak_mb"] = max(o["peak_mb"], mb)
        if reg:
            o["replans_ok"] &= all(e["n_replans"] == cl.EP_LEN // int(round(reg["commit_s"] / cl.DT)) for e in r[3])
        per_cell.setdefault(f"{regime_key((reg['delay_s'], reg['commit_s'])) if reg else 'frozen'} | {planner} | {args[0]}",
                            []).extend(r[3])
    for o in out.values():
        o["episodes_per_cpu_s"] = o["episodes"] / max(o["cpu_s"], 1e-9)
    return out


def regime_pilot(workers: int) -> dict:
    """Engineering run before the pre-registration (dev scenes, engineering draws 95-96): the
    public planner's throughput and peak memory per commitment interval; the oracle-invariance
    check (identical episodes at delay 0 and 2 s); the per-episode replan count; the identity
    check (no regime: Part 1b's public episodes, primary draws 20-21 on 6 scenes); and how much
    the regime map's world (anchoring, exact commitment, hold-aware proposals) changes the
    frozen baseline cell (descriptive)."""
    import json
    import time
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    dose, base = matched_dose(), frozen_proposal()
    _, lam = rs._v2_cfg()
    scenes, draws = rs._dev_scenes(), REGIME_ENGINEERING_DRAWS["pilot"]
    jobs = []
    for c in REGIME_COMMITS:
        for n in ("Ideal single (mean)", "Ideal set"):
            jobs += [("proposal", (n, 0, scenes, lam, regime_cfg(base, (REGIME_DELAYS[-1], c)), None, dose, d, False))
                     for d in draws]
        for dl in (REGIME_DELAYS[0], REGIME_DELAYS[-1]):
            jobs += [("proposal", ("Branch oracle", 0, scenes, lam, regime_cfg(base, (dl, c)), None, dose, d, False))
                     for d in draws]
    jobs += [("proposal", ("Ideal single (mean)", 0, scenes, lam, cfg, None, dose, d, False))
             for cfg in (base, regime_cfg(base, (0.0, 0.2))) for d in draws]
    id_scenes = primary_scenes()[:6]
    id_jobs = [("proposal", ("Ideal single (mean)", 0, id_scenes, lam, base, None, dose, d, False)) for d in (20, 21)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        t = time.perf_counter()
        res = list(ex.map(_pilot_task, jobs))
        wall = time.perf_counter() - t
        id_res = list(ex.map(_lat_task, id_jobs))
    cells = {}
    thr = pilot_summary(jobs, res, cells)
    inv = {f"commit {c:g} s": episodes_equal(cells[f"{regime_key((REGIME_DELAYS[0], c))} | proposal | Branch oracle"],
                                             cells[f"{regime_key((REGIME_DELAYS[-1], c))} | proposal | Branch oracle"])
           for c in REGIME_COMMITS}
    pub = json.loads(CONFIRM_1B_JSON.read_text(encoding="utf-8"))["episodes"]["proposal | Ideal single (mean)"]
    ref = {tuple(e[:4]): (e[5], e[6]) for e in pub if e[0] in id_scenes and e[3] in (20, 21)}
    mine = {(e["scene"], e["t_start"], e["persona"], e["draw_seed"]): (round(e["score"], 6), bool(e["at_fault"]))
            for r in id_res for e in r[3]}
    frozen, base_cell = cells["frozen | proposal | Ideal single (mean)"], \
        cells[f"{regime_key((0.0, 0.2))} | proposal | Ideal single (mean)"]
    same = [x["score"] == y["score"] and x["at_fault"] == y["at_fault"] for x, y in zip(frozen, base_cell)]
    payload = {"label": "engineering, before the pre-registration", "simulator_sha": _git("rev-parse", "HEAD"),
               "dirty": _git("status", "--porcelain", "--", "scripts") != "", "dose": dose,
               "scenes": "32 development scenes", "draws": list(draws), "workers": workers, "pool_wall_s": wall,
               "throughput": thr, "oracle_invariant_across_delay": inv,
               "identity_vs_part_1b": {"episodes": len(ref), "identical": mine == ref},
               "baseline_cell_vs_frozen": {"episodes": len(same), "identical_share": float(np.mean(same)),
                                           "frozen": summary(frozen), "regime_baseline": summary(base_cell)},
               "cells": {k: summary(v) for k, v in cells.items()}, "when": datetime.now().isoformat(timespec="seconds")}
    REGIME_PILOT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for k, v in thr.items():
        print(f"{k:32s} {v['episodes']:5d} episodes  {v['episodes_per_cpu_s']:.2f} per CPU s  peak {v['peak_mb']:.0f} MB  "
              f"replans ok {v['replans_ok']}")
    print("oracle invariant across delay:", inv, "| identity vs Part 1b:", payload["identity_vs_part_1b"])
    print("baseline cell vs frozen:", {k: payload["baseline_cell_vs_frozen"][k] for k in ("episodes", "identical_share")},
          "| saved:", REGIME_PILOT_JSON)
    return payload


def _anchor_pred_task(job) -> list:
    """One dev scene, no ego driving: at every replan of every eligible episode (delay 0), each
    ideal predictor's output for every target, anchored and unanchored. Per (predictor, target,
    replan): live branches, whether the most likely branch is the realised one, the largest
    output shift, the largest in the first second, and whether the set of finite branch futures
    differs (a future running past the end of the log)."""
    sid, draws, dose = job
    log = rs.log_of(sid)
    preds = {n: IdealPredictor(n) for n in ANCHOR_PREDICTORS}
    rows = []
    for _, t0 in eligible_episodes([sid]):
        for persona in cl.PERSONAS:
            for draw in draws:
                sc = rs.episode_scene(log)
                w = CFWorld(sc, log, t0, persona, draw, DOSES[dose], 0, True)
                sc["world"] = w
                for g in range(t0, t0 + cl.EP_LEN):
                    if g > t0:
                        w.step(g, None, None, 0.0)
                    if (g - t0) % REPLAN_FROZEN:
                        continue
                    out = {}
                    for anchor in (True, False):
                        w.anchor = anchor
                        out[anchor] = {n: {a: s for a, s, *_ in p.at(sc, g) if a in w.branch} for n, p in preds.items()}
                    w.anchor = True
                    for a in out[True]["Ideal set"]:
                        post = w.posterior(a, g)
                        most = max(post, key=post.get) == w.branch[a]
                        fin = bool((np.isfinite(out[True]["Ideal set"][a][..., 0])
                                    != np.isfinite(out[False]["Ideal set"][a][..., 0])).any())
                        for n in preds:
                            d = np.linalg.norm(out[True][n][a] - out[False][n][a], axis=-1)
                            mx = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
                            m1 = float(np.nanmax(d[:, :10])) if np.isfinite(d[:, :10]).any() else 0.0
                            rows.append((n, len(post), most, mx, m1, fin))
    return rows


ANCHOR_PREDICTORS = ("Branch oracle", "Ideal single (mean)", "Ideal single (most likely)", "Ideal set")


def regime_anchor_check(workers: int) -> dict:
    """Engineering check before the pre-registration (dev scenes, the pilot's engineering draws
    95-96): what the regime map's baseline cell (delay 0, commit 0.2 s) changes against the
    frozen world, and why. (1) Log-only: each ideal predictor's output, anchored vs unanchored.
    (2) Closed loop, proposal planner at margin 0: anchoring, exact commitment and the hold-aware
    proposals in each of the 8 on/off combinations; all off must equal the frozen simulator and
    all on the map's own regime path."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    from itertools import product
    dose, base = matched_dose(), frozen_proposal()
    _, lam = rs._v2_cfg()
    scenes, draws = rs._dev_scenes(), REGIME_ENGINEERING_DRAWS["pilot"]
    names = ("Branch oracle", "Ideal single (mean)", "Ideal set")
    variants = {"frozen (no regime)": base, "regime path": regime_cfg(base, (0.0, 0.2))}
    for on in product((False, True), repeat=3):
        variants["+".join(k for k, o in zip(("anchor", "exact", "hold"), on) if o) or "none"] = \
            {**base, "margin_m": 0.0, "regime": {"delay_s": 0.0, "commit_s": 0.2, **dict(zip(("anchor", "exact", "hold"), on))}}
    jobs, tags = [], []
    for v, cfg in variants.items():
        for n in names:
            for d in draws:
                for i in range(4):
                    jobs.append(("proposal", (n, 0, scenes[i::4], lam, cfg, None, dose, d, False)))
                    tags.append((v, n))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for part in ex.map(_anchor_pred_task, [(s, draws, dose) for s in scenes]) for r in part]
        res = list(ex.map(_lat_task, jobs))
    pred = {}
    for n in ANCHOR_PREDICTORS:
        rr = [r for r in rows if r[0] == n]
        amb = [r for r in rr if r[1] > 1]
        o = {"target_replans": len(rr), "ambiguous": len(amb),
             "changed_with_one_branch": int(sum(r[3] > 0 for r in rr if r[1] == 1)),
             "changed_share_ambiguous": float(np.mean([r[3] > 0 for r in amb])),
             "p95_shift_ambiguous_m": float(np.percentile([r[3] for r in amb], 95)),
             "max_shift_m": float(max(r[3] for r in rr)), "max_shift_first_1s_m": float(max(r[4] for r in rr))}
        if n == "Ideal single (most likely)":
            o["changed_when_most_likely_is_realised"] = int(sum(r[3] > 0 for r in rr if r[2]))
        pred[n] = o
    set_max = pred["Ideal set"]["max_shift_m"]
    mean = [r for r in rows if r[0] == "Ideal single (mean)" and r[1] > 1]
    pred["Ideal single (mean)"].update({
        "jump": "a shift larger than the Ideal set's largest shift of any branch future",
        "jump_share_ambiguous": float(np.mean([r[3] > set_max for r in mean])),
        "jumps_without_a_finite_futures_change": int(sum(r[3] > set_max and not r[5] for r in mean)),
        "finite_futures_change_share_ambiguous": float(np.mean([r[5] for r in mean])),
        "max_shift_without_a_finite_futures_change_m": float(max([r[3] for r in mean if not r[5]], default=0.0))})
    eps = {}
    for t, r in zip(tags, res):
        eps.setdefault(t, []).extend(r[3])
    for k in eps:
        eps[k].sort(key=lambda e: (e["scene"], e["t_start"], e["persona"], e["draw_seed"]))
    loop, checks = {}, {}
    for n in names:
        fr = eps[("frozen (no regime)", n)]
        loop[n] = {v: {"identical_share": float(np.mean([episodes_equal([x], [y]) for x, y in zip(fr, eps[(v, n)])])),
                       "episodes": len(eps[(v, n)]), "at_fault_episodes": int(sum(e["at_fault"] for e in eps[(v, n)])),
                       **{k: summary(eps[(v, n)])[k] for k in ("score", "harsh_braking_rate", "near_miss_rate")}}
                   for v in variants}
        checks[n] = {"all off == frozen": episodes_equal(fr, eps[("none", n)]),
                     "all on == regime path": episodes_equal(eps[("anchor+exact+hold", n)], eps[("regime path", n)])}
    checks["anchoring leaves the oracle's episodes unchanged"] = episodes_equal(eps[("none", "Branch oracle")],
                                                                              eps[("anchor", "Branch oracle")])
    checks["the oracle's prediction never changes"] = pred["Branch oracle"]["max_shift_m"] == 0.0
    payload = {"label": "engineering, before the pre-registration", "simulator_sha": _git("rev-parse", "HEAD"),
               "dirty": _git("status", "--porcelain", "--", "scripts") != "", "dose": dose,
               "scenes": "32 development scenes", "draws": list(draws), "workers": workers,
               "identical_share": "share of episodes equal to the frozen simulator on every logged field",
               "prediction_level": pred, "closed_loop": loop, "checks": checks,
               "when": datetime.now().isoformat(timespec="seconds")}
    REGIME_ANCHOR_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for n, v in pred.items():
        print(f"{n:28s} ambiguous {v['ambiguous']:5d}/{v['target_replans']}  changed {v['changed_share_ambiguous']:.1%}  "
              f"p95 {v['p95_shift_ambiguous_m']:.2f} m  max {v['max_shift_m']:.2f} m (first 1 s {v['max_shift_first_1s_m']:.2f})")
    for n, by in loop.items():
        for v, x in by.items():
            print(f"{n:22s} {v:20s} identical {x['identical_share']:.1%}  score {x['score']:.4f}  "
                  f"at-fault {x['at_fault_episodes']}")
    print("checks:", checks, "| saved:", REGIME_ANCHOR_JSON)
    return payload


# Statistics --------------------------------------------------------------
def weighted_contrast(cells: dict, weights: dict, metric: str, scenes: list, counts, signs: np.ndarray) -> dict:
    """sum_c w_c x rate_c over cells sharing their episodes (contrast_test with any weights);
    positive = better. p: one-sided scene sign-flip randomisation test (exact when each
    scene's contrast is symmetric about 0 under the null). CI: scene-cluster bootstrap
    percentile (counts None: no CI)."""
    names = list(weights)
    X, N, sign = _cell_arrays(cells, names, metric, scenes)
    w = np.array([weights[c] for c in names], float)[:, None]
    c, n = sign * (w * X).sum(0), N[0]
    est = float(c.sum() / n.sum())
    flipped = (signs @ c) / n.sum()
    out = {"gain": est, "p": float((np.sum(flipped >= est) + 1) / (len(flipped) + 1))}
    if counts is not None:
        boot = (counts @ c) / (counts @ n)
        out["ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return out


def _ranks(r: tuple) -> tuple:
    return REGIME_DELAYS.index(r[0]), REGIME_COMMITS.index(r[1])


def regime_select(cells: dict) -> dict:
    """Per regime, the best single and the best set cell (dev): lowest at-fault rate, then highest score."""
    key = lambda c: (-cells[c]["at_fault_rate"], cells[c]["score"])                 # noqa: E731
    return {regime_key(r): {"single": max((f"{regime_key(r)} | {c}" for c in REGIME_SINGLES), key=key),
                            "set": max((f"{regime_key(r)} | {c}" for c in REGIME_SETS), key=key)} for r in REGIMES}


def trend_weights(choice: dict) -> dict:
    """P1: sum over regimes of c_ij x (best set - best single), c_ij = row rank + column rank - 2."""
    w = {}
    for r in REGIMES:
        c = float(sum(_ranks(r)) - 2)
        if c:
            w[choice[regime_key(r)]["set"]], w[choice[regime_key(r)]["single"]] = c, -c
    return w


def interaction_weights(choice: dict) -> dict:
    """P3: (G_ss - G_sf) - (G_fs - G_ff), G = best set - best single; first letter the
    resolution (s = delay 2 s), second the reaction (s = commit 1 s)."""
    (f_d, s_d), (f_c, s_c) = (REGIME_DELAYS[0], REGIME_DELAYS[-1]), (REGIME_COMMITS[0], REGIME_COMMITS[-1])
    w = {}
    for r, s in (((s_d, s_c), 1.0), ((s_d, f_c), -1.0), ((f_d, s_c), -1.0), ((f_d, f_c), 1.0)):
        w[choice[regime_key(r)]["set"]], w[choice[regime_key(r)]["single"]] = s, -s
    return w


def regime_tests(cells: dict, choice: dict, seed: int = 0, flips: int = REGIME_FLIPS, ci: bool = True) -> dict:
    """{metric: [P1, P2, P3]}. P1 (additive trend, the cells fixed on dev) and P2 (adverse corner:
    best set vs best single, max_vs_max over all the corner's cells) form the family, Holm over
    2 per metric (at-fault primary, score secondary); P3 (interaction) is secondary, unadjusted."""
    first = cells[choice[regime_key(REGIME_CORNER)]["single"]]["per_scene"]
    scenes = sorted(first)
    assert all(cells[c]["per_scene"][s][0] == first[s][0] for c in cells for s in scenes), "cells must share their episodes"
    counts, signs = boot_counts(len(scenes), seed, None if ci else 1), flip_signs(len(scenes), seed, flips)
    corner = regime_key(REGIME_CORNER)
    out = {}
    for m in ("at_fault", "score"):
        fam = [{"test": "P1", "metric": m, **weighted_contrast(cells, trend_weights(choice), m, scenes, counts if ci else None, signs)},
               {"test": "P2", "metric": m, **max_vs_max(cells, [f"{corner} | {c}" for c in REGIME_SETS],
                                                         [f"{corner} | {c}" for c in REGIME_SINGLES], m, scenes, counts, signs)}]
        holm(fam)
        fam.append({"test": "P3", "metric": m, "secondary": True,
                    **weighted_contrast(cells, interaction_weights(choice), m, scenes, counts if ci else None, signs)})
        out[m] = fam
    return out


def regime_reading(tests: list, power_ok: bool, p3_power_ok: bool, corner_headroom: float | None,
                   calibrated: dict | None = None) -> str:
    """The pre-registered reading of the at-fault tests; a test that failed its calibration pass 2
    criterion (calibrated[test] False) cannot support a positive reading."""
    ok = calibrated or {}
    sig = {t["test"]: t["p_holm"] < REGIME_ALPHA and ok.get(t["test"], True) for t in tests if t["test"] in ("P1", "P2")}
    p3 = next(t for t in tests if t["test"] == "P3")["p"] < REGIME_ALPHA and ok.get("P3", True)
    if sig["P1"] and sig["P2"]:
        return "regime-dependent benefit; " + ("the factors interact (secondary)" if p3 else
                                               "additive" if p3_power_ok else "interaction inconclusive")
    if sig["P2"]:
        return "benefit in the adverse corner, no demonstrated trend"
    if sig["P1"]:
        return "trend without a demonstrated benefit"
    if corner_headroom is not None and corner_headroom < REGIME_EFFECT:
        return "no headroom"
    return "no regime-dependent benefit" if power_ok else "inconclusive"


def clopper_pearson(k: int, n: int) -> list:
    """Exact 95% binomial interval."""
    from scipy.stats import beta
    return [0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))]


# Calibration of the whole pipeline ----------------------------------------
def _structure(cells: dict, names: list, base_cell: int | None = 1) -> dict:
    """Per scene: the episode count, the base rates (cell base_cell, or the scene average over
    the cells) and every cell's deviation from the scene average (each cell's overall mean
    removed): the synthetic data's ingredients."""
    scenes = sorted(cells[names[0]]["per_scene"])
    n = np.array([cells[names[0]]["per_scene"][s][0] for s in scenes], float)
    m = {k: np.array([[cells[c]["per_scene"][s][col] for s in scenes] for c in names], float) / n
         for k, col in (("score", 1), ("at_fault", 2))}
    dev = {k: (v - v.mean(0)) - ((v - v.mean(0)) * n).sum(1, keepdims=True) / n.sum() for k, v in m.items()}
    return {"scenes": scenes, "n": n, "base": {k: m[k][base_cell] if base_cell is not None else m[k].mean(0) for k in m},
            "dev": dev}


def regime_structure_pass1() -> dict:
    """Pass 1: already-reported cells only. Primary: the five public cells of section 11 (draws
    0-9, base = the proposal single); dev: the proposal planner's cells of section 9 (draws
    10-19). The same structure in every regime."""
    import json
    pc = json.loads(CONFIRM_JSON.read_text(encoding="utf-8"))["cells"]
    dc = json.loads(MATCHED_JSON.read_text(encoding="utf-8"))["cells"]["proposal"]
    prim = _structure(pc, ["proposal | Branch oracle", "proposal | Ideal single (mean)", "proposal | Ideal single (most likely)",
                           "proposal | Ideal set", "proposal+margin | Ideal single (mean)"])
    dev = _structure(dc, ["Branch oracle", "Ideal single (mean)", "Ideal single (most likely)", "Ideal set"])
    return {"source": [CONFIRM_JSON.name, MATCHED_JSON.name], "real_draws": {"primary": 10, "dev": 10},
            "primary": {**prim, "regimes": {regime_key(r): {"base": prim["base"], "dev": prim["dev"]} for r in REGIMES}},
            "dev": {**dev, "regimes": {regime_key(r): {"base": dev["base"], "dev": dev["dev"]} for r in REGIMES}}}


def regime_structure_pass2(dev_cells: dict) -> dict:
    """Pass 2: the regime map's own dev cells (draws 60-64, no primary data). Dev: per regime its
    cells' rates and deviations. Primary: pass 1's primary structure with each regime's base
    rate scaled (at-fault) or shifted (score) as on dev against the baseline regime, and its
    deviations scaled by the regime's dev deviation spread against the baseline's."""
    p1 = regime_structure_pass1()
    cells = list(REGIME_SINGLES + REGIME_SETS)
    dev = {regime_key(r): _structure(dev_cells, [f"{regime_key(r)} | {c}" for c in cells], None) for r in REGIMES}
    b = dev[regime_key((REGIME_DELAYS[0], REGIME_COMMITS[0]))]
    prim = p1["primary"]
    regs = {}
    for r in REGIMES:
        d = dev[regime_key(r)]
        rate = lambda s, k: float((s["base"][k] * s["n"]).sum() / s["n"].sum())          # noqa: E731
        spread = lambda s, k: float(np.sqrt(np.mean(s["dev"][k] ** 2)))                 # noqa: E731
        af = (prim["base"]["at_fault"] * rate(d, "at_fault") / rate(b, "at_fault") if rate(b, "at_fault") > 0 else
              prim["base"]["at_fault"] + rate(d, "at_fault"))           # no baseline events: shift instead of scale
        regs[regime_key(r)] = {
            "base": {"at_fault": af, "score": prim["base"]["score"] + rate(d, "score") - rate(b, "score")},
            "dev": {k: prim["dev"][k] * (spread(d, k) / spread(b, k) if spread(b, k) > 0 else 1.0)
                    for k in ("at_fault", "score")}}
    any_dev = next(iter(dev.values()))
    return {"source": ["regime dev cells", CONFIRM_JSON.name], "real_draws": {"primary": 10, "dev": len(REGIME_DEV_DRAWS)},
            "primary": {**prim, "regimes": regs},
            "dev": {"scenes": any_dev["scenes"], "n": any_dev["n"],
                    "regimes": {k: {"base": v["base"], "dev": v["dev"]} for k, v in dev.items()}}}


def _spec_effect(spec: dict, r: tuple, c: str) -> tuple:
    """(score, at-fault) shift of cell c in regime r: the set advantage and the composite null's worse singles."""
    i, j = _ranks(r)
    e = spec.get("effect_pp", 0.0) / 100
    adv = {"none": 0.0, "constant": e, "additive": e * (i + j) / 4, "interaction": e * (i / 2) * (j / 2)}[spec["effect"]]
    if c in REGIME_SETS:
        return 0.0, -adv
    if spec.get("worse_singles_pp") and c != "contingency | Ideal single (mean)":
        return 0.0, spec["worse_singles_pp"] / 100
    return 0.0, 0.0


def _spec_noise(spec: dict, r: tuple, c: str) -> float:
    if not spec.get("unequal"):
        return 1.0
    return (1.0 + 0.25 * sum(_ranks(r))) * (3.0 if c in ("proposal+margin | Ideal single (mean)", "contingency | Ideal set") else 1.0)


def _synthetic(part: dict, spec: dict, rng, draws: int, real_draws: int, noise_mult: float) -> dict:
    """Every regime cell over one scene set, all sharing its episodes: per scene the regime's base
    plus the cell's shift plus one of the regime's real deviations with a random sign (an exact
    null), its spread scaled to the number of draws."""
    S = len(part["scenes"])
    n, sd = part["n"] * draws / real_draws, np.sqrt(real_draws / draws)
    cells = {}
    for r in REGIMES:
        g = part["regimes"][regime_key(r)]
        K = g["dev"]["score"].shape[0]
        for c in REGIME_SINGLES + REGIME_SETS:
            f = noise_mult * _spec_noise(spec, r, c) * sd
            pick, sgn = rng.integers(0, K, S), rng.choice((-1.0, 1.0), S)
            ds, da = _spec_effect(spec, r, c)
            sc = g["base"]["score"] + ds + f * sgn * g["dev"]["score"][pick, np.arange(S)]
            af = g["base"]["at_fault"] + da + f * sgn * g["dev"]["at_fault"][pick, np.arange(S)]
            ps = {s: [n[j], n[j] * sc[j], n[j] * af[j]] for j, s in enumerate(part["scenes"])}
            cells[f"{regime_key(r)} | {c}"] = {"per_scene": ps, "score": float((n * sc).sum() / n.sum()),
                                               "at_fault_rate": float((n * af).sum() / n.sum())}
    return cells


def _calib_chunk(job) -> list:
    """Datasets lo..hi-1 of one scenario through the whole pipeline: synthetic dev data, the dev
    selection, synthetic primary data, P1-P3 with Holm, the reading."""
    struct, spec, lo, hi = job
    out = []
    for i in range(lo, hi):
        rng = np.random.default_rng([spec["id"], i])
        choice = regime_select(_synthetic(struct["dev"], spec, rng, len(REGIME_DEV_DRAWS), struct["real_draws"]["dev"],
                                          spec.get("dev_noise", 1.0)))
        prim = _synthetic(struct["primary"], spec, rng, spec["draws"], struct["real_draws"]["primary"], 1.0)
        fam = regime_tests(prim, choice, seed=i, flips=REGIME_CALIB_FLIPS, ci=False)
        out.append({"tests": {m: {t["test"]: [t["p"], t.get("p_holm")] for t in ts} for m, ts in fam.items()},
                    "reading": regime_reading(fam["at_fault"], True, True, None)})
    return out


REGIME_SCENARIOS = (   # nulls: 5,000 datasets; power: 1,000 per draw count
    {"name": "null, equal variances", "effect": "none", "kind": "null"},
    {"name": "null, unequal variances", "effect": "none", "unequal": True, "kind": "null"},
    {"name": "null, selection noise (dev 3x noisier), unequal variances", "effect": "none", "unequal": True,
     "dev_noise": 3.0, "kind": "null"},
    {"name": "composite null, other singles +0.2 pt at-fault, unequal variances", "effect": "none", "unequal": True,
     "worse_singles_pp": 0.2, "kind": "null"},
    {"name": "no trend, constant set advantage 1.0 pt, unequal variances", "effect": "constant", "effect_pp": 1.0,
     "unequal": True, "kind": "constant"},
    {"name": "power, additive trend 0 -> 1.0 pt", "effect": "additive", "effect_pp": 1.0, "kind": "power"},
    {"name": "power, additive trend 0 -> 1.0 pt, unequal variances", "effect": "additive", "effect_pp": 1.0,
     "unequal": True, "kind": "power"},
    {"name": "power, interaction (both slow) 1.0 pt", "effect": "interaction", "effect_pp": 1.0, "kind": "power"})


def _calib_rates(results: list) -> dict:
    """Rejection rates with Clopper-Pearson intervals: P1/P2 Holm (the family), P3 raw, the reading."""
    n = len(results)
    rate = lambda k: {"rate": k / n, "ci95": clopper_pearson(k, n)}                  # noqa: E731
    out = {}
    for m in ("at_fault", "score"):
        t = [r["tests"][m] for r in results]
        out[m] = {"P1": rate(sum(x["P1"][1] < REGIME_ALPHA for x in t)), "P2": rate(sum(x["P2"][1] < REGIME_ALPHA for x in t)),
                  "P3": rate(sum(x["P3"][0] < REGIME_ALPHA for x in t)),
                  "P1_and_P2": rate(sum(x["P1"][1] < REGIME_ALPHA and x["P2"][1] < REGIME_ALPHA for x in t))}
    out["regime_dependent_reading"] = rate(sum(r["reading"].startswith("regime-dependent") for r in results))
    return out


def regime_calibration(workers: int, structure: dict | None = None, n_null: int = 5000, n_power: int = 1000,
                       draw_counts=(10, 20, 30), out_path=None) -> dict:
    """The whole pipeline on synthetic data. Nulls (5,000 datasets each): the upper 95% bound of
    every rejection rate (P1, P2 Holm; P3; the "regime-dependent benefit" reading) must be <=
    0.06; with a constant set advantage P1, P3 and the reading must be, P2's rate is power.
    Power (1,000 datasets per draw count): P1, P2, both, and P3 for the interaction. Also a
    numpy / CUDA benchmark of the sign-flip products (adopted only if faster and identical)."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    struct = structure or regime_structure_pass1()
    chunk = 250
    jobs, meta = [], []
    for sid, s in enumerate(REGIME_SCENARIOS):
        counts = (10,) if s["kind"] != "power" else draw_counts
        for D in counts:
            spec = {**s, "id": 1000 * sid + D, "draws": D}
            n = n_null if s["kind"] != "power" else n_power
            meta.append((s["name"], D, s["kind"]))
            jobs += [((struct, spec, lo, min(lo + chunk, n)), len(meta) - 1) for lo in range(0, n, chunk)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_calib_chunk, [j for j, _ in jobs]))
    results = {}
    for (_, k), p in zip(jobs, parts):
        results.setdefault(k, []).extend(p)
    scen = {}
    for k, (name, D, kind) in enumerate(meta):
        scen.setdefault(name, {"kind": kind})[f"{D} draws"] = {"datasets": len(results[k]), **_calib_rates(results[k])}
    ub = lambda x: x["ci95"][1]                                                       # noqa: E731
    crit = {"null_upper_bounds_le_0.06": {}, "power": {}}
    for name, v in scen.items():
        r = v["10 draws"]
        if v["kind"] == "null":
            crit["null_upper_bounds_le_0.06"][name] = {t: ub(r["at_fault"][t]) for t in ("P1", "P2", "P3")} | \
                {"reading": ub(r["regime_dependent_reading"])}
        elif v["kind"] == "constant":
            crit["null_upper_bounds_le_0.06"][name] = {"P1": ub(r["at_fault"]["P1"]), "P3": ub(r["at_fault"]["P3"]),
                                                      "reading": ub(r["regime_dependent_reading"])}
            crit["constant_advantage_P2_rate"] = r["at_fault"]["P2"]
        else:
            crit["power"][name] = {f"{D}": {t: v[f"{D} draws"]["at_fault"][t]["rate"] for t in ("P1", "P2", "P1_and_P2", "P3")}
                                   for D in draw_counts}
    crit["all_nulls_pass"] = all(x <= REGIME_FPR for v in crit["null_upper_bounds_le_0.06"].values() for x in v.values())
    crit["calibrated"] = {t: all(v[t] <= REGIME_FPR for v in crit["null_upper_bounds_le_0.06"].values() if t in v)
                          for t in ("P1", "P2", "P3", "reading")}
    out = {"structure_source": struct["source"], "datasets": {"null": n_null, "power": n_power},
           "flips_per_test": REGIME_CALIB_FLIPS, "pipeline": "dev selection -> P1 / P2 (Holm) / P3 -> reading",
           "scenarios": scen, "criteria": crit, "gpu_benchmark": _signflip_benchmark(struct),
           "when": datetime.now().isoformat(timespec="seconds")}
    path = out_path or REGIME_CALIBRATION_JSON
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, v in scen.items():
        for D, r in v.items():
            if D != "kind":
                print(f"{name} [{D}]: " + "  ".join(f"{t} {r['at_fault'][t]['rate']:.4f} [{r['at_fault'][t]['ci95'][0]:.4f}, "
                                                    f"{r['at_fault'][t]['ci95'][1]:.4f}]" for t in ("P1", "P2", "P3", "P1_and_P2"))
                      + f"  reading {r['regime_dependent_reading']['rate']:.4f}", flush=True)
    print("all nulls pass:", crit["all_nulls_pass"], "| saved:", path)
    return out


def _signflip_benchmark(struct: dict, reps: int = 200) -> dict:
    """numpy vs torch-CUDA for the calibration's sign-flip products (flips x scenes) @ (scenes x
    cells), and what that is worth end to end: the products' share of a dataset's time bounds
    the whole pipeline's speed-up. Adopted only if identical and the bound is >= 2x (the rule of
    the simulator's GPU benchmark)."""
    import time
    rng = np.random.default_rng(0)
    S = len(struct["primary"]["scenes"])
    signs, D = flip_signs(S, 0, REGIME_CALIB_FLIPS), rng.normal(size=(reps, S, 9))
    t = time.perf_counter()
    a = np.stack([signs @ D[i] for i in range(reps)])
    t_np = time.perf_counter() - t
    spec = {**REGIME_SCENARIOS[1], "id": 0, "draws": 10}
    t = time.perf_counter()
    _calib_chunk((struct, spec, 0, 20))
    t_dataset = (time.perf_counter() - t) / 20
    share = min(1.0, 6 * (t_np / reps) / t_dataset)          # 3 products x 2 metrics per dataset
    end = {"dataset_s": t_dataset, "products_share": share}
    try:
        import torch
        if not torch.cuda.is_available():
            return {"numpy_s": t_np, **end, "cuda": "unavailable", "adopt": False}
        sg, Dg = torch.tensor(signs, device="cuda", dtype=torch.float64), torch.tensor(D, device="cuda", dtype=torch.float64)
        (sg @ Dg[0]).sum().item()
        torch.cuda.synchronize()
        t = time.perf_counter()
        b = torch.stack([sg @ Dg[i] for i in range(reps)]).cpu().numpy()
        t_gpu = time.perf_counter() - t
    except Exception as e:                                                            # noqa: BLE001
        return {"numpy_s": t_np, **end, "cuda": f"failed: {e}", "adopt": False}
    same = bool(np.array_equal(a, b))
    bound = 1.0 / (1.0 - share * (1.0 - t_gpu / t_np))
    return {"numpy_s": t_np, "cuda_s": t_gpu, "products_speedup": t_np / t_gpu, "identical": same,
            "max_abs_diff": float(np.abs(a - b).max()), **end, "end_to_end_speedup_bound": bound,
            "adopt": bool(same and bound >= 2.0)}


# Tuning and evaluation runners (after the pre-registration) --------------
def check_replans(jobs, res) -> None:
    """Every regime episode replanned exactly every commitment interval (the setter took effect)."""
    for (_, a), r in zip(jobs, res):
        reg = a[4].get("regime")
        if reg:
            k = int(round(reg["commit_s"] / cl.DT))
            assert all(e["n_replans"] == cl.EP_LEN // k for e in r[3]), f"replan count off for {reg}"


def tune_margins(ex, task, planner: str, label: str, preds, base: dict, scenes, lam, dose: str, weighted: bool) -> dict:
    """Dev draws 60-64, every regime and predictor: the margin grid (lowest at-fault rate, then
    highest score; extended by REGIME_MARGINS_EDGE where the grid's top wins) and the dev cells
    "<regime> | label | predictor" (margin 0; not for the most-likely single) and
    "<regime> | label+margin | predictor" (tuned)."""
    def run(margins, which):
        jobs = [(planner, (n, 0, scenes, lam, regime_cfg(base, r, m), None, dose, d, weighted))
                for r, n in sorted(which) for m in margins for d in REGIME_DEV_DRAWS]
        res = list(ex.map(task, jobs))
        check_replans(jobs, res)
        eps = {}
        for (_, a), x in zip(jobs, res):
            reg = a[4]["regime"]
            eps.setdefault(((reg["delay_s"], reg["commit_s"]), a[0], a[4]["margin_m"]), []).extend(x[3])
        return {k: (v, summary(v)) for k, v in eps.items()}
    pairs = {(r, n) for r in REGIMES for n in preds}
    eps = run(REGIME_MARGINS, pairs)
    choose = lambda r, n: max((m for rr, nn, m in eps if (rr, nn) == (r, n)),                          # noqa: E731
                              key=lambda m: (-eps[(r, n, m)][1]["at_fault_rate"], eps[(r, n, m)][1]["score"]))
    edge = {(r, n) for r, n in pairs if choose(r, n) == REGIME_MARGINS[-1]}
    if edge:
        eps.update(run(REGIME_MARGINS_EDGE, edge))
    rows, chosen, cells = {}, {}, {}
    for r, n in sorted(pairs):
        k = regime_key(r)
        rows.setdefault(k, {})[n] = [{"margin_m": m, "summary": eps[(r, n, m)][1]}
                                     for m in sorted(m for rr, nn, m in eps if (rr, nn) == (r, n))]
        chosen.setdefault(k, {})[n] = m = choose(r, n)
        cells[f"{k} | {label}+margin | {n}"] = {**eps[(r, n, m)][1], "per_scene": per_scene(eps[(r, n, m)][0]), "margin_m": m}
        if n != "Ideal single (most likely)":
            cells[f"{k} | {label} | {n}"] = {**eps[(r, n, 0.0)][1], "per_scene": per_scene(eps[(r, n, 0.0)][0]), "margin_m": 0.0}
    return {"margin_rows": rows, "chosen_margin_m": chosen, "cells": cells,
            "edge_extended": sorted(f"{regime_key(r)} | {n}" for r, n in edge)}


def regime_tune(workers: int) -> None:
    """The public dev runs: the proposal planner's margins for the Ideal single (mean), the
    Ideal single (most likely) and the Ideal set in every regime, and the public dev cells. The
    confidential runner adds its cells and makes the dev selection (regime_select)."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    ref, dose = preregistered(), matched_dose()
    assert_draws_free("dev", REGIME_DEV_DRAWS)
    _, lam = rs._v2_cfg()
    base, scenes = frozen_proposal(), rs._dev_scenes()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        t = tune_margins(ex, _lat_task, "proposal", "proposal",
                         ("Ideal single (mean)", "Ideal single (most likely)", "Ideal set"), base, scenes, lam, dose, False)
    payload = {**ref, **repro({"proposal": base, "regimes": REGIMES, "margins": REGIME_MARGINS}, scenes), "dose": dose,
               "scenes": "32 development scenes", "draws": list(REGIME_DEV_DRAWS), "planner_cfg": base, **t,
               "when": datetime.now().isoformat(timespec="seconds")}
    REGIME_TUNE_JSON.write_text(json.dumps(payload), encoding="utf-8")
    for k, v in t["chosen_margin_m"].items():
        print(f"{k}: chosen margins {v}")
    print("edge-extended:", t["edge_extended"], "| saved:", REGIME_TUNE_JSON)


def regime_eval_draws() -> tuple:
    """The evaluation draws fixed by the committed sample-size addendum."""
    import json
    rel = str(REGIME_SAMPLE_JSON.relative_to(rs.REPO)).replace("\\", "/")
    assert _git("log", "-1", "--format=%H", "--", rel) and _git("status", "--porcelain", "--", rel) == "", \
        "commit the sample-size addendum first"
    return REGIME_EVAL_DRAWS[json.loads(REGIME_SAMPLE_JSON.read_text(encoding="utf-8"))["draws"]]


def regime_cells(jobs, res) -> tuple:
    """Cells "<regime> | planner label | predictor" pooled over draws, and their episode records."""
    eps = {}
    for (label, a), r in zip(jobs, res):
        reg = a[4]["regime"]
        eps.setdefault(f"{regime_key((reg['delay_s'], reg['commit_s']))} | {label} | {a[0]}", []).extend(r[3])
    cells = {k: {**summary(v), "per_scene": per_scene(v)} for k, v in eps.items()}
    recs = {k: [[e["scene"], e["t_start"], e["persona"], e["draw_seed"], e["branches"], round(e["score"], 6),
                 bool(e["at_fault"])] for e in v] for k, v in eps.items()}
    return cells, recs


def regime_jobs(planner: str, label: str, preds, base: dict, chosen: dict, scenes, lam, dose: str, draws,
                weighted: bool) -> tuple:
    """Per regime: label | predictor at margin 0 (not the most-likely single) and label+margin |
    predictor at its tuned margin (not run when that is 0: the cell is the margin-0 one); the
    oracle once per commitment interval at delay 0 and at the corner (the invariance check)."""
    jobs, alias = [], {}
    for r in REGIMES:
        k = regime_key(r)
        for n in preds:
            m = chosen[k][n]
            if n != "Ideal single (most likely)":
                jobs += [(label, (n, 0, scenes, lam, regime_cfg(base, r), None, dose, d, weighted)) for d in draws]
            if m == 0.0 and n != "Ideal single (most likely)":
                alias[f"{k} | {label}+margin | {n}"] = f"{k} | {label} | {n}"
            else:
                jobs += [(f"{label}+margin", (n, 0, scenes, lam, regime_cfg(base, r, m), None, dose, d, weighted)) for d in draws]
    for r in [(REGIME_DELAYS[0], c) for c in REGIME_COMMITS] + [REGIME_CORNER]:
        jobs += [(label, ("Branch oracle", 0, scenes, lam, regime_cfg(base, r), None, dose, d, weighted)) for d in draws]
    return [(planner, a, lbl) for lbl, a in jobs], alias


def regime_map(workers: int) -> None:
    """The public cells on the 73 primary scenes at the addendum's draws: the proposal planner's
    single (mean) and set at margin 0 and tuned, the tuned most-likely single, and its oracle."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    ref, dose = preregistered(), matched_dose()
    tuned = json.loads(REGIME_TUNE_JSON.read_text(encoding="utf-8"))
    assert tuned["rule_commit"] == ref["rule_commit"], "tuning ran under a different rule commit"
    draws = regime_eval_draws()
    assert_draws_free("primary", draws)
    _, lam = rs._v2_cfg()
    base, scenes = frozen_proposal(), primary_scenes()
    jobs, alias = regime_jobs("proposal", "proposal", ("Ideal single (mean)", "Ideal single (most likely)", "Ideal set"),
                              base, tuned["chosen_margin_m"], scenes, lam, dose, draws, False)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_lat_task, [(p, a) for p, a, _ in jobs]))
    check_replans([(p, a) for p, a, _ in jobs], res)
    cells, recs = regime_cells([(lbl, a) for _, a, lbl in jobs], res)
    for k, v in alias.items():
        cells[k], recs[k] = cells[v], recs[v]
    payload = {**ref, **repro({"proposal": base, "regimes": REGIMES, "chosen_margin_m": tuned["chosen_margin_m"]}, scenes),
               "dose": dose, "scenes": "73 primary scenes", "draws": list(draws), "planner_cfg": base,
               "chosen_margin_m": tuned["chosen_margin_m"], "aliases": alias, "cells": cells, "episodes": recs,
               "when": datetime.now().isoformat(timespec="seconds")}
    REGIME_MAP_JSON.write_text(json.dumps(payload), encoding="utf-8")
    print_cells({"": cells}, {})
    print("saved:", REGIME_MAP_JSON)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "eligibility", "stage1", "tune", "matched", "confirm-tune", "confirm",
                                    "path-validity", "path-stage1", "path-tune", "path-confirm", "confirm-1b",
                                    "stats-calibration", "draw-provenance", "regime-resolution", "regime-pilot",
                                    "regime-anchor-check", "regime-calibration", "regime-tune", "regime-map"))
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
    elif args.cmd == "path-validity":
        path_validity(args.workers)
    elif args.cmd == "path-stage1":
        path_stage1(args.workers)
    elif args.cmd == "path-tune":
        path_tune(args.workers)
    elif args.cmd == "path-confirm":
        path_confirm(args.workers)
    elif args.cmd == "confirm-1b":
        confirm_1b(args.workers)
    elif args.cmd == "stats-calibration":
        stats_calibration()
    elif args.cmd == "draw-provenance":
        draw_provenance_cmd()
    elif args.cmd == "regime-resolution":
        regime_resolution(args.workers)
    elif args.cmd == "regime-pilot":
        regime_pilot(args.workers)
    elif args.cmd == "regime-anchor-check":
        regime_anchor_check(args.workers)
    elif args.cmd == "regime-calibration":
        regime_calibration(args.workers)
    elif args.cmd == "regime-tune":
        regime_tune(args.workers)
    elif args.cmd == "regime-map":
        regime_map(args.workers)
    else:
        stage1(args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
