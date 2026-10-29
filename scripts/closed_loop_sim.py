"""Closed-loop (non-reactive) test: does a predictor make a planner drive safely?

Pre-registered in docs/wiki/closed-loop-test.md. Every recorded test scene is
replayed; the other road users follow their logged tracks (they do not react),
and the ego is driven by one fixed sampling planner whose only varying input is
the predictor package (its predictions plus its val-calibrated 90% conformal
band, exactly as in the safety scorecard). Every package sees identical
episodes.

Ego: follows the human driver's (smoothed) route in Frenet coordinates (s, d)
with a car footprint; every 0.2 s it picks one of 35 manoeuvres (7 constant
accelerations x 5 lateral offsets, reached over 1.5 s) for the next 3 s that
keeps its footprint clear of every agent's predicted footprint inflated by the
band; among clear manoeuvres it minimises speed-tracking and comfort costs, and
if none is clear it takes the least-penetrating one. Agents present without a
full 2 s history are extrapolated at constant velocity (same for every package).

Episode score (nuPlan-style): 0 after an at-fault collision, otherwise
(5 progress + 2 no-harsh-braking + 2 no-near-miss) / 9.

Usage:
    python scripts/closed_loop_sim.py --split val --references-only   # planner check
    python scripts/closed_loop_sim.py                                 # full test run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from conformal_wrapper import radii_per_class, radii_per_timestep
from dd_bootstrap import bootstrap_counts, ci, per_scene_sums, ratio_replicates, scene_codes
from dd_split import load_splits

REPO = Path(__file__).resolve().parent.parent
SCENES = REPO / "outputs" / "multiagent_world_v1" / "scenes"
EGO_CSV = REPO / "outputs" / "ego_trajectories.csv"
CL_DIR = REPO / "outputs" / "closed_loop"
PRED_DIR = CL_DIR / "preds"
CV_DIR = REPO / "outputs" / "cv_folds"            # fold models + their canonical dumps
CV_PRED_DIR = CL_DIR / "cv"                       # fold predictions on dense windows
OUT_JSON = REPO / "outputs" / "benchmark_v2" / "closed_loop.json"
LATENCY = REPO / "outputs" / "benchmark_v2" / "safety_latency.json"
RULE_DOC = "docs/wiki/closed-loop-test.md"

DT = 0.1
REPLAN = 2                          # frames between replans (0.2 s)
# planning horizon 3 s: checked every 0.1 s for the first second, then every 0.2 s
PLAN_OFFSETS = np.array(list(range(1, 11)) + list(range(12, 31, 2)))
H_FRAMES = int(PLAN_OFFSETS[-1])
SAFETY_BUFFER = 0.25                # m, added to every predicted footprint (all packages)
EP_STARTS = (19, 29, 39, 49)
EP_LEN = 50                         # frames (5 s)
PERSONAS = {"nominal": 0.0, "assertive": 6.0}     # minimum reference speed (m/s)
ACCELS = (-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0)
LAT_TARGETS = (-1.5, -0.75, 0.0, 0.75, 1.5)
LAT_TIME = 1.5
V_MAX = 15.0
ROUTE_EXT = 80.0
HARSH_DECEL = -4.0
NEAR_MISS_M = 0.5
AT_FAULT_MIN_SPEED = 0.5
AT_FAULT_REAR_M = -1.0              # near-miss clearance counts agents from 1 m behind the sensor
EGO_CENTRE = 0.3                    # ego centre ahead of the sensor (m), ground-truth footprint
FRONT_M = 1.0                       # "in front": agent centre >= 1 m ahead of the ego centre
SWERVE_MS = 0.1                     # ego lateral speed toward the agent that counts as swerving
CUT_IN_MS = 2.0                     # agent lateral speed toward the ego that makes it a cut-in
SCORE_W = (5.0, 2.0, 2.0)
SEEDS = (0, 1, 2)
GLITCH_M = 1.0                      # ego pose > 1 m off its 9-frame median path = pose glitch
COLLISION_TOL = 0.25                # m of footprint overlap tolerated before a contact counts
# Planner band handling (v1 defaults: band is a hard constraint over the whole horizon,
# least-penetration fallback). Overridden per run via run_package's cfg.
PLAN_CFG = {"hard_s": 3.0, "soft_w": 0.0, "fallback": "least_penetration"}
# Planner v3 options (development, docs/wiki/planner.md); these defaults reproduce v2 and
# are reset on every run_package call.
#   accels   longitudinal accelerations to sample
#   clear_w  weight of the clearance cost: sum over agents of max(0, clear_m - gap)^2,
#            gap = predicted footprint-to-footprint distance (no band, no buffer)
#   jerk_w   weight of (a - previous a)^2;  lat_w  weight of (d_target - previous d_target)^2
V3_DEFAULTS = {"accels": ACCELS, "clear_w": 0.0, "clear_m": 1.0, "jerk_w": 0.0, "lat_w": 0.0}
PLAN_CFG.update(V3_DEFAULTS)

# Nominal footprints (IDD-3D box sizes are not in this repo): circle centres along
# the heading and a common radius.
# Ego: 4.5 m x 1.7 m, centred 0.5 m ahead of the sensor (the logged ego position).
# Measured from where annotated agents never appear around the sensor on train+val
# scenes (nearest centres: 4.4 m ahead, 3.3 m behind, 1.3 m to either side).
EGO = ((-0.9, 0.5, 1.9), 0.85)
EGO_REACH = 1.4                     # max distance of an ego circle from the ego centre
CAR = ((-1.3, 0.0, 1.3), 0.85)      # 4.3 m x 1.7 m
FOOTPRINTS = {
    **{c: ((0.0,), 0.3) for c in ("Pedestrian", "Child")},
    **{c: ((-0.6, 0.6), 0.35) for c in ("MotorcyleRider", "ScooterRider", "BicycleRider",
                                         "Motorcycle", "Scooter", "Bicycle", "BicycleGroup")},
    **{c: CAR for c in ("Car", "TourCar", "Van", "PoliceCar")},
    **{c: ((-2.2, -0.73, 0.73, 2.2), 1.2) for c in ("Truck", "Crane")},     # 6.8 m x 2.4 m
    "Bus": ((-3.0, -1.0, 1.0, 3.0), 1.25),                                  # 8.5 m x 2.5 m
}
OTHER = ((0.0,), 0.5)
# Ground-truth collision geometry (instrument v2): oriented rectangles (length, width),
# same nominal sizes the circles approximate. The planner keeps the circles.
# Ego: the region no annotated agent edge enters around the sensor in (almost) all frames,
# slow ones included (p0.5: 2.0 m ahead, 1.4 m behind, ~1 m to each side) -> 3.4 x 1.7 m
# centred 0.3 m ahead. The planner plans with the larger circle footprint (EGO).
EGO_RECT = (3.4, 1.7)
RECTS = {
    **{c: (0.6, 0.6) for c in ("Pedestrian", "Child")},
    **{c: (1.9, 0.7) for c in ("MotorcyleRider", "ScooterRider", "BicycleRider",
                               "Motorcycle", "Scooter", "Bicycle", "BicycleGroup")},
    **{c: (4.3, 1.7) for c in ("Car", "TourCar", "Van", "PoliceCar")},
    **{c: (6.8, 2.4) for c in ("Truck", "Crane")},
    "Bus": (8.5, 2.5),
}
OTHER_RECT = (1.0, 1.0)
VRU = {"Pedestrian", "Child", "MotorcyleRider", "ScooterRider", "BicycleRider", "BicycleGroup"}

# package -> (kind, prediction file key, scorecard source for its val-calibrated band)
PACKAGES = {
    "CV (vel_k=5)": ("single", "cv", "cv"),
    "MLP": ("single", "mlp", "mlp"),
    "ST-GAT (MHA)": ("single", "stgat", "preds_stgat.npz"),
    "Map-free transformer": ("single", "transformer", "preds_transformer.npz"),
    "K-mode (top-1)": ("single", "mfp", "preds_mfp.npz"),
    "AgentFormer (most-likely)": ("single", "agentformer", "preds_agentformer.npz"),
    "LED (mean-denoised)": ("single", "led", "preds_led.npz"),
    "Trajectron++ (most-likely)": ("single", "trajectronpp", "preds_trajectronpp_canonical.npz"),
    "K-mode set (10 modes)": ("set", "mfp", "preds_mfp.npz"),
    "AgentFormer set (10 samples)": ("set", "agentformer", "preds_agentformer.npz"),
    "LED set (10 samples)": ("set", "led", "preds_led.npz"),
    "Trajectron++ set (10 samples)": ("set", "trajectronpp", "preds_trajectronpp_canonical.npz"),
}
# Packages that exist only as cross-validation fold models (closed-loop v2 amendment 3);
# kept out of PACKAGES so the frozen-split runs and the dev grid are unchanged.
EXTRA_PACKAGES = {
    "MF transformer (top-1)": ("single", "mftf", "preds_mftf.npz"),
    "MF transformer set (10 modes)": ("set", "mftf", "preds_mftf.npz"),
}
ALL_PACKAGES = {**PACKAGES, **EXTRA_PACKAGES}
REFERENCES = ("Human driver (log replay)", "Oracle (true future, no band)", "No prediction (ignore agents)")


def footprint(cls: str):
    return FOOTPRINTS.get(str(cls), OTHER)


# --------------------------------------------------------------------------
# Scene + route
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def ego_table():
    e = pd.read_csv(EGO_CSV, dtype={"frame_id": str})
    e["f"] = e.frame_id.astype(int)
    # ego_yaw is offset from the direction of motion (sensor axis convention):
    # measured once over all moving frames, used only when the ego never moves.
    offs = []
    for _, g in e.groupby("scene_id"):
        g = g.sort_values("f"); xy = g[["ego_x", "ego_y"]].to_numpy(); yaw = g.ego_yaw.to_numpy()
        d = xy[5:] - xy[:-5]; mv = np.linalg.norm(d, axis=1) > 2.0
        offs += list(np.angle(np.exp(1j * (np.arctan2(d[mv, 1], d[mv, 0]) - yaw[:-5][mv]))))
    return e, float(np.median(offs))


def smooth(x: np.ndarray, w: int = 9) -> np.ndarray:
    k = np.ones(w) / w
    pad = np.pad(x, ((w // 2, w // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def medfilt(x: np.ndarray, w: int = 9) -> np.ndarray:
    pad = np.pad(x, ((w // 2, w // 2), (0, 0)), mode="edge")
    return np.stack([np.median(np.lib.stride_tricks.sliding_window_view(pad[:, j], w), axis=1)
                     for j in range(x.shape[1])], 1)


def rect(centre, heading, size) -> np.ndarray:
    """(4,2) corners of an oriented rectangle (length along heading, width across)."""
    L, W = size
    u = np.asarray(heading, float)
    n = np.array([-u[1], u[0]])
    return centre + np.array([L / 2 * u + W / 2 * n, L / 2 * u - W / 2 * n,
                              -L / 2 * u - W / 2 * n, -L / 2 * u + W / 2 * n])


def rect_gap(A: np.ndarray, B: np.ndarray) -> float:
    """Signed gap between two convex quads: Euclidean distance when apart, minus the
    smallest separating-axis overlap when they intersect."""
    overlaps = []
    for Q in (A, B):
        for i in range(2):
            e = Q[(i + 1) % 4] - Q[i]
            ax = np.array([-e[1], e[0]]) / max(np.linalg.norm(e), 1e-12)
            pa, pb = A @ ax, B @ ax
            overlaps.append(min(pa.max(), pb.max()) - max(pa.min(), pb.min()))
    if min(overlaps) > 0:
        return -float(min(overlaps))

    def pt_seg(P, Q):   # min distance from points P (n,2) to the 4 edges of Q
        a, b = Q, np.roll(Q, -1, axis=0)
        ab = b - a
        t = np.clip(np.einsum("pkj,kj->pk", P[:, None] - a[None], ab) / np.einsum("kj,kj->k", ab, ab), 0, 1)
        return np.linalg.norm(P[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1).min()
    return float(min(pt_seg(A, B), pt_seg(B, A)))


def rect_size(cls) -> tuple:
    return RECTS.get(str(cls), OTHER_RECT)


class Route:
    """Arc-length parametrised polyline (0.25 m grid) with tangents and normals."""

    def __init__(self, pts: np.ndarray):
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(s[-1])
        self.s_grid = np.arange(0.0, self.length + 0.25, 0.25)
        self.xy = np.stack([np.interp(self.s_grid, s, pts[:, j]) for j in range(2)], 1)
        tan = np.gradient(self.xy, axis=0)
        tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
        self.tan = tan
        self.nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        self.head = tan                     # footprint orientation; replaced by set_heading

    def set_heading(self, s_h: np.ndarray, head: np.ndarray) -> None:
        """Orient the ego along the human's logged heading at each arc length (the pose yaw):
        robust where the path tangent is noise (crawling in jams). Beyond the human's
        path the last heading holds."""
        idx = np.clip(np.searchsorted(s_h, self.s_grid, side="right") - 1, 0, len(s_h) - 1)
        h = head[idx]
        self.head = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-9)

    def pose(self, s: np.ndarray, d: np.ndarray):
        """World position and heading unit vector for arc length s and lateral offset d."""
        i = np.clip(np.rint(np.asarray(s) / 0.25).astype(int), 0, len(self.s_grid) - 1)
        return self.xy[i] + np.asarray(d)[..., None] * self.nrm[i], self.head[i]

    def project(self, p: np.ndarray) -> float:
        return float(self.s_grid[np.argmin(np.linalg.norm(self.xy - p, axis=1))])


def load_scene(sid: str) -> dict:
    d = np.load(SCENES / f"{sid}.npz", allow_pickle=True)
    e, yaw_off = ego_table()
    g = e[e.scene_id == sid].set_index("f").loc[[int(f) for f in d["frame_id"]]]
    raw = g[["ego_x", "ego_y"]].to_numpy(float)
    med = medfilt(raw)                          # robust to single-frame pose glitches
    glitch = np.linalg.norm(raw - med, axis=1) > GLITCH_M
    P = smooth(med)
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    if step.sum() < 2.0:                       # the driver never moves: heading from yaw
        h = g.ego_yaw.to_numpy()[0] + yaw_off
        u = np.array([np.cos(h), np.sin(h)])
        pts = np.stack([P[0], P[0] + ROUTE_EXT * u])
        s_h = np.zeros(len(P))
    else:
        keep = np.concatenate([[True], step > 0.02])
        pts = P[keep]
        tail = pts[-1] - pts[max(0, len(pts) - 12)]
        u = tail / max(np.linalg.norm(tail), 1e-9)
        pts = np.vstack([pts, pts[-1] + ROUTE_EXT * u])
        s_h = np.concatenate([[0.0], np.cumsum(step)])
    route = Route(pts)
    yaw_head = np.stack([np.cos(g.ego_yaw.to_numpy() + yaw_off), np.sin(g.ego_yaw.to_numpy() + yaw_off)], 1)
    route.set_heading(s_h, yaw_head)
    v_h = np.clip(np.gradient(s_h) / DT, 0.0, V_MAX)
    v_h = smooth(v_h[:, None])[:, 0]
    pos, mask = d["pos"].astype(np.float64), d["mask"].astype(bool)
    cls = np.asarray(d["agent_type"], dtype=str)
    return {"sid": sid, "pos": pos, "mask": mask, "cls": cls, "agent_id": d["agent_id"].astype(int),
            "ego_raw": raw, "glitch": glitch,
            # the sensor's own orientation (pose yaw + the measured axis offset): the frame the
            # annotations were made in, so the human replay keeps their exact geometry
            "ego_head_raw": yaw_head,
            "heading": agent_headings(pos, mask, route), "route": route, "s_h": s_h, "v_h": v_h,
            "T": pos.shape[0]}


def agent_headings(pos, mask, route) -> np.ndarray:
    """(T,A,2) unit heading per frame from the logged motion (+-2 frames), held through
    stops; agents that never move take the route tangent at their position."""
    T, A, _ = pos.shape
    H = np.full((T, A, 2), np.nan)
    for a in range(A):
        for t in range(T):
            lo, hi = max(0, t - 2), min(T - 1, t + 2)
            if mask[lo, a] and mask[hi, a] and hi > lo:
                v = pos[hi, a] - pos[lo, a]
                if np.linalg.norm(v) > 0.3:
                    H[t, a] = v / np.linalg.norm(v)
        valid = ~np.isnan(H[:, a, 0])
        if valid.any():
            idx = np.where(valid)[0]
            nearest = idx[np.abs(np.arange(T)[:, None] - idx[None]).argmin(axis=1)]
            H[:, a] = H[nearest, a]
        else:
            first = np.argmax(mask[:, a]) if mask[:, a].any() else 0
            _, tan = route.pose(np.array(route.project(pos[first, a])), np.array(0.0))
            H[:, a] = tan
    return H


def circles(centre, heading, fp):
    """(..., n, 2) circle centres along the heading for footprint fp."""
    offs = np.asarray(fp[0])
    return centre[..., None, :] + offs[:, None] * heading[..., None, :]


# --------------------------------------------------------------------------
# Predictions available to the planner at frame t
# --------------------------------------------------------------------------
class Predictor:
    """Per-agent predicted positions + band radii for one package (or reference)."""

    def __init__(self, name: str, seed: int, fold: int | None = None):
        """fold=None: the frozen-split models; fold=k: the cross-validation models of
        fold k, with bands calibrated on that fold's val scenes."""
        self.name, self.seed = name, seed
        self.kind = "ref" if name in REFERENCES else ALL_PACKAGES[name][0]
        self.cv_band = band_table("single", "cv", 0) if fold is None else fold_band_table("cv", fold, seed)
        if self.kind == "ref":
            self.rows, self.band = {}, None
            return
        kind, key, source = ALL_PACKAGES[name]
        # frozen-split models: test-scene predictions plus, when present, val-scene ones
        # (planner development); fold models: that fold's test scenes
        pdirs = [PRED_DIR, CL_DIR / "preds_val"] if fold is None else [CV_PRED_DIR / f"fold{fold}" / "preds"]
        P, self.rows = [], {}
        for pdir in pdirs:
            f = pdir / f"seed{seed}" / f"{key}.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=True)
            off = sum(len(x) for x in P)
            P.append(z["samples"].astype(np.float64) if kind == "set" else z["pred"].astype(np.float64)[:, None])
            self.rows.update({(str(s), int(t0) + 19, int(a)): off + i
                              for i, (s, t0, a) in enumerate(zip(z["scene_id"], z["t0"], z["agent_id"]))})
        self.p = np.concatenate(P)
        self.band = band_table(kind, source, seed) if fold is None else fold_band_table(key, fold, seed, kind)

    def at(self, sc: dict, t: int, lam: float = 1.0):
        """Agents present at frame t: list of (agent index, samples (K,60,2), band (60,))."""
        out = []
        present = np.where(sc["mask"][t])[0]
        for a in present:
            cls = sc["cls"][a]
            if self.name == "No prediction (ignore agents)":
                continue
            if self.name == "Oracle (true future, no band)":
                fut = sc["pos"][t + 1:t + 61, a]
                m = sc["mask"][t + 1:t + 61, a]
                f = np.full((60, 2), np.nan); f[:len(fut)][m] = fut[m]
                out.append((a, f[None], np.zeros(60)))
                continue
            i = self.rows.get((sc["sid"], t, int(sc["agent_id"][a])))
            if i is not None:
                out.append((a, self.p[i], lam * band_row(self.band, cls)))
            else:                                  # no full 2 s history: constant velocity
                out.append((a, cv_extrapolate(sc, t, a)[None], lam * band_row(self.cv_band, cls)))
        return out


def cv_extrapolate(sc, t, a) -> np.ndarray:
    hist = []
    for k in range(t, max(-1, t - 6), -1):
        if not sc["mask"][k, a]:
            break
        hist.append(sc["pos"][k, a])
    hist = np.array(hist[::-1])
    v = (hist[-1] - hist[0]) / max(len(hist) - 1, 1)
    return hist[-1] + np.arange(1, 61)[:, None] * v[None]


@lru_cache(maxsize=None)
def band_table(kind: str, source: str, seed: int):
    """Val-calibrated per-step radii (marginal + per class), as in the safety scorecard."""
    import safety_scorecard as sc
    pop = val_population()
    pred = sc.package_predictions(kind, source, seed, pop, sc.K_SET)
    val = pop["split"] == "val"
    s = sc.min_sample_dist(pred[val], pop["gt"][val])
    marginal = radii_per_timestep(s, pop["fm"][val], sc.ALPHA)
    return marginal, radii_per_class(s, pop["fm"][val], pop["cls"][val], sc.ALPHA, marginal, sc.MIN_CAL)


@lru_cache(maxsize=None)
def fold_band_table(key: str, fold: int, seed: int = 0, kind: str = "single"):
    """Per-step radii calibrated on fold `fold`'s val scenes (canonical windows); for a
    set ("set" kind) the score is the min-over-samples error, as in the scorecard."""
    import safety_scorecard as sc
    from conformal_wrapper import cv_predict
    pop = sc.load_population(REPO / "splits" / "cv_folds" / f"fold{fold}.json")
    val = pop["split"] == "val"
    if key == "cv":
        pred = pop["obs"][:, -1:, :] + cv_predict(pop["obs"], 5)
    elif key == "mlp":
        w = np.load(CV_DIR / f"fold{fold}" / f"mlp_best_seed{seed}.npz")
        pred = sc.mlp_world(pop["obs"], {k: w[k] for k in w.files})
    else:
        d = np.load(CV_DIR / f"fold{fold}" / "dumps" / f"preds_{key}.npz", allow_pickle=True)
        assert np.array_equal(d["split"].astype(str), pop["split"]), f"fold {fold} {key}: row order"
        assert np.array_equal(d["fut_mask"].astype(bool), pop["fm"]), f"fold {fold} {key}: fut_mask"
        pred = (d["samples"][:, :10] if kind == "set" else d["pred"][:, None]).astype(np.float64)
    if pred.ndim == 3:
        pred = pred[:, None]
    s = np.linalg.norm(pred[val] - pop["gt"][val][:, None], axis=-1).min(axis=1)
    marginal = radii_per_timestep(s, pop["fm"][val], sc.ALPHA)
    return marginal, radii_per_class(s, pop["fm"][val], pop["cls"][val], sc.ALPHA, marginal, sc.MIN_CAL)


@lru_cache(maxsize=1)
def val_population():
    import safety_scorecard as sc
    return sc.load_population(REPO / "splits" / "scene_splits.json")


def band_row(table, cls) -> np.ndarray:
    marginal, per_class = table
    return per_class.get(str(cls), marginal)


# --------------------------------------------------------------------------
# Planner + simulation
# --------------------------------------------------------------------------
def rollout(s0, d0, v0, a, d_target, n_frames):
    """Ego (s, d, v) for frames 1..n_frames under constant accel a and a linear
    lateral move to d_target over LAT_TIME."""
    s, d, v = np.empty(n_frames), np.empty(n_frames), np.empty(n_frames)
    rate = (d_target - d0) / LAT_TIME
    cs, cd, cv = s0, d0, v0
    for k in range(n_frames):
        nv = min(max(cv + a * DT, 0.0), V_MAX)
        cs += 0.5 * (cv + nv) * DT
        cd = cd + rate * DT if abs(d_target - cd) > abs(rate * DT) else d_target
        cv = nv
        s[k], d[k], v[k] = cs, cd, cv
    return s, d, v


def plan(sc, route, s0, d0, v0, v_ref, preds):
    """Choose (a, d_target) for the next replan interval. Reads the previous command from
    sc["cmd_prev"] and leaves sc["plan_fallback"] (True when no manoeuvre was clear)."""
    cands = [(a, dt) for a in PLAN_CFG["accels"] for dt in LAT_TARGETS]
    idx = PLAN_OFFSETS - 1                                               # prediction indices
    n_steps = len(idx)
    E, v_end = [], []
    for a, dt in cands:
        s, d, v = rollout(s0, d0, v0, a, dt, H_FRAMES)
        xy, tan = route.pose(s[idx], d[idx])
        E.append(circles(xy, tan, EGO))                                  # (15, 3, 2)
        v_end.append(v[-1])
    E = np.stack(E)                                                      # (C, steps, 3, 2)
    hard = PLAN_OFFSETS * DT <= PLAN_CFG["hard_s"] + 1e-9                # steps under the hard constraint
    worst = np.zeros(len(cands))                                         # hard-step penetration
    soft = np.zeros(len(cands))                                          # later-step penetration
    clear = np.zeros(len(cands))                                         # clearance cost (v3)
    clear_on = PLAN_CFG["clear_w"] > 0
    ego_c = E.mean(axis=2)                                               # centres (C, steps, 2)
    for a_idx, samples, band in preds:
        fp = footprint(sc["cls"][a_idx])
        P = samples[:, idx]                                              # (K, steps, 2)
        ok = np.isfinite(P[..., 0])
        if not ok.any():
            continue
        # heading from the predicted motion over >= 0.3 s; slow agents keep their logged heading
        full = np.concatenate([np.repeat(sc["pos_now"][a_idx][None, None], samples.shape[0], 0), samples], 1)
        back = np.maximum(idx + 1 - 3, 0)
        hd = full[:, idx + 1] - full[:, back]
        n = np.linalg.norm(hd, axis=-1, keepdims=True)
        hd = np.where(n > 0.5, hd / np.maximum(n, 1e-9), sc["head_now"][a_idx])
        reach = EGO[1] + fp[1] + band[idx] + EGO_REACH + max(np.abs(fp[0])) + 0.5
        if clear_on:
            reach = reach + PLAN_CFG["clear_m"]
        cd = np.linalg.norm(ego_c[:, None] - np.where(ok[..., None], P, 1e9)[None], axis=-1)   # (C,K,15)
        if not (cd < reach[None, None]).any():
            continue
        hd = np.where(np.isfinite(hd), hd, sc["head_now"][a_idx])
        A = circles(np.where(ok[..., None], P, 1e9), hd, fp)             # (K, steps, m, 2)
        dist = np.linalg.norm(E[:, None, :, :, None] - A[None, :, :, None, :, :], axis=-1)  # (C,K,steps,3,m)
        if clear_on:
            gap = dist.min(axis=(3, 4)) - (EGO[1] + fp[1])               # (C,K,steps)
            clear += np.clip(PLAN_CFG["clear_m"] - gap, 0, None).max(axis=(1, 2)) ** 2
        need = EGO[1] + fp[1] + band[idx] + SAFETY_BUFFER                # (steps,)
        pen = ((need[None, None, :, None, None] - dist) / need[None, None, :, None, None]).max(axis=(1, 3, 4))
        worst = np.maximum(worst, pen[:, hard].max(axis=1))             # pen: (C, steps)
        if (~hard).any():
            soft = np.maximum(soft, np.clip(pen[:, ~hard], 0, None).max(axis=1))
    feasible = worst <= 0.0
    cost = np.array([((ve - v_ref) / 2.0) ** 2 + 0.05 * a * a + 0.3 * dt * dt + 0.5 * abs(dt - d0)
                     for (a, dt), ve in zip(cands, v_end)]) + PLAN_CFG["soft_w"] * soft
    cost = cost + v3_cost(cands, sc, clear)
    sc["plan_fallback"] = not feasible.any()
    if feasible.any():
        return cands[int(np.argmin(np.where(feasible, cost, np.inf)))]
    if PLAN_CFG["fallback"] == "brake":                                  # emergency stop, no swerve
        return ACCELS[0], min(LAT_TARGETS, key=lambda x: abs(x - d0))
    return cands[int(np.lexsort((cost, worst))[0])]                      # least penetration


def v3_cost(cands, sc, clear) -> np.ndarray:
    """Planner v3 terms: clearance plus command smoothness (0 with the v2 defaults)."""
    a_prev, d_prev = sc.get("cmd_prev", (0.0, 0.0))
    a, dt = np.array(cands, dtype=float).T
    return (PLAN_CFG["clear_w"] * clear + PLAN_CFG["jerk_w"] * (a - a_prev) ** 2
            + PLAN_CFG["lat_w"] * (dt - d_prev) ** 2)


def run_episode(sc, t_start, persona_min_v, predictor, lam=1.0) -> dict:
    route, T = sc["route"], sc["T"]
    human = predictor.name == "Human driver (log replay)"
    s, d, v = sc["s_h"][t_start], 0.0, sc["v_h"][t_start]
    a_cmd, d_target = 0.0, 0.0
    harsh, min_clear, min_clear_moving = False, np.inf, np.inf
    hit, hit_vru, hits = False, False, set()
    cmds, n_fallback, plan_s = [], 0, 0.0              # diagnostics (not in the score)
    for f in range(t_start, t_start + EP_LEN):
        d_prev = d
        if human:
            s, d, v = sc["s_h"][f + 1], 0.0, sc["v_h"][f + 1]
        else:
            if (f - t_start) % REPLAN == 0:
                sc["pos_now"] = sc["pos"][f]
                sc["head_now"] = sc["heading"][f]
                v_ref = max(sc["v_h"][min(f + 30, T - 1)], persona_min_v)
                sc["cmd_prev"] = (a_cmd, d_target)
                sc.pop("plan_fallback", None)
                t_plan = time.perf_counter()
                a_cmd, d_target = plan(sc, route, s, d, v, v_ref, predictor.at(sc, f, lam))
                plan_s += time.perf_counter() - t_plan
                fb = sc.pop("plan_fallback", None)              # None: the planner doesn't report it
                n_fallback = None if fb is None or n_fallback is None else n_fallback + fb
                cmds.append(a_cmd)
                harsh |= a_cmd <= HARSH_DECEL
            (s,), (d,), (v,) = rollout(s, d, v, a_cmd, d_target, 1)
        # ground truth at frame f+1
        g = f + 1
        xy, tan = route.pose(np.array(s), np.array(d))
        if human:      # the raw logged pose keeps the annotations' exact geometry
            xy, tan = sc["ego_raw"][g], sc["ego_head_raw"][g]
        sc["route_tan_now"] = tan
        ego = rect(xy + EGO_CENTRE * tan, tan, EGO_RECT)
        if "react" in sc:               # reactive agents (reactive_sim.py) move to frame g first
            sc["react"](g, xy + EGO_CENTRE * tan, tan, float(v))
        nrm = np.array([-tan[1], tan[0]])
        ddot = (d - d_prev) / DT
        for a in np.where(sc["mask"][g])[0]:
            size = rect_size(sc["cls"][a])
            rel = sc["pos"][g, a] - xy
            if rel @ rel > 400.0:                                       # > 20 m away: no contact
                continue
            gap = rect_gap(ego, rect(sc["pos"][g, a], sc["heading"][g, a], size))
            lon, lat = float(rel @ tan), float(rel @ nrm)
            if lon >= AT_FAULT_REAR_M:
                min_clear = min(min_clear, gap)
                if v >= AT_FAULT_MIN_SPEED:
                    min_clear_moving = min(min_clear_moving, gap)
            if gap < -COLLISION_TOL and v >= AT_FAULT_MIN_SPEED and a not in hits and at_fault(sc, g, a, lon, lat, ddot, size):
                hits.add(a); hit = True
                hit_vru |= sc["cls"][a] in VRU
    t_end = t_start + EP_LEN
    prog_h = sc["s_h"][t_end] - sc["s_h"][t_start]
    # nothing to achieve when the human barely moved (< 2 m): full progress
    progress = 1.0 if prog_h < 2.0 else min(1.0, (s - sc["s_h"][t_start]) / prog_h)
    no_harsh, no_near = (not harsh), bool(min_clear >= NEAR_MISS_M)
    w = SCORE_W
    score = 0.0 if hit else (w[0] * progress + w[1] * no_harsh + w[2] * no_near) / sum(w)
    return {"scene": sc["sid"], "t_start": t_start, "score": float(score), "at_fault": hit,
            "at_fault_vru": hit_vru, "n_at_fault_agents": len(hits), "progress": float(progress),
            "harsh_braking": harsh, "near_miss": not no_near,
            "min_clearance_m": float(min_clear) if np.isfinite(min_clear) else None,
            # diagnostics: near-miss while the ego moves, fallback replans, command jerk
            # (RMS change of the commanded acceleration per second), planning time
            "near_miss_moving": bool(min_clear_moving < NEAR_MISS_M), "hit_agents": sorted(int(x) for x in hits),
            "n_replans": len(cmds), "n_fallback": n_fallback,
            "jerk_rms": float(np.sqrt(np.mean((np.diff(cmds) / (REPLAN * DT)) ** 2))) if len(cmds) > 1 else 0.0,
            "plan_ms": 1000.0 * plan_s / max(len(cmds), 1)}


def at_fault(sc, g, a, lon, lat, ddot, size) -> bool:
    """The moving ego caused the contact: the agent is in front of it, or the ego was
    swerving toward it; an agent cutting in sideways faster than CUT_IN_MS is not
    the ego's fault (non-reactive replay, nuPlan-style)."""
    half_w = 0.5 * EGO_RECT[1]
    in_front = (lon - EGO_CENTRE) >= FRONT_M and abs(lat) <= half_w + 0.5 * size[1] + 0.5 * half_w
    swerving = ddot * np.sign(lat) > SWERVE_MS and (lon - EGO_CENTRE) >= -1.0
    if not (in_front or swerving):
        return False
    if g >= 2 and sc["mask"][g - 2, a]:
        vel = (sc["pos"][g, a] - sc["pos"][g - 2, a]) / (2 * DT)
        nrm = np.array([-sc["route_tan_now"][1], sc["route_tan_now"][0]])
        if -np.sign(lat) * float(vel @ nrm) > CUT_IN_MS:
            return False
    return True


def episode_ok(sc, t0) -> bool:
    """The episode fits in the scene and its ego poses are free of glitches
    (data-quality rule, the same for every package)."""
    return t0 + EP_LEN <= sc["T"] - 1 and not sc["glitch"][t0:t0 + EP_LEN + 1].any()


NOMINAL_RECTS = dict(RECTS)
PRIMARY_INSTRUMENT = {"collision_tol": COLLISION_TOL, "ego_rect": EGO_RECT, "ego_centre": EGO_CENTRE,
                      "glitch_m": GLITCH_M, "agent_scale": 1.0}


def set_instrument(inst: dict) -> None:
    """Override ground-truth instrument settings in this process (sensitivity analysis):
    collision_tol, ego_rect, ego_centre, glitch_m, agent_scale (x nominal agent sizes)."""
    global COLLISION_TOL, EGO_RECT, EGO_CENTRE, GLITCH_M, RECTS, OTHER_RECT
    v = {**PRIMARY_INSTRUMENT, **inst}          # always relative to the primary instrument
    COLLISION_TOL, EGO_RECT, EGO_CENTRE, GLITCH_M = v["collision_tol"], tuple(v["ego_rect"]), v["ego_centre"], v["glitch_m"]
    k = v["agent_scale"]
    RECTS = {c: (L * k, W * k) for c, (L, W) in NOMINAL_RECTS.items()}
    OTHER_RECT = (1.0 * k, 1.0 * k)


def run_package(args) -> tuple:
    """args = (package, seed, scene ids, band scale, planner cfg, {scene: fold} or None);
    cfg may carry an "instrument" dict of ground-truth overrides (sensitivity analysis)."""
    name, seed, scene_ids, lam, cfg, folds = args
    cfg = dict(cfg)
    set_instrument(cfg.pop("instrument", {}))   # pool workers are reused: reset every task
    PLAN_CFG.update({**V3_DEFAULTS, **cfg})
    predictors, eps, excluded = {}, [], 0
    for sid in scene_ids:
        fold = None if folds is None else folds[sid]
        if fold not in predictors:
            predictors[fold] = Predictor(name, seed, fold)
        sc = load_scene(sid)
        for t0 in EP_STARTS:
            if not episode_ok(sc, t0):
                excluded += len(PERSONAS)
                continue
            for persona, vmin in PERSONAS.items():
                r = run_episode(sc, t0, vmin, predictors[fold], lam)
                r["persona"], r["fold"] = persona, fold
                eps.append(r)
    return name, seed, lam, eps, excluded


# --------------------------------------------------------------------------
# Statistics + decision rule
# --------------------------------------------------------------------------
def summarise(eps: list) -> dict:
    f = lambda k: float(np.mean([e[k] for e in eps]))   # noqa: E731
    return {"score": f("score"), "at_fault_rate": f("at_fault"), "at_fault_vru_rate": f("at_fault_vru"),
            "progress": f("progress"), "harsh_braking_rate": f("harsh_braking"),
            "near_miss_rate": f("near_miss"), "n_episodes": len(eps),
            "by_persona": {p: {"score": float(np.mean([e["score"] for e in eps if e["persona"] == p])),
                               "at_fault_rate": float(np.mean([e["at_fault"] for e in eps if e["persona"] == p]))}
                           for p in PERSONAS}}


def replicates(eps_by_seed: list, key: str, scenes, counts, rng) -> np.ndarray:
    """(B,) seed-pooled scene-bootstrap replicates of the 3-seed mean of `key`."""
    reps = []
    for eps in eps_by_seed:
        codes, _ = scene_codes(np.array([e["scene"] for e in eps], dtype=object), scenes)
        num = per_scene_sums(codes, len(scenes), np.array([float(e[key]) for e in eps]))
        den = per_scene_sums(codes, len(scenes), np.ones(len(eps)))
        reps.append(ratio_replicates(counts, num, den))
    reps = np.stack(reps, 1)
    k = reps.shape[1]
    w = rng.multinomial(k, np.full(k, 1.0 / k), size=counts.shape[0]) / k
    return (reps * w).sum(1)


def paired(reps, a, b, point) -> dict:
    d = reps[a] - reps[b]
    lo, hi = ci(d[np.isfinite(d)])
    return {"a": a, "b": b, "diff": point[a] - point[b], "ci95": [lo, hi], "tied": bool(lo <= 0.0 <= hi)}


def decide(point_p, reps_p, point_v, reps_v, latency) -> dict:
    leader = max(point_p, key=point_p.get)
    vs = [paired(reps_p, leader, n, point_p) for n in point_p if n != leader]
    tied = [leader] + [c["b"] for c in vs if c["tied"]]
    steps = [f"Leader on the episode score: {leader} ({point_p[leader]:.3f})."]
    if len(tied) == 1:
        steps.append("No package is statistically tied with the leader: it is the pick.")
        return {"leader": leader, "vs_leader": vs, "tied_on_score": tied, "pick": leader, "steps": steps}
    steps.append(f"Tied with the leader (paired 95% CI includes 0): {', '.join(tied[1:])}.")
    best = min(tied, key=point_v.get)
    pv = [paired(reps_v, best, n, point_v) for n in tied if n != best]
    keep = [best] + [c["b"] for c in pv if c["tied"]]
    steps.append(f"Tie-break 1 (at-fault collisions with pedestrians / two-wheelers, lower is "
                 f"better): best {best} ({point_v[best]:.3f}); still tied: {', '.join(keep[1:]) or 'none'}.")
    pick = keep[0]
    if len(keep) > 1:
        pick = min(keep, key=lambda n: latency[n]["median_ms"]) if latency else None
        steps.append(f"Tie-break 2 (latency): {pick}." if pick else "Tie-break 2 needs latency: pending.")
    return {"leader": leader, "vs_leader": vs, "tied_on_score": tied, "vru_pairs": pv,
            "tied_on_vru": keep, "pick": pick, "steps": steps}


def rule_commit():
    try:
        return subprocess.run(["git", "log", "-1", "--format=%H", "--", RULE_DOC], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--references-only", action="store_true", help="planner check: references only")
    ap.add_argument("--packages", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--save-json", default=None)
    ap.add_argument("--episodes-out", default=None,
                    help="also write every episode result (per package, seed, band scale) as JSON")
    args = ap.parse_args()

    mapping = load_splits(REPO / "splits" / "scene_splits.json")
    scene_ids = sorted(p.stem for p in SCENES.glob("*.npz") if mapping.get(p.stem) == args.split)
    names = list(REFERENCES) if args.references_only else (args.packages or list(PACKAGES)) + list(REFERENCES)
    cfg = dict(PLAN_CFG)
    tasks = [(n, s, scene_ids, 1.0, cfg, None) for n in names for s in ((0,) if n in REFERENCES else SEEDS)]
    if not args.references_only and args.packages is None:      # reported sensitivity: no band
        tasks += [(n, s, scene_ids, 0.0, cfg, None) for n in PACKAGES for s in SEEDS]
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name, seed, lam, eps, _ in ex.map(run_package, tasks):
            results.setdefault((name, lam), {})[seed] = eps
            s = summarise(eps)
            print(f"{name:32s} seed {seed} band x{lam:.0f}: score {s['score']:.3f}  at-fault "
                  f"{s['at_fault_rate']:.3f}  VRU {s['at_fault_vru_rate']:.3f}  progress {s['progress']:.2f}  "
                  f"harsh {s['harsh_braking_rate']:.2f}  near-miss {s['near_miss_rate']:.2f}", flush=True)

    if args.episodes_out:
        Path(args.episodes_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.episodes_out).write_text(json.dumps(
            [{"package": n, "band_scale": lam, "seed": s, "episodes": eps}
             for (n, lam), by_seed in results.items() for s, eps in by_seed.items()]), encoding="utf-8")
    scenes = np.asarray(sorted(scene_ids), dtype=object)
    counts = bootstrap_counts(len(scenes), args.n_boot, 0)
    rng = np.random.default_rng(1)
    packages, reps_p, reps_v = {}, {}, {}
    for (name, lam), by_seed in results.items():
        seeds = sorted(by_seed)
        per = [summarise(by_seed[s]) for s in seeds]
        entry = {k: float(np.mean([p[k] for p in per])) for k in per[0] if k not in ("by_persona", "n_episodes")}
        entry.update({"n_episodes": per[0]["n_episodes"], "seeds": seeds, "band_scale": lam,
                      "per_seed": {k: [p[k] for p in per] for k in ("score", "at_fault_rate", "at_fault_vru_rate")},
                      "by_persona": {pp: {k: float(np.mean([p["by_persona"][pp][k] for p in per]))
                                          for k in per[0]["by_persona"][pp]} for pp in PERSONAS}})
        if lam == 1.0:
            rp = replicates([by_seed[s] for s in seeds], "score", scenes, counts, rng)
            entry["score_ci95"] = ci(rp)
            if name in PACKAGES:
                reps_p[name] = rp
                reps_v[name] = replicates([by_seed[s] for s in seeds], "at_fault_vru", scenes, counts, rng)
            packages[name] = {**packages.get(name, {}), **entry}
        else:
            packages.setdefault(name, {})["no_band"] = {k: entry[k] for k in
                                                        ("score", "at_fault_rate", "at_fault_vru_rate", "progress")}

    decision = None
    if reps_p:
        latency = json.loads(LATENCY.read_text(encoding="utf-8"))["packages"] if LATENCY.exists() else None
        decision = decide({n: packages[n]["score"] for n in reps_p}, reps_p,
                          {n: packages[n]["at_fault_vru_rate"] for n in reps_v}, reps_v, latency)
        for s in decision["steps"]:
            print(" ->", s)

    payload = {"rule_doc": RULE_DOC, "rule_commit": rule_commit(), "split": args.split,
               "settings": {"episodes_per_scene": len(EP_STARTS) * len(PERSONAS), "episode_s": EP_LEN * DT,
                            "replan_s": REPLAN * DT, "horizon_s": H_FRAMES * DT,
                            "safety_buffer_m": SAFETY_BUFFER,
                            "personas_min_speed": PERSONAS, "accels": ACCELS, "lateral_targets": LAT_TARGETS,
                            "score_weights": SCORE_W, "n_boot": args.n_boot, "seeds": SEEDS},
               "n_scenes": len(scene_ids), "packages": packages, "decision": decision,
               "when": datetime.now().isoformat(timespec="seconds"),
               "note": "Non-reactive closed loop: agents replay their logs. Footprints are nominal "
                       "per class (no box sizes in the data)."}
    out = Path(args.save_json) if args.save_json else (
        OUT_JSON if args.split == "test" and not args.references_only else CL_DIR / f"closed_loop_{args.split}_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
