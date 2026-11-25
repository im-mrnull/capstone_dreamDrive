"""With vs without our planner on the same Indian scene: closed-loop numbers and CARLA demo packs (development data).

"With our planner": the smarter planner (scripts/proposal_planner.py, the tuned settings of
outputs/benchmark_v2/headroom_matched_tune.json, band x 0 as in the tuned runs) driving on ST-GAT's predictions
(benchmark-v2 seed 0, trained on IDD-3D), exactly as scripts/viz_closed_loop_pack.py drives.
"Without planner": closed_loop_sim's reference "No prediction (ignore agents)": the same car on the same route
towards the same reference speed (the human driver's), but it does not see or predict any other road user.
LED's 10 sampled futures (benchmark-v2 seed 0, IDD-3D) are drawn for nearby road users; they are display only:
the planner's decisions use ST-GAT, the configuration the closed-loop numbers come from.

Scenes: IDD-3D development scenes (train + val) that pass the ego-motion quality gate (scripts/ego_quality.py,
outputs/data_quality/ego_quality_dev.json). Predictions: val scenes use the benchmark-v2 model's saved dense-window
predictions (outputs/closed_loop/preds_val); train scenes use the cross-validation model whose held-out fold contains
the scene (outputs/closed_loop/cv/fold<k>/preds), so every prediction is out of sample. Rows of frozen-test scenes in
those files are dropped on load; no test-split scene is loaded. Development results, not a confirmed result.

Why the planner reacts (the interaction panel): at every replan the planner is re-run with no road users and with
each nearby road user alone; the road user whose presence alone lowers the command most (by > 0.25 m/s^2) is the one
it is reacting to. This is the planner's own decision, not a separate rule.

Usage:
    python scripts/viz_planner_compare.py screen --persona assertive   # both drives on every clean dev scene
    python scripts/viz_planner_compare.py episodes                     # 5 s episode protocol, both, clean dev scenes
    python scripts/viz_planner_compare.py pack 20220118112945_seq_26   # the two replay packs for one scene
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop_sim as cl                                              # noqa: E402
import viz_closed_loop_pack as vp                                         # noqa: E402
import proposal_planner as pp                                             # noqa: E402
from dd_split import load_splits                                          # noqa: E402

REPO = cl.REPO
OUT = REPO / "outputs" / "viz" / "planner_compare"
PACKS = vp.OUT
PRED_VAL = cl.CL_DIR / "preds_val" / "seed0"
QUALITY = REPO / "outputs" / "data_quality" / "ego_quality_dev.json"
OURS, NONE, LED = "ST-GAT (MHA)", "No prediction (ignore agents)", "LED set (10 samples)"
FOCUS_DROP = 0.25          # m/s^2: a road user "causes" the reaction if alone it lowers the command this much
FOCUS_RADIUS = 40.0        # m: road users considered for the explanation
HOLD_CLEAR = 30            # frames (3 s) the "safe distance now" message may follow a reaction
LABEL = {"Car": "Car", "TourCar": "Car", "PoliceCar": "Police car", "Van": "Van", "Truck": "Truck", "Bus": "Bus",
         "MotorcyleRider": "Motorcycle", "Motorcycle": "Motorcycle", "ScooterRider": "Scooter", "Scooter": "Scooter",
         "BicycleRider": "Bicycle", "Bicycle": "Bicycle", "BicycleGroup": "Cyclists", "Pedestrian": "Pedestrian",
         "Child": "Child", "Animal": "Animal"}


def label(cls: str) -> str:
    return LABEL.get(cls, re.sub(r"(?<!^)([A-Z])", r" \1", cls).capitalize())


class ValPredictor(cl.Predictor):
    """closed_loop_sim.Predictor reading only the val-scene dense-window predictions (no test-split file), with a
    zero band (the tuned planner runs at band x 0, so the band never enters a decision)."""

    def __init__(self, name: str, seed: int = 0, fold: int | None = None):
        self.name, self.seed, self.fold = name, seed, fold
        self.kind = "ref" if name in cl.REFERENCES else cl.ALL_PACKAGES[name][0]
        zero = (np.zeros(60), {})
        self.cv_band, self.band, self.rows = zero, zero, {}
        if self.kind == "ref":
            return
        kind, key, _ = cl.ALL_PACKAGES[name]
        src = PRED_VAL.parent if fold is None else cl.CV_PRED_DIR / f"fold{fold}" / "preds"
        z = np.load(src / f"seed{seed}" / f"{key}.npz", allow_pickle=True)
        sid = np.asarray(z["scene_id"]).astype(str)
        keep = np.array([SPLIT.get(x) in ("train", "val") for x in sid])          # frozen-test rows dropped
        self.p = (z["samples"] if kind == "set" else z["pred"][:, None])[keep].astype(np.float64)
        self.rows = {(str(s_), int(t0) + 19, int(a)): i
                     for i, (s_, t0, a) in enumerate(zip(sid[keep], z["t0"][keep], z["agent_id"][keep]))}

    def has(self, sc: dict, t: int, a: int) -> int | None:
        return self.rows.get((sc["sid"], t, int(sc["agent_id"][a])))


SPLIT = load_splits(REPO / "splits" / "scene_splits.json")


def fold_of(sid: str) -> int:
    """The cross-validation fold whose held-out part contains sid (its model never trained on sid)."""
    for k in range(5):
        f = json.loads((REPO / "splits" / "cv_folds" / f"fold{k}.json").read_text(encoding="utf-8"))
        if sid in f["splits"]["test"]:
            assert sid not in f["splits"]["train"] and sid not in f["splits"]["val"]
            return k
    raise KeyError(sid)


_PRED = {}


def predictor(name: str, sid: str) -> ValPredictor:
    """Val scene: the benchmark-v2 model (preds_val). Train scene: the CV model that held it out."""
    assert SPLIT.get(sid) in ("train", "val"), f"{sid}: not a development scene"
    fold = None if SPLIT[sid] == "val" or name in cl.REFERENCES else fold_of(sid)
    if (name, fold) not in _PRED:
        _PRED[(name, fold)] = ValPredictor(name, fold=fold)
    return _PRED[(name, fold)]


def val_scenes() -> list[str]:
    bench = load_splits(REPO / "splits" / "scene_splits.json")
    z = np.load(PRED_VAL / "stgat.npz", allow_pickle=True)
    ids = sorted(set(str(s) for s in z["scene_id"]))
    assert all(bench.get(s) == "val" for s in ids), "preds_val holds a non-val scene"
    return ids


def clean_scenes(splits=("val", "train")) -> list[str]:
    """Development scenes passing the ego-motion quality gate, val first."""
    rows = json.loads(QUALITY.read_text(encoding="utf-8"))["rows"]
    ok = {r["scene"] for r in rows if r["clean"]}
    return [s for sp in splits for s in sorted(ok) if SPLIT.get(s) == sp]


def load_scene(sid: str) -> dict:
    """closed_loop_sim.load_scene with the demo route rule (rebuild_route). Only scenes passing the ego-motion quality
    gate: on 2026-11-24 three val scenes looked like a flipped ego yaw; on 2026-11-25 that turned out to be the
    ego-motion estimate running backwards (locked onto moving traffic: pedestrians move at ~9 m/s there), which no
    heading fix repairs (docs/wiki/chaos-gap.md, "Ego-motion estimate reversed")."""
    assert sid in clean_scenes(), f"{sid}: fails the ego-motion quality gate (scripts/ego_quality.py)"
    sc = cl.load_scene(sid)
    rebuild_route(sc)
    return sc


CRAWL_M, EXT_TOL_DEG = 5.0, 10.0
TAN_MAX = float(np.tan(np.radians(30.0)))   # sideways step <= tan(30 deg) x forward step: no sliding at standstill


def rollout_nh(s0, d0, v0, a, d_target, n_frames):
    """closed_loop_sim.rollout with the lateral move limited by forward motion (a car cannot move sideways without
    moving forward); identical when the car is fast enough (|rate| <= TAN_MAX x speed)."""
    s, d, v = np.empty(n_frames), np.empty(n_frames), np.empty(n_frames)
    rate = abs(d_target - d0) / cl.LAT_TIME
    cs, cd, cv = s0, d0, v0
    for k in range(n_frames):
        nv = min(max(cv + a * cl.DT, 0.0), cl.V_MAX)
        ds = 0.5 * (cv + nv) * cl.DT
        cs += ds
        cd += np.sign(d_target - cd) * min(rate * cl.DT, TAN_MAX * ds, abs(d_target - cd))
        cv = nv
        s[k], d[k], v[k] = cs, cd, cv
    return s, d, v


def idm_roll_nh(s0, d0, v0, G, DT_, tracks, k0: int = 0, n: int = cl.H_FRAMES):
    """proposal_planner.idm_roll with the same forward-motion limit on the lateral move."""
    a_max, b, T, gap0 = pp.IDM_EGO
    C = len(G)
    s, d, v = np.broadcast_to(s0, C).astype(float), np.broadcast_to(d0, C).astype(float), np.broadcast_to(v0, C).astype(float)
    rate = np.abs(DT_ - d) / cl.LAT_TIME
    S, Dd, V, A0 = np.empty((C, n)), np.empty((C, n)), np.empty((C, n)), None
    for k in range(k0, k0 + n):
        free = np.maximum(a_max * (1.0 - (v / np.maximum(G, 0.1)) ** 4), -b)
        acc = free
        if tracks is not None:
            ts, td, tl, tw = tracks
            sa, da = ts[:, k], td[:, k]
            vs = (ts[:, k + 1] - sa) / cl.DT
            inlane = np.abs(da[None] - d[:, None]) < cl.EGO[1] + tw[None] + pp.LANE_MARGIN
            gap = sa[None] - s[:, None] - pp.EGO_FRONT - tl[None]
            ahead = inlane & (sa[None] > s[:, None]) & np.isfinite(gap)
            gap = np.where(ahead, gap, np.inf)
            j = np.argmin(gap, axis=1)
            g = gap[np.arange(C), j]
            vl = np.where(np.isfinite(g), np.nan_to_num(vs[j]), v)
            star = gap0 + v * T + v * (v - vl) / (2.0 * np.sqrt(a_max * b))
            follow = np.where(g > 0.1, (np.maximum(star, 0.0) / np.maximum(g, 0.1)) ** 2, np.inf)
            acc = np.where(np.isfinite(g), free - a_max * follow, acc)
        acc = np.clip(np.nan_to_num(acc, nan=pp.A_MIN, neginf=pp.A_MIN), pp.A_MIN, pp.A_MAX)
        if A0 is None:
            A0 = acc.copy()
        nv = np.clip(v + acc * cl.DT, 0.0, cl.V_MAX)
        ds = 0.5 * (v + nv) * cl.DT
        s = s + ds
        d = d + np.sign(DT_ - d) * np.minimum(np.minimum(rate * cl.DT, TAN_MAX * ds), np.abs(DT_ - d))
        v = nv
        S[:, k - k0], Dd[:, k - k0], V[:, k - k0] = s, d, v
    return A0, S, Dd, V


def install() -> tuple[dict, float]:
    """The tuned planner (viz_closed_loop_pack.install_planner) on the kinematically limited ego (demo variant)."""
    cfg, lam = vp.install_planner()
    cl.rollout, pp.idm_roll = rollout_nh, idm_roll_nh
    return cfg, lam


def rebuild_route(sc: dict) -> None:
    """The route our car follows must point where a car can go. closed_loop_sim builds it from the logged path and
    extends it 80 m along the direction of the path's last 12 points; when the logged car crawled (103308_seq_5:
    2.4 m in 10 s) that direction is position jitter, 45 deg off the car's heading on the extension, so the car
    slides sideways. Rule: if the logged path is shorter than CRAWL_M, the route is a straight line along the mean
    logged heading; else, if the extension direction differs from the final logged heading by more than EXT_TOL_DEG,
    the extension follows that heading. Otherwise the route is unchanged. sc["route_rebuilt"] records it."""
    sc["route_rebuilt"] = None
    s_h, head = sc["s_h"], sc["ego_head_raw"]
    if s_h[-1] < 2.0:                                   # closed_loop_sim already builds it along the heading
        return
    P = cl.smooth(cl.medfilt(sc["ego_raw"]))
    if s_h[-1] < CRAWL_M:
        u = head.mean(axis=0)
        u = u / np.linalg.norm(u)
        pts = np.stack([P[0], P[0] + (s_h[-1] + cl.ROUTE_EXT) * u])
        sc["route_rebuilt"] = "crawl: straight along the logged heading"
    else:
        step = np.linalg.norm(np.diff(P, axis=0), axis=1)
        pts = P[np.concatenate([[True], step > 0.02])]
        tail = pts[-1] - pts[max(0, len(pts) - 12)]
        ut, ue = tail / max(np.linalg.norm(tail), 1e-9), head[-1] / np.linalg.norm(head[-1])
        if np.degrees(np.arccos(np.clip(ut @ ue, -1.0, 1.0))) <= EXT_TOL_DEG:
            return
        pts = np.vstack([pts, pts[-1] + cl.ROUTE_EXT * ue])
        sc["route_rebuilt"] = "extension along the final logged heading"
    route = cl.Route(pts)
    route.set_heading(s_h, head)
    sc["route"] = route
    sc["heading"] = cl.agent_headings(sc["pos"], sc["mask"], route)    # never-moving road users face the route


# ---------------------------------------------------------------- explaining the planner's reaction
class Explainer:
    """on_replan hook for viz_closed_loop_pack.drive: which road user the planner is reacting to."""

    def __init__(self, sc: dict):
        self.sc, self.rec = sc, {}

    def __call__(self, f, s, d, v, v_ref, preds, a_cmd, d_target):
        sc, route = self.sc, self.sc["route"]
        xy = route.pose(np.array(s), np.array(d))[0]
        a_free, d_free = cl.plan(sc, route, s, d, v, v_ref, [])
        best, score_best, info = None, FOCUS_DROP, (0.0, False)
        for pred in preds:
            a = int(pred[0])
            if np.linalg.norm(sc["pos_now"][a] - xy) > FOCUS_RADIUS:
                continue
            aj, dj = cl.plan(sc, route, s, d, v, v_ref, [pred])
            side = abs(dj - d_free) > 0.05                                  # alone, it makes the planner sidestep
            score = (a_free - aj) + (2.0 if side else 0.0)
            if score > score_best:
                best, score_best, info = a, score, (float(a_free - aj), bool(side))
        self.rec[f] = {"a": float(a_cmd), "d_target": float(d_target), "a_free": float(a_free),
                       "focus": best, "drop_a": info[0], "side": info[1]}

    def active(self) -> dict:
        """Replan frame -> the road user the planner is reacting to, with hysteresis for the display: it alone lowers
        the command by >= 1 m/s^2 or makes the planner sidestep, or it is the focus in 2 of the last 3 replans."""
        reps, out = sorted(self.rec), {}
        for i, r in enumerate(reps):
            c, x = self.rec[r]["focus"], self.rec[r]
            recent = [self.rec[q]["focus"] for q in reps[max(0, i - 2):i + 1]]
            out[r] = c if c is not None and (x["drop_a"] >= 1.0 or x["side"] or recent.count(c) >= 2) else None
        for i in range(1, len(reps) - 1):                     # a single 0.2 s gap inside one reaction is filled
            if out[reps[i]] is None and out[reps[i - 1]] is not None and out[reps[i - 1]] == out[reps[i + 1]]:
                out[reps[i]] = out[reps[i - 1]]
        return out


def ego_gap(sc, xy, tan, a, t) -> float:
    ego = cl.rect(xy + cl.EGO_CENTRE * tan, tan, cl.EGO_RECT)
    return float(cl.rect_gap(ego, cl.rect(sc["pos"][t, a], sc["heading"][t, a], cl.rect_size(sc["cls"][a]))))


def relation(xy, tan, p) -> tuple[str, float, float]:
    rel = p - xy
    lon, lat = float(rel @ tan), float(rel @ np.array([-tan[1], tan[0]]))
    side = "on the left" if lat > 2.0 else "on the right" if lat < -2.0 else ""
    where = "ahead" if lon > 2.0 else "behind" if lon < -2.0 else "beside us"
    return (f"{where} {side}".strip() if where != "beside us" else f"beside us {side}".strip()), lon, lat


def state_of(a_s: float, v: float) -> tuple[str, str]:
    if v < 0.3 and a_s <= 0.2:
        return "STOPPED", "amber"
    if a_s <= -2.0:
        return "BRAKING", "red"
    if a_s <= -0.4:
        return "SLOWING DOWN", "amber"
    if a_s < 0.4:
        return "HOLDING SPEED", "blue"
    return "ACCELERATING", "green"


def hud_ours(sc, res, ex: Explainer) -> tuple[list, np.ndarray, list, list]:
    """Per frame: panel text, the road user to highlight, its highlight kind and tag."""
    T, rec, active = sc["T"], ex.rec, ex.active()
    hud, focus, kind, tag = [None] * T, np.full(T, -1, np.int32), [""] * T, [""] * T
    hit = next(((g, a, fault) for g, a, fault in res["contacts"]), None)     # first contact of any kind
    last = None                                                  # (agent, frame, sidestep) of the last reaction
    gaps = {}
    for f in range(vp.T_START, res["t_end"] + 1):
        r = max(k for k in rec if k <= f) if any(k <= f for k in rec) else None
        xy, h = res["ego"][f], res["head"][f]
        tan = np.array([np.cos(h), np.sin(h)])
        v = float(res["speed"][f])
        acc = res["accel"][max(vp.T_START, f - 4):f + 1]
        a_s = float(np.nanmean(acc)) if np.isfinite(acc).any() else 0.0
        a_now = float(res["accel"][f]) if np.isfinite(res["accel"][f]) else a_s
        st, col = state_of(min(a_s, a_now), v)       # braking shows at once; accelerating once it is sustained
        line2, line3 = "Road ahead clear - driving at traffic speed", ""
        a_f = active.get(r) if r is not None else None
        if a_f is not None and sc["mask"][f, a_f]:
            last = (a_f, f, rec[r]["side"] and rec[r]["focus"] == a_f)
        elif last is not None and (f - last[1] > HOLD_CLEAR or not sc["mask"][f, last[0]]):
            last = None
        if last is not None:
            a, side = last[0], last[2]
            g = ego_gap(sc, xy, tan, a, f)
            gaps.setdefault(a, {})[f] = g
            prev = gaps[a].get(f - 2)
            closing = (prev - g) / (2 * cl.DT) if prev is not None else 0.0
            where, lon, _ = relation(xy, tan, sc["pos"][f, a])
            name, dist = label(sc["cls"][a]), f"{max(g, 0.0):.1f} m"
            ahead = lon > 2.0
            aside = "moving aside to pass it safely" if ahead else "moving aside to let it pass"
            if a_f is not None and st != "ACCELERATING":           # reacting now
                focus[f], kind[f], tag[f] = a, "watch", dist
                coming = where.replace("behind", "from behind").replace("beside us", "alongside us")
                line2 = f"{name} approaching {coming} - distance {dist}" if closing > 0.3 else \
                    f"{name} {where} - distance {dist}"
                line3 = {"BRAKING": "braking to keep a safe gap", "SLOWING DOWN": "slowing down to keep a safe gap",
                         "STOPPED": "waiting until it is safe",
                         "HOLDING SPEED": aside if side else "holding speed, not closing in"}[st]
            elif st == "ACCELERATING":                            # its gap is safe again
                focus[f], kind[f], tag[f] = a, "clear", dist
                if g >= 2.0:
                    line2 = f"Safe distance from {name.lower()} ({dist})"
                    line3 = "can accelerate now" + (" - " + aside.replace(" safely", "") if a_f is not None and side else "")
                else:                                             # close but not converging: no "safe distance" claim
                    line2 = f"{name} {where} - {dist}"
                    line3 = "no conflict predicted - can accelerate"
            else:
                line2 = f"{name} {where} - {dist}"
                line3 = "no conflict predicted any more"
        if hit is not None and f >= hit[0]:
            name = label(sc["cls"][hit[1]]).lower()
            if hit[2]:
                st, col = "COLLISION", "red"
                line2, line3 = f"hit a {name} - our car at fault", "the planner did not avoid it"
            else:                                         # the at-fault rule: the other road user ran into us
                st, col = "CONTACT", "amber"
                line2, line3 = f"a {name} ran into our car - not our fault", "recorded traffic does not react to us"
            focus[f], kind[f], tag[f] = hit[1], "contact", st
        hud[f] = {"header": "OUR PLANNER", "state": st, "col": col, "line2": line2, "line3": line3,
                  "speed": f"{v * 3.6:4.0f} km/h", "cmd": f"{float(res['accel'][f]) if np.isfinite(res['accel'][f]) else 0.0:+.1f} m/s2"}
    return hud, focus, kind, tag, (hit[0] if hit else None)


def hud_none(sc, res) -> tuple[list, np.ndarray, list, list, int | None]:
    """The car without a planner: the nearest road user in its path, and its first contact (at fault or not)."""
    T = sc["T"]
    hud, focus, kind, tag = [None] * T, np.full(T, -1, np.int32), [""] * T, [""] * T
    hit = next(((g, a, fault) for g, a, fault in res["contacts"]), None)
    for f in range(vp.T_START, res["t_end"] + 1):
        xy, h = res["ego"][f], res["head"][f]
        tan = np.array([np.cos(h), np.sin(h)])
        v = float(res["speed"][f])
        line2, line3, st, col = "keeps the human driver's speed", "does not look at other road users", \
            "NOT LOOKING", "grey"
        if hit is not None and f >= hit[0]:
            a = hit[1]
            if hit[2]:
                st, col = "COLLISION", "red"
                line2, line3 = f"hit a {label(sc['cls'][a]).lower()} - our car at fault", "never saw it coming"
            else:
                st, col = "CONTACT", "amber"
                line2, line3 = f"a {label(sc['cls'][a]).lower()} ran into our car - not our fault",                     "recorded traffic does not react to us"
            focus[f], kind[f], tag[f] = a, "contact", st
        else:
            best = None
            for a in np.where(sc["mask"][f])[0]:
                where, lon, lat = relation(xy, tan, sc["pos"][f, a])
                if lon < 0.0 or abs(lat) > 3.0:
                    continue
                g = ego_gap(sc, xy, tan, a, f)
                if g < 8.0 and (best is None or g < best[1]):
                    best = (a, g, where)
            if best is not None:
                a, g, where = best
                focus[f], kind[f], tag[f] = a, "threat", f"{max(g, 0.0):.1f} m"
                line2 = f"{label(sc['cls'][a])} {where} - distance {max(g, 0.0):.1f} m"
                line3 = "no reaction: does not see it"
        hud[f] = {"header": "WITHOUT PLANNER", "state": st, "col": col, "line2": line2, "line3": line3,
                  "speed": f"{v * 3.6:4.0f} km/h", "cmd": f"{float(res['accel'][f]) if np.isfinite(res['accel'][f]) else 0.0:+.1f} m/s2"}
    return hud, focus, kind, tag, (hit[0] if hit else None)


def travel_heading(res: dict) -> np.ndarray:
    """Display heading: the direction the car travels (+-1 frame), so the body turns when it moves aside; the
    route heading while it is (nearly) stopped. The simulator's footprint keeps the route heading."""
    e, h = res["ego"], res["head"].copy()
    out = h.copy()
    for f in range(len(e)):
        lo, hi = max(0, f - 1), min(len(e) - 1, f + 1)
        if np.isfinite(e[lo]).all() and np.isfinite(e[hi]).all() and np.isfinite(h[f]):
            dv = e[hi] - e[lo]
            if np.linalg.norm(dv) / ((hi - lo) * cl.DT) > 0.5:
                out[f] = np.arctan2(dv[1], dv[0])
    return out


def led_samples(sc, res, led: ValPredictor, K: int = 10) -> np.ndarray:
    """(T, A, K, 60, 2): LED's sampled futures at each replan frame (and advanced one step on the next frame), for
    road users with a full 2 s history (LED's own windows); others get none."""
    T, A = sc["T"], sc["pos"].shape[1]
    out = np.full((T, A, K, vp.PRED_LEN, 2), np.nan, np.float32)
    for f in res["seen"]:
        for a in np.where(sc["mask"][f])[0]:
            i = led.has(sc, f, a)
            if i is None:
                continue
            s = led.p[i][:K]
            out[f, a] = s
            if f + 1 < T:
                out[f + 1, a, :, :-1] = s[:, 1:]
    return out


# ---------------------------------------------------------------- commands
def screen(args) -> int:
    cfg, lam = install()
    rows = []
    for sid in (args.scenes or clean_scenes()):
        r = {"scene": sid, "split": SPLIT[sid], "model": "benchmark-v2" if SPLIT[sid] == "val" else f"cv fold {fold_of(sid)}"}
        for name, pred in (("ours", predictor(OURS, sid)), ("none", predictor(NONE, sid))):
            sc = load_scene(sid)
            ex = Explainer(sc) if name == "ours" else None
            res = vp.drive(sc, pred, lam, on_replan=ex, vmin=cl.PERSONAS[args.persona])
            m = res["metrics"]
            hits = [(g, a) for g, a, fault in res["contacts"] if fault]
            r[name] = {**m, "first_at_fault": None if not hits else
                       {"frame": int(hits[0][0]), "t_s": round((hits[0][0] - vp.T_START) * cl.DT, 1),
                        "class": str(sc["cls"][hits[0][1]])},
                       "overlaps_not_at_fault": len({a for g, a, fault in res["contacts"] if not fault})}
            if ex is not None:
                foc = [v_["focus"] for v_ in ex.rec.values() if v_["focus"] is not None]
                r[name]["reactions"] = len(foc)
                r[name]["reacted_to"] = sorted({label(str(sc["cls"][a])) for a in foc})
        rows.append(r)
        o, n = r["ours"], r["none"]
        print(f"{sid}: ours contacts {o['at_fault_contacts']} gap {o['min_clearance_m']} harsh {o['harsh_braking']} "
              f"prog {o['progress']:.2f} reactions {o['reactions']} {o['reacted_to']} | none contacts "
              f"{n['at_fault_contacts']} first {n['first_at_fault']} gap {n['min_clearance_m']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"screen_clean_{args.persona}.json").write_text(json.dumps({"planner_cfg": cfg, "band": lam, "rows": rows}, indent=1,
                                                default=str), encoding="utf-8")
    return 0


def episodes(args) -> int:
    """closed_loop_sim's 5 s episode protocol (starts 19/29/39/49, both personas) on every clean dev scene."""
    cfg, lam = install()
    out = {}
    for name in (OURS, NONE):
        eps = []
        for sid in clean_scenes():
            pred = predictor(name, sid)
            sc = load_scene(sid)
            for t0 in cl.EP_STARTS:
                if not cl.episode_ok(sc, t0):
                    continue
                for persona, vmin in cl.PERSONAS.items():
                    e = cl.run_episode(sc, t0, vmin, pred, lam)
                    e["persona"] = persona
                    eps.append(e)
        s = cl.summarise(eps)
        s["at_fault_episodes"] = int(sum(e["at_fault"] for e in eps))
        s["near_miss_episodes"] = int(sum(e["near_miss"] for e in eps))
        s["harsh_braking_episodes"] = int(sum(e["harsh_braking"] for e in eps))
        s["scenes_with_at_fault"] = sorted({e["scene"] for e in eps if e["at_fault"]})
        for sp_ in ("val", "train"):
            sub = [e for e in eps if SPLIT[e["scene"]] == sp_]
            s[f"{sp_}_at_fault"] = f"{sum(e['at_fault'] for e in sub)}/{len(sub)}"
            for p_ in cl.PERSONAS:
                q = [e for e in sub if e["persona"] == p_]
                s[f"{sp_}_{p_}_at_fault"] = f"{sum(e['at_fault'] for e in q)}/{len(q)}"
        out[name] = {"summary": s, "episodes": eps}
        print(f"{name}: {s['at_fault_episodes']}/{s['n_episodes']} at-fault, near-miss {s['near_miss_episodes']}, "
              f"harsh {s['harsh_braking_episodes']}, progress {s['progress']:.3f}, score {s['score']:.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "episodes_clean.json").write_text(json.dumps(
        {"what": "development (IDD-3D dev scenes passing the ego-motion quality gate: val with the benchmark-v2 "
                 "model, train with the CV model that held the scene out); proposal planner tuned cfg, band x " + str(lam),
         "scenes": clean_scenes(),
         "planner_cfg": cfg, "results": out}, indent=1, default=str), encoding="utf-8")
    return 0


def pack(args) -> int:
    cfg, lam = install()
    if args.scene not in clean_scenes():
        raise SystemExit(f"{args.scene}: not a development scene passing the ego-motion quality gate")
    marginal, per_class = np.zeros(60), {}
    radii = {"__marginal__": marginal, **per_class}               # no tube is drawn in this comparison
    led = predictor(LED, args.scene)
    planner = "smarter planner (PDM-style proposals)"
    split = "validation scene" if SPLIT[args.scene] == "val" else "training scene, held-out CV model"
    for name, pred in (("ours", predictor(OURS, args.scene)), ("none", predictor(NONE, args.scene))):
        sc = load_scene(args.scene)
        ex = Explainer(sc) if name == "ours" else None
        res = vp.drive(sc, pred, lam, on_replan=ex, vmin=cl.PERSONAS[args.persona])
        meta = {"source": "IDD-3D", "place": f"India (IDD-3D, {split})",
                "model": "ST-GAT + LED (IDD-3D)" if name == "ours" else "no prediction", "planner": planner}
        t_last = res["t_end"]
        if name == "ours":
            hud, focus, kind, tag, hit = hud_ours(sc, res, ex)
            samples = led_samples(sc, res, led)
            if hit is not None:
                t_last = min(res["t_end"], hit + 3)                # a collision ends the drive (+0.3 s)
        else:
            hud, focus, kind, tag, hit = hud_none(sc, res)
            samples = None
            if hit is not None:
                t_last = min(res["t_end"], hit + 3)                # the drive ends at the collision (+0.3 s)
        p = vp.pack_from(sc, {**res, "head": travel_heading(res)}, radii, meta, t_last=t_last)
        keep = slice(vp.T_START, t_last + 1)
        if samples is not None:
            p["samples"] = samples[keep]
        else:
            p["ml"][:] = np.nan                                    # it predicts nothing
        p["hud"] = json.dumps(hud[keep])
        p["focus"], p["focus_kind"] = focus[keep], np.asarray(kind[keep], dtype=object)
        p["focus_tag"] = np.asarray(tag[keep], dtype=object)
        m = json.loads(p["metrics"])
        m["caption"] = (f"closest gap {m['min_clearance_m']:.1f} m  |  " if m["min_clearance_m"] is not None else "") + \
            f"harsh braking: {'yes' if m['harsh_braking'] else 'none'}  |  at-fault collision: " + \
            ("yes" if m["at_fault_contacts"] else "none")
        p["metrics"] = json.dumps(m)
        f = PACKS / f"cmp_{name}_{args.scene}.npz"
        p["persona"] = args.persona
        np.savez_compressed(f, **p)
        print(f"{name}: {m['caption']} -> {f.name} ({len(hud[keep])} frames)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("screen")
    a.add_argument("--scenes", nargs="*", default=None)
    a.add_argument("--persona", default="nominal", choices=list(cl.PERSONAS))
    sub.add_parser("episodes")
    b = sub.add_parser("pack")
    b.add_argument("scene")
    b.add_argument("--persona", default="assertive", choices=list(cl.PERSONAS),
                   help="assertive: both cars are asked for at least 6 m/s (closed_loop_sim.PERSONAS)")
    args = ap.parse_args()
    return {"screen": screen, "episodes": episodes, "pack": pack}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
