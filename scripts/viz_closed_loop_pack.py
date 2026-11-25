"""Closed-loop replay packs for the CARLA side-by-side demo: the smarter planner drives, ST-GAT predicts.

The ego is driven by the smarter planner (scripts/proposal_planner.py: PDM-Closed style IDM proposals, Dauner et
al., CoRL 2023, with the tuned settings of outputs/benchmark_v2/headroom_matched_tune.json), replanning every
0.2 s exactly as in closed_loop_sim.run_episode; the other road users replay their logged tracks (non-reactive,
as in the closed-loop test). The planner sees ST-GAT, the most accurate single-path benchmark model:
  - IDD-3D (India): the benchmark-v2 seed-0 ST-GAT trained on IDD-3D, through its saved dense-window
    predictions on the val scenes (outputs/closed_loop/preds_val, closed_loop_sim.Predictor);
  - Argoverse 2 (US): the AV2-matched seed-0 ST-GAT (trained on Argoverse 2 only), run here at every replan frame
    on the agents with a full 2 s history.
Agents without a full 2 s history get constant velocity, as in the closed-loop test. The band setting is the
tuned planner's (band x 0, so a 0.25 m safety buffer); the drawn 90% tubes are each model's own val calibration
(IDD-3D val / AV2 val) and are for display only.

Only IDD-3D val scenes are accepted: the frozen test split is refused. Demo output, not a result.

The pack has viz_prepare.py's fields, cropped to the frames the planner drives (2 s of history first), plus:
  ego_heading (T,)       the ego's heading along its route
  ego_plan (T, 30, 2)    the ego's next 3 s at its current command
  ego_human (T, 2)       the recorded human driver
  metrics                model, planner, closest gap, harsh braking, at-fault contact, progress vs the human

Usage:
    python scripts/viz_closed_loop_pack.py --idd 20220118112945_seq_26 --av2 015e5479-1888-4d13-816c-689d9437599e
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop_sim as cl                                              # noqa: E402
import conformal_wrapper as cw                                            # noqa: E402
import dump_predictions as dp                                             # noqa: E402
import proposal_planner as pp                                             # noqa: E402
from dd_data import collate_pad                                           # noqa: E402
from dd_split import load_splits                                          # noqa: E402

REPO = cl.REPO
OUT = REPO / "outputs" / "viz" / "packs"
TUNE = REPO / "outputs" / "benchmark_v2" / "headroom_matched_tune.json"
V2_GRID = REPO / "outputs" / "benchmark_v2" / "closed_loop_dev_grid.json"
AV2_DIR = REPO / "data_external" / "av2" / "val"
STGAT_AV2 = REPO / "outputs" / "chaos_gap" / "dev" / "stgat_matched.pt"
STGAT_AV2_DUMP = REPO / "outputs" / "chaos_gap" / "dev" / "stgat_matched_eval_preds.npz"
AV2_CLASS = {"vehicle": "Car", "bus": "Bus", "pedestrian": "Pedestrian",
             "cyclist": "BicycleRider", "motorcyclist": "MotorcyleRider"}
T_START = 19                        # the first frame with a full 2 s history
PLAN_STEPS = 30                     # 3 s
PRED_LEN = 60


def install_planner() -> tuple[dict, float]:
    """The smarter planner with its tuned settings, and the band scale of the tuned runs (counterfactual_sim)."""
    cfg = json.loads(TUNE.read_text(encoding="utf-8"))["proposal"]["chosen"]
    cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items()}
    pp.use(cfg)
    lam = float(json.loads(V2_GRID.read_text(encoding="utf-8"))["chosen"]["band"])
    return cfg, lam


# ---------------------------------------------------------------- scenes
def read_av2_raw(sid: str) -> dict:
    """A raw Argoverse 2 val scenario (data_external/av2/val/<id>/): tracks, the AV, dataset yaw, map, as in
    viz_prepare.load_av2."""
    sdir = AV2_DIR / sid
    d = pd.read_parquet(sdir / f"scenario_{sid}.parquet")
    T = int(d.num_timestamps.iloc[0])
    ego = d[d.track_id == "AV"].sort_values("timestep")
    ego_xy = np.full((T, 2), np.nan)
    ego_xy[ego.timestep.values] = ego[["position_x", "position_y"]].values
    d = d[d.object_type.isin(AV2_CLASS) & (d.track_id != "AV")]
    ids = sorted(d.track_id.unique())
    ai = {a: i for i, a in enumerate(ids)}
    pos, hdg = np.full((T, len(ids), 2), np.nan), np.full((T, len(ids)), np.nan)
    pos[d.timestep.values, d.track_id.map(ai).values] = d[["position_x", "position_y"]].values
    hdg[d.timestep.values, d.track_id.map(ai).values] = d.heading.values
    cls = d.drop_duplicates("track_id").set_index("track_id").object_type.reindex(ids).map(AV2_CLASS).values
    m = json.loads((sdir / f"log_map_archive_{sid}.json").read_text(encoding="utf-8"))
    lanes = [np.array([[p["x"], p["y"]] for p in seg[f"{side}_lane_boundary"]])
             for seg in m["lane_segments"].values() for side in ("left", "right")]
    marks = [str(seg.get(f"{side}_lane_mark_type", "NONE")) for seg in m["lane_segments"].values() for side in ("left", "right")]
    crossings = [np.array([[p["x"], p["y"]] for p in c["edge1"]] + [[p["x"], p["y"]] for p in c["edge2"]][::-1])
                 for c in m["pedestrian_crossings"].values()]
    city = str(pd.read_parquet(sdir / f"scenario_{sid}.parquet", columns=["city"]).city.iloc[0])
    return {"name": sid, "pos": pos, "ego": ego_xy, "heading": hdg, "classes": cls, "ids": np.asarray(ids, dtype=object),
            "lanes": lanes, "lane_marks": marks, "crossings": crossings, "place": f"{city.title()}, USA (Argoverse 2)"}


def load_av2_scene(key: str) -> tuple[dict, list, list, str]:
    """An Argoverse 2 scenario in closed_loop_sim's scene layout: a raw val scenario id, or a viz_prepare.py pack
    stem (av2_<id8>). The route, heading and reference speed are built from the AV's own track exactly as
    closed_loop_sim.load_scene builds them from the logged ego."""
    pk = read_av2_raw(key) if (AV2_DIR / key).is_dir() else dict(np.load(OUT / f"{key}.npz", allow_pickle=True))
    P0 = np.asarray(pk["pos"], np.float64)
    T = P0.shape[0]
    mask = np.isfinite(P0).all(axis=2)
    pos = np.where(mask[..., None], P0, 0.0)
    raw = np.asarray(pk["ego"], np.float64)
    assert np.isfinite(raw).all(), "the AV must be present at every timestep"
    med = cl.medfilt(raw)
    glitch = np.linalg.norm(raw - med, axis=1) > cl.GLITCH_M
    P = cl.smooth(med)
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    keep = np.concatenate([[True], step > 0.02])
    pts = P[keep]
    tail = pts[-1] - pts[max(0, len(pts) - 12)]
    u = tail / max(np.linalg.norm(tail), 1e-9)
    pts = np.vstack([pts, pts[-1] + cl.ROUTE_EXT * u])
    s_h = np.concatenate([[0.0], np.cumsum(step)])
    route = cl.Route(pts)
    # the AV's heading: its smoothed direction of motion along the route (held while slow)
    yaw_head = np.stack([np.interp(s_h, route.s_grid, route.tan[:, j]) for j in range(2)], 1)
    yaw_head /= np.maximum(np.linalg.norm(yaw_head, axis=1, keepdims=True), 1e-9)
    route.set_heading(s_h, yaw_head)
    v_h = cl.smooth(np.clip(np.gradient(s_h) / cl.DT, 0.0, cl.V_MAX)[:, None])[:, 0]
    hd = np.asarray(pk["heading"], np.float64)                      # dataset yaw, NaN where absent
    H = np.stack([np.cos(hd), np.sin(hd)], -1)
    H = np.where(mask[..., None] & np.isfinite(H), H, np.nan)
    fallback = cl.agent_headings(pos, mask, route)             # held / route tangent where yaw is missing
    H = np.where(np.isfinite(H), H, fallback)
    sc = {"sid": str(pk["name"]), "pos": pos, "mask": mask, "cls": np.asarray(pk["classes"], dtype=str),
          "agent_id": np.arange(P0.shape[1]), "ego_raw": raw, "glitch": glitch, "ego_head_raw": yaw_head,
          "heading": H, "route": route, "s_h": s_h, "v_h": v_h, "T": T, "names": pk["ids"]}
    lanes = list(zip([np.asarray(l, float) for l in pk["lanes"]], [str(m) for m in pk["lane_marks"]]))
    crossings = [np.asarray(c, float) for c in pk["crossings"]]
    return sc, lanes, crossings, str(pk["place"])


# ---------------------------------------------------------------- predictors
def av2_band(alpha: float = 0.1) -> np.ndarray:
    """90% per-step radii of the AV2-matched ST-GAT on its AV2 val rows (display only)."""
    p = np.load(STGAT_AV2_DUMP, allow_pickle=True)
    v = p["split"].astype(object) == "val"                    # av2_matched_dev.json: val = AV2 val, test = IDD val
    s, valid = cw.scores_and_valid(p["pred"][v], p["gt"][v], p["fut_mask"][v])
    return cw.radii_per_timestep(s, valid, alpha)


class AV2STGAT:
    """closed_loop_sim.Predictor's interface for the AV2-trained ST-GAT, run on the scene at frame t."""

    name = "ST-GAT (trained on Argoverse 2)"

    def __init__(self, device: str = "cpu"):
        self.fn = dp.load_stgat(torch.device(device), ckpt=str(STGAT_AV2))
        self.marginal = av2_band()

    @torch.no_grad()
    def predict(self, sc: dict, t: int) -> dict:
        if t < T_START:
            return {}
        full = [a for a in range(sc["pos"].shape[1]) if sc["mask"][t - T_START:t + 1, a].all()]
        if not full:
            return {}
        obs = sc["pos"][t - T_START:t + 1, full].transpose(1, 2, 0).astype(np.float32)     # (M, 2, 20)
        rel = np.zeros_like(obs)
        rel[:, :, 1:] = np.diff(obs, axis=2)
        M = len(full)
        item = (obs, np.zeros((M, 2, PRED_LEN), np.float32), rel, np.zeros((M, 2, PRED_LEN), np.float32),
                np.zeros((M, PRED_LEN), bool), np.asarray(sc["cls"][full], dtype=object))
        out = self.fn(collate_pad([item]))[0].permute(0, 2, 1).cpu().numpy()             # (M, 60, 2)
        return {a: out[i].astype(np.float64) for i, a in enumerate(full)}

    def at(self, sc: dict, t: int, lam: float = 1.0) -> list:
        P = self.predict(sc, t)
        band = lam * self.marginal
        return [(a, (P[a] if a in P else cl.cv_extrapolate(sc, t, a))[None], band) for a in np.where(sc["mask"][t])[0]]


# ---------------------------------------------------------------- driving
def pose_c(route, s: float, d: float):
    """route.pose without the 0.25 m grid snap (linear between grid points), for drawing: at walking pace the snapped
    pose advances in 0.25 m jumps and, between them, only sideways. The simulator's checks keep route.pose."""
    x = float(np.clip(s / 0.25, 0, len(route.s_grid) - 1))
    i = min(int(x), len(route.s_grid) - 2)
    w = x - i
    xy = (1 - w) * route.xy[i] + w * route.xy[i + 1]
    nrm = (1 - w) * route.nrm[i] + w * route.nrm[i + 1]
    hd = (1 - w) * route.head[i] + w * route.head[i + 1]
    return xy + d * nrm / max(np.linalg.norm(nrm), 1e-9), hd / max(np.linalg.norm(hd), 1e-9)


def drive(sc: dict, predictor, lam: float, on_replan=None, vmin: float = 0.0) -> dict:
    """closed_loop_sim.run_episode's planner branch from T_START to the end of the scene (nominal persona),
    recording the ego's pose, its 3 s plan and the predictions the planner saw at each replan.
    on_replan(f, s, d, v, v_ref, preds, a_cmd, d_target), if given, is called after each replan with the planner's
    inputs still set in sc (for explaining the decision; it must not change sc's planner state).
    vmin: the persona's minimum reference speed (closed_loop_sim.PERSONAS; 0 = nominal)."""
    route, T = sc["route"], sc["T"]
    t_end = T - 2
    s, d, v = sc["s_h"][T_START], 0.0, sc["v_h"][T_START]
    a_cmd, d_target = 0.0, 0.0
    ego, head = np.full((T, 2), np.nan), np.full(T, np.nan)
    plan = np.full((T, PLAN_STEPS, 2), np.nan)
    speed, accel = np.full(T, np.nan), np.full(T, np.nan)
    xy0, tan0 = pose_c(route, s, 0.0)                                  # the start pose on the route
    ego[T_START], head[T_START] = xy0, np.arctan2(tan0[1], tan0[0])
    speed[T_START] = v
    seen, harsh, min_clear, hits, n_fb, cmds = {}, False, np.inf, set(), 0, []
    contacts = []                                   # (frame, agent, at fault) for every footprint overlap
    for f in range(T_START, t_end):
        d_prev = d
        if (f - T_START) % cl.REPLAN == 0:
            sc["pos_now"], sc["head_now"] = sc["pos"][f], sc["heading"][f]
            v_ref = max(sc["v_h"][min(f + 30, T - 1)], vmin)
            sc["cmd_prev"] = (a_cmd, d_target)
            sc.pop("plan_fallback", None)
            preds = predictor.at(sc, f, lam)
            seen[f] = preds
            a_cmd, d_target = cl.plan(sc, route, s, d, v, v_ref, preds)
            n_fb += int(bool(sc.pop("plan_fallback", False)))
            cmds.append(float(a_cmd))
            harsh |= a_cmd <= cl.HARSH_DECEL
            if on_replan is not None:
                on_replan(f, s, d, v, v_ref, preds, a_cmd, d_target)
                sc.pop("plan_fallback", None)
        accel[f] = a_cmd
        S, D, _ = cl.rollout(s, d, v, a_cmd, d_target, PLAN_STEPS)
        plan[f] = route.pose(S, D)[0]
        (s,), (d,), (v,) = cl.rollout(s, d, v, a_cmd, d_target, 1)
        g = f + 1
        xy, tan = route.pose(np.array(s), np.array(d))
        xy_c, tan_c = pose_c(route, float(s), float(d))                # drawn pose (no grid snap)
        ego[g], head[g], speed[g] = xy_c, np.arctan2(tan_c[1], tan_c[0]), v
        sc["route_tan_now"] = tan
        box = cl.rect(xy + cl.EGO_CENTRE * tan, tan, cl.EGO_RECT)
        nrm = np.array([-tan[1], tan[0]])
        ddot = (d - d_prev) / cl.DT
        for a in np.where(sc["mask"][g])[0]:
            size = cl.rect_size(sc["cls"][a])
            rel = sc["pos"][g, a] - xy
            if rel @ rel > 400.0:
                continue
            gap = cl.rect_gap(box, cl.rect(sc["pos"][g, a], sc["heading"][g, a], size))
            lon, lat = float(rel @ tan), float(rel @ nrm)
            if lon >= cl.AT_FAULT_REAR_M:
                min_clear = min(min_clear, gap)
            if gap < -cl.COLLISION_TOL:
                fault = bool(v >= cl.AT_FAULT_MIN_SPEED and cl.at_fault(sc, g, a, lon, lat, ddot, size))
                contacts.append((g, int(a), fault))
                if fault and a not in hits:
                    hits.add(int(a))
    prog_h = sc["s_h"][t_end] - sc["s_h"][T_START]
    progress = 1.0 if prog_h < 2.0 else min(1.0, (s - sc["s_h"][T_START]) / prog_h)
    return {"ego": ego, "head": head, "plan": plan, "seen": seen, "t_end": t_end, "speed": speed, "accel": accel,
            "contacts": contacts,
            "metrics": {"min_clearance_m": float(min_clear) if np.isfinite(min_clear) else None,
                        "harsh_braking": bool(harsh), "at_fault_contacts": len(hits), "progress": float(progress),
                        "n_replans": len(cmds), "n_no_clear_manoeuvre": n_fb,
                        "seconds_driven": (t_end - T_START) * cl.DT}}


def overlay_predictions(sc: dict, seen: dict) -> np.ndarray:
    """(T, A, 60, 2): at a replan frame the predictions the planner used; on the frame after it the same
    predictions advanced one step (so the drawn path always starts at the agent's current time)."""
    T, A = sc["T"], sc["pos"].shape[1]
    ml = np.full((T, A, PRED_LEN, 2), np.nan, np.float32)
    for f, preds in seen.items():
        for a, samples, _ in preds:
            ml[f, a] = samples[0]
            if f + 1 < T:
                ml[f + 1, a, :-1] = samples[0][1:]
    return ml


def build(sc: dict, predictor, lam: float, radii: dict, meta: dict, lanes=(), crossings=()) -> dict:
    return pack_from(sc, drive(sc, predictor, lam), radii, meta, lanes, crossings)


def pack_from(sc: dict, res: dict, radii: dict, meta: dict, lanes=(), crossings=(), t_last: int | None = None) -> dict:
    """The replay pack of a drive (frames T_START..t_last, default the drive's end)."""
    ml = overlay_predictions(sc, res["seen"])
    keep = slice(T_START, (res["t_end"] if t_last is None else t_last) + 1)
    pos = np.where(sc["mask"][..., None], sc["pos"], np.nan)
    heading = np.arctan2(sc["heading"][..., 1], sc["heading"][..., 0])
    ego = res["ego"][keep]
    m = {**res["metrics"], **{k: meta[k] for k in ("model", "planner")}}
    gap = m["min_clearance_m"]
    m["caption"] = (f"{meta['model']}  |  closest gap {gap:.1f} m" if gap is not None else meta["model"]) + \
        f"  |  harsh braking: {'yes' if m['harsh_braking'] else 'none'}  |  at-fault contact: " + \
        (str(m["at_fault_contacts"]) if m["at_fault_contacts"] else "none")
    names = sc.get("names", sc["agent_id"])
    T = ego.shape[0]
    return {
        "name": sc["sid"], "source": meta["source"], "place": meta["place"],
        "center": np.nanmean(ego, axis=0), "pos": pos[keep].astype(np.float32),
        "ids": np.asarray([str(x) for x in names], dtype=object), "classes": np.asarray(sc["cls"], dtype=object),
        "heading": heading[keep].astype(np.float32), "ego": ego.astype(np.float32),
        "ego_heading": res["head"][keep].astype(np.float32), "ego_plan": res["plan"][keep].astype(np.float32),
        "ego_human": sc["ego_raw"][keep].astype(np.float32),
        "ml": ml[keep], "samples": np.full((T, pos.shape[1], 0, PRED_LEN, 2), np.nan, np.float32),
        "radii_classes": np.asarray(list(radii), dtype=object),
        "radii": np.stack([radii[k] for k in radii]).astype(np.float32),
        "lanes": np.asarray([l for l, _ in lanes] + [None], dtype=object)[:-1],
        "lane_marks": np.asarray([mk for _, mk in lanes], dtype=object),
        "crossings": np.asarray(list(crossings) + [None], dtype=object)[:-1],
        "metrics": json.dumps(m),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idd", nargs="*", default=[])
    ap.add_argument("--av2", nargs="*", default=[])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    cfg, lam = install_planner()
    planner = "smarter planner (PDM-style proposals)"
    bench = load_splits(REPO / "splits" / "scene_splits.json")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.idd:
        pred = cl.Predictor("ST-GAT (MHA)", 0)
        marginal, per_class = cl.band_table("single", "preds_stgat.npz", 0)
        radii = {"__marginal__": marginal, **per_class}
        for sid in args.idd:
            if bench.get(sid) != "val":
                raise SystemExit(f"{sid} is in the {bench.get(sid)!r} split; only IDD-3D val scenes are used")
            sc = cl.load_scene(sid)
            p = build(sc, pred, lam, radii, {"source": "IDD-3D", "place": "India (IDD-3D, validation scene)",
                                             "model": "ST-GAT trained on IDD-3D", "planner": planner})
            f = out / f"cl_idd_{sid}.npz"
            np.savez_compressed(f, **p)
            print(f"IDD-3D {sid}: {json.loads(p['metrics'])['caption']} -> {f.name}", flush=True)
    if args.av2:
        pred = AV2STGAT()
        radii = {"__marginal__": pred.marginal}
        for stem in args.av2:                                  # raw val scenario ids or viz_prepare pack stems
            sc, lanes, crossings, place = load_av2_scene(stem)
            stem = stem if stem.startswith("av2_") else f"av2_{stem[:8]}"
            p = build(sc, pred, lam, radii, {"source": "Argoverse 2", "place": place,
                                             "model": "ST-GAT trained on Argoverse 2", "planner": planner},
                      lanes, crossings)
            f = out / f"cl_{stem}.npz"
            np.savez_compressed(f, **p)
            print(f"AV2 {stem}: {json.loads(p['metrics'])['caption']} -> {f.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
