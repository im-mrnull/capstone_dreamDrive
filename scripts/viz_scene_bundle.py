"""Scene bundles for the trajectory-prediction scene viewer (development; local files only).

One JSON bundle per scene holds everything scripts/viz_render.py draws. Everything comes from data
already in the repository; no model is run and nothing is simulated:

  - the scene's recorded tracks at 10 Hz in its world frame (metres), classes, motion headings and
    the ego (scripts/viz_prepare.py loaders);
  - map geometry where it exists: Argoverse 2 lane boundaries and crosswalks. IDD-3D has no map
    (and no LiDAR, boxes or yaw in this repository), so none is drawn for it;
  - every canonical prediction window of the scene (2 s history, 6 s future, issued at frame
    t0 + 19) with each agent's history, ground-truth future and validity mask;
  - each model's saved predictions for those windows in the world frame: its most-likely path (as
    that model defines it) and, where saved, its unordered samples or its scored modes with
    probabilities. Rows are matched by (scene, t0, agent id) and checked against the canonical
    ground truth as they are read. Trajectron++ rows where it had no node (the evaluator's CV
    fallback) are flagged;
  - per model, 90% split-conformal bands (per step; per class where it has >= 30 calibration rows
    at that step, else the marginal radius; scripts/conformal_wrapper.py), calibrated on the other
    half of IDD val, never on the displayed scene;
  - the safety scorecard's quantities through its own functions (scripts/safety_scorecard.py):
    conflict = the true position within 5 m of the ego's actual position at a valid future step;
    danger score D = min over valid steps of (d_t - 5 m) / r_t, flagged at D <= 1;
  - kinematic indicators with stated definitions (constant-velocity time to conflict, minimum
    separations, close pairs), labelled as indicators, not safety-evaluation outputs.

Positions are stored as int16 centimetres relative to an anchor (the scene origin for tracks, the
agent's last observed point for predictions), so they are exact to 0.5 cm.

Usage:
    python scripts/viz_scene_bundle.py --dataset idd --scene 20220118105838_seq_8
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))
from dd_split import load_splits  # noqa: E402

OUT = REPO / "outputs" / "viz" / "bundles"
OBS, PRED = 20, 60
ALPHA, MIN_CAL, CONFLICT_M, K_SET, CLOSE_M, MOVING_MPS = 0.1, 30, 5.0, 10, 3.0, 1.0
IDD_CANON = REPO / "outputs" / "canonical_world_obs20_pred60.npz"
BENCH_SPLITS = REPO / "splits" / "scene_splits.json"
HALVES = REPO / "splits" / "chaos_gap" / "ft" / "idd_k72_r0_dev.json"     # IDD val: val = half A, test = half B
SEED0 = REPO / "outputs" / "benchmark_v2" / "seed_dumps" / "seed0"
CG = REPO / "outputs" / "chaos_gap"
CG_DEV = CG / "dev"
CG_DATA = CG / "av2_matched_dev.npz"
CG_SPLITS = REPO / "splits" / "chaos_gap" / "av2_matched_dev.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").lstrip("0123456789_"), SCRIPTS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vp, cw, sc, m9 = (_load(n) for n in ("viz_prepare.py", "conformal_wrapper.py", "safety_scorecard.py",
                                     "9_make_agent_windows_from_canonical.py"))

GROUP = {"Car": "car", "PoliceCar": "car", "TourCar": "car", "Van": "car", "Bus": "large", "Truck": "large",
         "MotorcyleRider": "moto", "ScooterRider": "moto", "Motorcycle": "moto_parked", "Scooter": "moto_parked",
         "BicycleRider": "bike", "BicycleGroup": "bike", "Bicycle": "bike_parked", "Pedestrian": "ped",
         "Animal": "animal"}

IDD_TRAINED = "IDD-3D (in-domain)"
AV2_TRAINED = "Argoverse 2, AV2-matched (zero-shot on IDD-3D)"
BENCH = "benchmark v2, seed 0 (selected on all IDD val)"
DEV = "chaos-gap development, seed 0 (selected on AV2 val)"
# (key, family, label, training domain, status, kind, source). kind: "point" (one path), "samples"
# (unordered draws) or "modes" (scored modes with probabilities). Sources are read only if present.
MODELS = [
    ("cv", "CV", "CV (vel_k = 5)", "none (no training)", "deterministic baseline", "point", {"compute": "cv"}),
    ("mlp_idd", "MLP", "MLP", IDD_TRAINED, BENCH, "point", {"compute": "mlp", "weights": "outputs/benchmark_v2/mlp_best_seed0.npz"}),
    ("stgat_idd", "ST-GAT", "ST-GAT (MHA)", IDD_TRAINED, BENCH, "point", {"dump": SEED0 / "preds_stgat.npz"}),
    ("transformer_idd", "Transformer", "Map-free transformer", IDD_TRAINED, BENCH, "point", {"dump": SEED0 / "preds_transformer.npz"}),
    ("mfp_idd", "MFP", "MFP (K-mode decoder, top-1)", IDD_TRAINED, BENCH, "modes", {"dump": SEED0 / "preds_mfp.npz"}),
    ("agentformer_idd", "AgentFormer", "AgentFormer", IDD_TRAINED, BENCH, "samples", {"dump": SEED0 / "preds_agentformer.npz"}),
    ("led_idd", "LED", "LED", IDD_TRAINED, BENCH, "samples", {"dump": SEED0 / "preds_led.npz"}),
    ("tpp_idd", "Trajectron++", "Trajectron++", IDD_TRAINED, BENCH, "samples", {"dump": SEED0 / "preds_trajectronpp_canonical.npz"}),
    ("autobot_idd", "AutoBot", "AutoBot-Joint", IDD_TRAINED,
     "chaos-gap development reference, seed 0 (selected on IDD val half A)", "modes",
     {"dump": CG_DEV / "autobot_idd_ref_preds.npz", "data": IDD_CANON, "splits": HALVES}),
    ("mlp_av2", "MLP", "MLP", AV2_TRAINED, DEV, "point", {"dump": CG_DEV / "mlp_matched_eval_preds.npz", "windows": True}),
    ("stgat_av2", "ST-GAT", "ST-GAT (MHA)", AV2_TRAINED, DEV, "point", {"dump": CG_DEV / "stgat_matched_eval_preds.npz"}),
    ("transformer_av2", "Transformer", "Map-free transformer", AV2_TRAINED, DEV, "point", {"dump": CG_DEV / "transformer_matched_eval_preds.npz"}),
    ("mfp_av2", "MFP", "MFP (K-mode decoder, top-1)", AV2_TRAINED, DEV, "modes", {"dump": CG_DEV / "mfp_matched_preds.npz"}),
    ("agentformer_av2", "AgentFormer", "AgentFormer", AV2_TRAINED, DEV, "samples",
     {"dump": CG_DEV / "agentformer_matched_eval_preds.npz", "fallback": CG_DEV / "agentformer_matched_preds.npz"}),
    ("led_av2", "LED", "LED", AV2_TRAINED, DEV, "samples", {"dump": CG_DEV / "led_matched_eval_preds.npz"}),
    ("autobot_av2", "AutoBot", "AutoBot-Joint", AV2_TRAINED, DEV, "modes", {"dump": CG_DEV / "autobot_matched_ent8_x4_eval_preds.npz"}),
    ("tpp_av2", "Trajectron++", "Trajectron++", AV2_TRAINED, DEV, "samples",
     {"excluded": "excluded: failed the pre-specified motion check on AV2 val (docs/wiki/chaos-gap.md)"}),
]
CAVEATS = {"autobot_idd": "fails on scene 20220118103308_seq_27 (median ADE 33.5 m); not a valid in-domain "
                          "reference until investigated (outputs/chaos_gap/dev/halves_report.json)"}
ML_DEFINITION = {
    "CV": "constant velocity: mean of the last 5 observed steps, extrapolated",
    "MLP": "single regressed path", "ST-GAT": "single regressed path (free-running decoder)",
    "Transformer": "single regressed path",
    "MFP": "the mode ranked first by the learned, context-only mode-scoring head (non-oracle)",
    "AgentFormer": "the z = 0 rollout of the CVAE prior (2 Hz keyframes, linearly upsampled to 10 Hz)",
    "LED": "the initializer's mean (non-oracle)",
    "Trajectron++": "the GMM mode of the most probable latent mode (z_mode, gmm_mode)",
    "AutoBot": "the scene's most probable joint mode",
}
SET_DEFINITION = {
    "samples": "unordered draws from the model's distribution (no probabilities; equally weighted)",
    "modes": "scored modes with the model's probabilities",
}


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------
def pack(a, anchor=None) -> dict:
    """int16 centimetres relative to anchor (exact to 0.5 cm), float32 if out of range. NaN -> -32768."""
    a = np.asarray(a, np.float64)
    rel = a - (0.0 if anchor is None else anchor)
    cm = np.round(rel * 100.0)
    if np.nanmax(np.abs(cm), initial=0) < 32767:
        q = np.where(np.isfinite(cm), cm, -32768).astype("<i2")
        return {"dt": "i2cm", "shape": list(a.shape), "b64": base64.b64encode(q.tobytes()).decode()}
    return {"dt": "f4", "shape": list(a.shape), "b64": base64.b64encode(np.asarray(rel, "<f4").tobytes()).decode()}


def unpack(p: dict, anchor=None) -> np.ndarray:
    raw = base64.b64decode(p["b64"])
    if p["dt"] == "i2cm":
        q = np.frombuffer(raw, "<i2").astype(np.float64)
        a = np.where(q == -32768, np.nan, q / 100.0)
    else:
        a = np.frombuffer(raw, "<f4").astype(np.float64)
    return a.reshape(p["shape"]) + (0.0 if anchor is None else anchor)


def pack_f(a) -> dict:
    return {"dt": "f4", "shape": list(np.shape(a)), "b64": base64.b64encode(np.asarray(a, "<f4").tobytes()).decode()}


def bits(a) -> dict:
    a = np.asarray(a, bool)
    return {"dt": "bits", "shape": list(a.shape), "b64": base64.b64encode(np.packbits(a.ravel()).tobytes()).decode()}


def unbits(p: dict) -> np.ndarray:
    n = int(np.prod(p["shape"]))
    return np.unpackbits(np.frombuffer(base64.b64decode(p["b64"]), np.uint8))[:n].astype(bool).reshape(p["shape"])


def _rel(p) -> str:
    return os.path.relpath(p, REPO).replace("\\", "/")


# --------------------------------------------------------------------------
# Rows: canonical keys, dumps in the world frame
# --------------------------------------------------------------------------
def canonical_order(data: Path, splits: Path) -> tuple[dict, list]:
    """A canonical file and its val-then-test rows in dump order: (scene, t0, agent_id, row)."""
    c = dict(np.load(data, allow_pickle=True))
    m = load_splits(splits)
    order = []
    for want in ("val", "test"):
        for q, (a, b) in enumerate(c["seq_start_end"]):
            sid = str(c["scene_id"][q])
            if m.get(sid) == want:
                order += [(sid, int(c["t0"][q]), int(c["agent_id"][i]), i) for i in range(a, b)]
    return c, order


def window_rotations(obs: np.ndarray) -> np.ndarray:
    """Script 9's per-agent rotations (recent heading -> +x) from world-frame observations."""
    o = obs - obs[:, -1:, :]
    k = m9.HEADING_K
    return m9.rotation_matrices((o[:, -k:, :] - o[:, -k - 1:-1, :]).sum(axis=1)).astype(np.float64)


def read_dump(spec: dict, kind: str, family: str, keys: set):
    """({key: {"ml", "set", "probs", "fallback"}} in the world frame for the wanted keys, file read)."""
    path = Path(spec["dump"])
    if not path.exists() and spec.get("fallback") and Path(spec["fallback"]).exists():
        path, kind = Path(spec["fallback"]), "point"
    if not path.exists():
        return None, _rel(path)
    data = Path(spec.get("data", CG_DATA if CG_DEV in path.parents else IDD_CANON))
    splits = Path(spec.get("splits", CG_SPLITS if CG_DEV in path.parents else BENCH_SPLITS))
    c, order = canonical_order(data, splits)
    p = np.load(path, allow_pickle=True)
    if len(p["pred"]) != len(order):
        raise SystemExit(f"{path}: {len(p['pred'])} rows, but {data.name} under {splits.name} has {len(order)}")
    want = [j for j, k in enumerate(order) if k[:3] in keys]           # never reads any other row (frozen test)
    rows = np.array([order[j][3] for j in want], int)
    assert list(map(str, p["scene_id"][want])) == [order[j][0] for j in want], f"{path}: scene order"
    gt_c = c["pred_traj"][rows].transpose(0, 2, 1).astype(np.float64)
    obs = c["obs_traj"][rows].transpose(0, 2, 1).astype(np.float64)
    fm = c["fut_mask"][rows].astype(bool)
    assert np.array_equal(np.asarray(p["fut_mask"][want], bool), fm), f"{path}: fut_mask"
    pred, gt = p["pred"][want].astype(np.float64), p["gt"][want].astype(np.float64)
    if spec.get("windows"):                                             # script 9 frame -> world
        R, last = window_rotations(obs), obs[:, -1:, :]
        pred = np.einsum("nji,ntj->nti", R, pred) + last
        gt = np.einsum("nji,ntj->nti", R, gt) + last
    err = np.abs(np.where(fm[..., None], gt - gt_c, 0)).max() if len(rows) else 0.0
    assert err < 1e-3, f"{path}: ground truth differs from the canonical rows by {err:.2e} m"
    sets = probs = None
    if kind in ("samples", "modes"):
        key = "samples" if "samples" in p.files else "modes" if "modes" in p.files else None
        sets = None if key is None else p[key][want].astype(np.float64)
        if "probs" in p.files:
            probs = p["probs"][want].astype(np.float64)
        elif "mode_logits" in p.files:
            z = p["mode_logits"][want].astype(np.float64)
            probs = np.exp(z - z.max(1, keepdims=True))
            probs /= probs.sum(1, keepdims=True)
    fallback = np.zeros(len(want), bool)
    if family == "Trajectron++":                                        # the evaluator's CV stand-in rows
        cvp = obs[:, -1:, :] + cw.cv_predict(obs, 5)
        same = np.abs(pred - cvp).reshape(len(want), -1).max(1) < 1e-3
        if sets is not None:
            same &= np.abs(sets - pred[:, None]).reshape(len(want), -1).max(1) < 1e-3
        fallback = same
    out = {order[j][:3]: {"ml": pred[n], "set": None if sets is None else sets[n],
                          "probs": None if probs is None else probs[n], "fallback": bool(fallback[n])}
           for n, j in enumerate(want)}
    return out, _rel(path)


def compute_model(spec: dict, rows: dict) -> dict:
    """CV and the IDD-trained MLP, reproduced in numpy as the safety scorecard does."""
    keys = list(rows)
    obs = np.stack([rows[k]["obs"] for k in keys])
    if spec["compute"] == "cv":
        pred = obs[:, -1:, :] + cw.cv_predict(obs, 5)
    else:
        w = np.load(REPO / spec["weights"])
        pred = sc.mlp_world(obs, {k: w[k] for k in w.files})
    return {k: {"ml": pred[n], "set": None, "probs": None, "fallback": False} for n, k in enumerate(keys)}


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------
def idd_rows(scenes: set, data: Path = IDD_CANON) -> dict:
    """(scene, t0, agent_id) -> history, future, mask, class for the canonical rows of scenes."""
    c = np.load(data, allow_pickle=True)
    out = {}
    for q, (a, b) in enumerate(c["seq_start_end"]):
        sid = str(c["scene_id"][q])
        if sid in scenes:
            for i in range(a, b):
                out[(sid, int(c["t0"][q]), int(c["agent_id"][i]))] = {
                    "obs": c["obs_traj"][i].T.astype(np.float64), "gt": c["pred_traj"][i].T.astype(np.float64),
                    "fm": c["fut_mask"][i].astype(bool), "cls": str(c["agent_type"][i])}
    return out


def ttc_cv(dp: np.ndarray, dv: np.ndarray, radius: float, horizon: float) -> float:
    """First t in [0, horizon] with |dp + dv t| <= radius (constant velocity), else inf."""
    if np.linalg.norm(dp) <= radius:
        return 0.0
    a, b, c = dv @ dv, 2 * dp @ dv, dp @ dp - radius ** 2
    if a < 1e-12:
        return np.inf
    disc = b * b - 4 * a * c
    if disc < 0:
        return np.inf
    t = (-b - np.sqrt(disc)) / (2 * a)
    return float(t) if 0 <= t <= horizon else np.inf


def close_pairs(paths: np.ndarray, valid: np.ndarray, limit: float) -> list:
    """[i, j, min distance, step] for agent pairs whose positions at the same valid step come within limit."""
    out = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            v = valid[i] & valid[j]
            if not v.any():
                continue
            d = np.where(v, np.linalg.norm(paths[i] - paths[j], axis=1), np.inf)
            s = int(np.argmin(d))
            if d[s] <= limit:
                out.append([i, j, round(float(d[s]), 3), s])
    return out


def moving_flags(pos: np.ndarray) -> np.ndarray:
    """Agents whose displacement over some 1 s reaches MOVING_MPS (robust to IDD's per-frame jitter)."""
    d = np.linalg.norm(pos[10:] - pos[:-10], axis=2) / 1.0
    d = np.where(np.isfinite(d), d, -np.inf).max(axis=0)               # agents never seen 1 s apart: -inf
    return d >= MOVING_MPS


def ego_heading(xy: np.ndarray, yaw: np.ndarray) -> tuple[np.ndarray, float]:
    """The ego's forward direction. ego_yaw is the sensor frame's rotation, whose forward axis is not +x
    (the ego drives along its sensor's -y): heading = yaw + the scene's median offset between the motion
    direction (moving >= 2 m/s over 1 s) and yaw; -90 degrees if the ego never moves that fast."""
    v = xy[10:] - xy[:-10]
    moving = np.linalg.norm(v, axis=1) > 2.0
    off = np.angle(np.exp(1j * (np.arctan2(v[:, 1], v[:, 0]) - yaw[5:-5])))[moving]
    offset = float(np.angle(np.mean(np.exp(1j * off)))) if moving.any() else -np.pi / 2
    return yaw + offset, offset


def build_idd(scene: str) -> dict:
    halves = load_splits(HALVES)
    if halves.get(scene) not in ("val", "test"):
        raise SystemExit(f"{scene} is not an IDD val scene; only val scenes are visualised (the test split is frozen)")
    cal_split = "test" if halves[scene] == "val" else "val"            # always the other half
    cal_scenes = sorted(s for s, k in halves.items() if k == cal_split)
    raw = vp.load_idd(scene)
    T = raw["pos"].shape[0]
    frames = sorted(pd.read_csv(REPO / "outputs" / "parsed_positions_world" / f"{scene}.csv",
                                dtype={"frame_id": str}).frame_id.unique())
    ego_csv = pd.read_csv(REPO / "outputs" / "ego_trajectories.csv", dtype={"frame_id": str})
    yaw = ego_csv[ego_csv.scene_id == scene].set_index("frame_id").reindex(frames).ego_yaw.values.astype(float)
    ids = [int(i) for i in raw["ids"]]
    rows = idd_rows({scene, *cal_scenes})
    disp = sorted((k for k in rows if k[0] == scene), key=lambda k: (k[1], ids.index(k[2])))
    eh, off = ego_heading(raw["ego"].astype(np.float64), yaw)
    return {"dataset": "IDD-3D", "scene": scene, "raw": raw, "T": T, "ids": ids,
            "heading": vp.headings(raw["pos"], None), "has_heading": moving_flags(raw["pos"]),
            "ego_heading": eh, "ego_heading_note": f"ego_yaw + the scene's median offset to the motion direction "
                                                  f"({np.degrees(off):.1f} deg; ego_yaw is the sensor frame's rotation)",
            "rows": rows, "disp": disp, "cal": [k for k in rows if k[0] in cal_scenes],
            "cal_scenes": cal_scenes, "cal_label": f"IDD val half {'A' if cal_split == 'val' else 'B'}",
            "split_label": f"IDD val (dev half {'B' if halves[scene] == 'test' else 'A'})", "origin": np.zeros(2),
            "models": [m[0] for m in MODELS], "domain_override": {},
            "frame": ("IDD-3D world frame: origin and axes = the first ego pose; ego motion estimated from the "
                      "annotations (scripts/0_compensate_ego_motion.py, an approximation)"),
            "ego_source": "annotation-based ego-motion estimate (outputs/ego_trajectories.csv)",
            "notes": ["No map: IDD-3D roads are unmapped, and its LiDAR, boxes and yaw are not in this repository.",
                      "Agent markers use nominal class sizes; IDD-3D box sizes are not in this repository.",
                      "Headings are motion directions (smoothed); agents that never move 1 m in a second have none.",
                      "The ego pose is the annotation-based estimate, not an official pose.",
                      "Predictions exist only at the canonical windows: issued every 0.3 s from 1.9 s to 3.7 s.",
                      "Development figures: not for publication before the IP review."]}


AV2_IN_DOMAIN = "Argoverse 2, AV2-matched (in-domain on AV2 val)"


def build_av2(uuid: str) -> dict:
    """An Argoverse 2 val scenario of the chaos-gap AV2-matched file: AV2-trained models in-domain, real
    lane boundaries and crosswalks, dataset headings, the AV as the ego. Bands calibrated on the other
    AV2 val scenarios."""
    scene = "av2_" + uuid
    m = load_splits(CG_SPLITS)
    if m.get(scene) != "val":
        raise SystemExit(f"{scene} is not an AV2 val scenario of {CG_SPLITS.name}")
    cal_scenes = sorted(s for s, k in m.items() if k == "val" and s != scene)
    raw = vp.load_av2(uuid)
    T = raw["pos"].shape[0]
    d = pd.read_parquet(vp.AV2_DIR / uuid / f"scenario_{uuid}.parquet")
    av = d[d.track_id == "AV"].sort_values("timestep")
    ego_h = np.full(T, np.nan)
    ego_h[av.timestep.values] = av.heading.values
    ids = [int(i) for i in raw["ids"]]
    rows = idd_rows({scene, *cal_scenes}, CG_DATA)
    disp = sorted((k for k in rows if k[0] == scene), key=lambda k: (k[1], ids.index(k[2])))
    origin = np.round(np.nanmean(raw["ego"].astype(np.float64), axis=0))
    keep = ["cv"] + [k for k, *_ in MODELS if k.endswith("_av2")]
    return {"dataset": "Argoverse 2", "scene": scene, "raw": raw, "T": T, "ids": ids,
            "heading": vp.headings(raw["pos"], raw["heading"]), "has_heading": np.ones(len(ids), bool),
            "ego_heading": ego_h, "ego_heading_note": "the AV's recorded heading",
            "rows": rows, "disp": disp, "cal": [k for k in rows if k[0] in cal_scenes], "cal_scenes": cal_scenes,
            "cal_label": f"AV2 val (the other {len(cal_scenes)} scenarios)", "split_label": "AV2 val",
            "origin": origin, "models": keep, "domain_override": {k: AV2_IN_DOMAIN for k in keep if k != "cv"},
            "frame": f"Argoverse 2 city frame ({raw['place']}), shifted by {origin.tolist()} m",
            "ego_source": "the Argoverse 2 AV track",
            "notes": ["Map: Argoverse 2 lane boundaries (with their marking types) and crosswalks for this scenario.",
                      "Agent markers use nominal class sizes; Argoverse 2 motion-forecasting data has no box sizes.",
                      "Headings are the dataset's.",
                      "One canonical window per scenario: the prediction is issued at 4.9 s (frame 49).",
                      "Development figures: not for publication before the IP review."]}


def model_predictions(b: dict) -> tuple[list, dict]:
    keys = set(b["disp"]) | set(b["cal"])
    meta, preds = [], {}
    for key, family, label, domain, status, kind, spec in MODELS:
        if key not in b["models"]:
            continue
        domain = b["domain_override"].get(key, domain)
        entry = {"key": key, "family": family, "label": label, "domain": domain, "status": status,
                 "ml_definition": ML_DEFINITION[family], "caveat": CAVEATS.get(key)}
        if "excluded" in spec:
            meta.append({**entry, "kind": kind, "available": False, "source": None, "reason": spec["excluded"]})
            continue
        if "compute" in spec:
            got = compute_model(spec, {k: b["rows"][k] for k in keys})
            src = ("scripts/conformal_wrapper.py cv_predict" if spec["compute"] == "cv"
                   else f"scripts/safety_scorecard.py mlp_world with {spec['weights']}")
        else:
            got, src = read_dump(spec, kind, family, keys)
        if got is None:
            meta.append({**entry, "kind": kind, "available": False, "source": src,
                         "reason": "prediction file not written yet"})
            continue
        missing = [k for k in b["disp"] if k not in got]
        if missing:
            raise SystemExit(f"{key}: {len(missing)} display rows missing, e.g. {missing[0]}")
        first = got[b["disp"][0]]
        has_set = first["set"] is not None
        meta.append({**entry, "available": True, "source": src, "kind": kind if has_set else "point",
                     "K": int(first["set"].shape[0]) if has_set else 1,
                     "set_definition": SET_DEFINITION[kind] if has_set else None,
                     "has_probs": first["probs"] is not None})
        preds[key] = got
    return meta, preds


def conformal(b: dict, got: dict, use_set: bool) -> tuple[np.ndarray, dict]:
    """(n_disp, 60) radii for the displayed rows, calibrated on b['cal'] only."""
    cal, disp = b["cal"], b["disp"]
    gt = np.stack([b["rows"][k]["gt"] for k in cal])
    fm = np.stack([b["rows"][k]["fm"] for k in cal])
    cls = np.array([b["rows"][k]["cls"] for k in cal])
    if use_set:
        k_used = min(K_SET, got[cal[0]]["set"].shape[0])
        s = sc.min_sample_dist(np.stack([got[k]["set"][:k_used] for k in cal]), gt)
    else:
        s = np.linalg.norm(np.stack([got[k]["ml"] for k in cal]) - gt, axis=-1)
    marginal = cw.radii_per_timestep(s, fm, ALPHA)
    per_class = cw.radii_per_class(s, fm, cls, ALPHA, marginal, MIN_CAL)
    r = cw.radii_matrix(np.array([b["rows"][k]["cls"] for k in disp]), marginal, per_class)
    info = {"alpha": ALPHA, "min_cal_per_class_step": MIN_CAL, "cal_rows": len(cal),
            "score": (f"min over the first {k_used} set members of the error" if use_set
                      else "error of the most-likely path"),
            "per_class_profiles": sorted(per_class)}
    return np.asarray(r, np.float64), info


def _round_list(a, nd=3):
    return [None if (v is None or not np.isfinite(v)) else round(float(v), nd) for v in a]


def build_bundle(b: dict) -> dict:
    raw, T, origin = b["raw"], b["T"], b["origin"]
    meta, preds = model_predictions(b)
    bands = {}
    for key, got in preds.items():
        m = next(x for x in meta if x["key"] == key)
        bands[key] = {"ml": conformal(b, got, use_set=False)}
        m["conformal"] = bands[key]["ml"][1]
        if m["kind"] != "point":
            bands[key]["set"] = conformal(b, got, use_set=True)
            m["conformal_set"] = bands[key]["set"][1]
    ego = raw["ego"].astype(np.float64)
    windows = []
    for t0 in sorted({k[1] for k in b["disp"]}):
        keys = [k for k in b["disp"] if k[1] == t0]
        idx = [b["disp"].index(k) for k in keys]
        n, issue = len(keys), t0 + OBS - 1
        hist = np.stack([b["rows"][k]["obs"] for k in keys])
        gt = np.stack([b["rows"][k]["gt"] for k in keys])
        fm = np.stack([b["rows"][k]["fm"] for k in keys])
        last = hist[:, -1, :]
        ego_fut = ego[issue + 1:issue + 1 + PRED]
        assert ego_fut.shape == (PRED, 2) and np.isfinite(ego_fut).all(), (b["scene"], t0)
        ego_futs = np.broadcast_to(ego_fut, (n, PRED, 2))
        ego_now, ego_v = ego[issue], (ego[issue] - ego[issue - 10]) / 1.0
        agent_v = (hist[:, -1] - hist[:, -11]) / 1.0
        d_gt = np.linalg.norm(gt - ego_futs, axis=-1)
        any_fm = fm.any(1)
        w = {"t0": t0, "issue_frame": issue, "agents": [b["ids"].index(k[2]) for k in keys],
             "cls": [b["rows"][k]["cls"] for k in keys],
             "hist": pack(hist, origin), "gt": pack(np.where(fm[..., None], gt, np.nan), last[:, None, :]),
             "fm": bits(fm), "ego_fut": pack(ego_fut, origin),
             "conflict": sc.conflict_labels(gt, fm, ego_futs).tolist(),
             "kin": {"ttc_cv_s": _round_list([ttc_cv(last[i] - ego_now, agent_v[i] - ego_v, CONFLICT_M, PRED / 10)
                                              for i in range(n)], 2),
                     "dist_now_m": _round_list(np.linalg.norm(last - ego_now, axis=1), 2),
                     "speed_mps": _round_list(np.linalg.norm(agent_v, axis=1), 2),
                     "min_sep_gt_m": _round_list([np.min(d_gt[i][fm[i]]) if any_fm[i] else np.nan for i in range(n)], 2),
                     "min_sep_gt_step": [int(np.flatnonzero(fm[i])[np.argmin(d_gt[i][fm[i]])]) if any_fm[i] else None
                                         for i in range(n)]},
             "pairs_gt": close_pairs(gt, fm, CLOSE_M), "preds": {}}
        last_valid = np.array([np.flatnonzero(f)[-1] if f.any() else 0 for f in fm])
        for key, got in preds.items():
            ml = np.stack([got[k]["ml"] for k in keys])
            e = np.linalg.norm(ml - gt, axis=-1)
            r_ml = bands[key]["ml"][0][idx]
            cover = (e <= r_ml) | ~fm
            entry = {"ml": pack(ml, last[:, None, :]), "radii_ml": pack_f(r_ml),
                     "fallback": [bool(got[k]["fallback"]) for k in keys],
                     "ade": _round_list([e[i][fm[i]].mean() if any_fm[i] else np.nan for i in range(n)]),
                     "fde": _round_list([e[i][last_valid[i]] if any_fm[i] else np.nan for i in range(n)]),
                     "covered_steps": _round_list([cover[i][fm[i]].mean() if any_fm[i] else np.nan for i in range(n)]),
                     "covered_all": [bool(cover[i].all()) if any_fm[i] else None for i in range(n)],
                     "danger_ml": _round_list(sc.danger_scores(ml[:, None], r_ml, fm, ego_futs)),
                     "pairs_pred": close_pairs(ml, np.ones_like(fm), CLOSE_M)}
            if "set" in bands[key]:
                st = np.stack([got[k]["set"] for k in keys])
                r_set = bands[key]["set"][0][idx]
                es = np.linalg.norm(st - gt[:, None], axis=-1)
                ade_k = np.nanmean(np.where(fm[:, None], es, np.nan), axis=2)
                entry.update(set=pack(st, last[:, None, None, :]), radii_set=pack_f(r_set),
                             min_ade=_round_list([np.nanmin(ade_k[i]) if any_fm[i] else np.nan for i in range(n)]),
                             min_fde=_round_list([es[i, :, last_valid[i]].min() if any_fm[i] else np.nan for i in range(n)]),
                             oracle_index=[int(np.nanargmin(ade_k[i])) if any_fm[i] else None for i in range(n)],
                             danger_set=_round_list(sc.danger_scores(st[:, :K_SET], r_set, fm, ego_futs)))
                if got[keys[0]]["probs"] is not None:
                    entry["probs"] = pack_f(np.stack([got[k]["probs"] for k in keys]))
            w["preds"][key] = entry
        windows.append(w)
    return {
        "schema": 1,
        "scene": {"dataset": b["dataset"], "id": b["scene"], "place": raw["place"], "split": b["split_label"],
                  "fps": 10, "n_frames": T, "origin": origin.tolist(), "units": "metres, seconds",
                  "frame": b["frame"], "notes": b["notes"], "built": datetime.now().isoformat(timespec="seconds")},
        "classes": {"group": GROUP, "default_group": "other"},
        "agents": {"ids": b["ids"], "cls": [str(c) for c in raw["classes"]],
                   "group": [GROUP.get(str(c), "other") for c in raw["classes"]],
                   "has_heading": b["has_heading"].tolist(), "track": pack(raw["pos"].astype(np.float64), origin),
                   "heading": pack_f(b["heading"])},
        "ego": {"track": pack(ego, origin), "heading": pack_f(b["ego_heading"]),
                "source": f"{b['ego_source']}; heading = {b['ego_heading_note']}"},
        "map": None if not raw["lanes"] else {
            "lanes": [{"pts": pack(l, origin), "mark": mk} for l, mk in raw["lanes"]],
            "crossings": [pack(cz, origin) for cz in raw["crossings"]]},
        "models": meta,
        "calibration": {"scenes": b["cal_scenes"], "label": b["cal_label"], "rows": len(b["cal"])},
        "definitions": {
            "conflict": f"the agent's true position within {CONFLICT_M:g} m of the ego's actual position at the same "
                        "valid future step (safety scorecard; docs/wiki/safety-scorecard.md)",
            "danger": f"D = min over valid steps of (d_t - {CONFLICT_M:g} m) / r_t, with d_t the distance from the ego's "
                      "actual position to the nearest predicted position and r_t the 90% conformal radius; flagged "
                      "at D <= 1 (safety scorecard)",
            "conformal": "90% split-conformal radius per future step, per class where a class has >= 30 "
                         "calibration rows at that step, else the marginal radius (scripts/conformal_wrapper.py)",
            "ttc_cv": f"kinematic indicator, not a safety-evaluation output: time until the agent and the ego, both "
                      f"at constant velocity (their last observed second), come within {CONFLICT_M:g} m",
            "min_sep": "minimum centre distance over the valid future steps (no box extents)",
            "close_pair": f"two agents whose centres come within {CLOSE_M:g} m at the same step (recorded futures, "
                          "or one model's most-likely paths)",
            "oracle": "best-of-K = the set member closest to the ground truth: uses the future, never deployable",
            "fallback": "Trajectron++ had no node for this window; its evaluator substitutes CV for every sample",
        },
        "windows": windows,
    }


PUBLIC_MODELS = ("cv", "mlp_idd", "stgat_idd", "transformer_idd", "mfp_idd", "agentformer_idd", "led_idd", "tpp_idd")
WINDOW_PRIVATE = ("conflict", "kin", "pairs_gt", "ego_fut")
PRED_PRIVATE = ("radii_ml", "radii_set", "covered_steps", "covered_all", "danger_ml", "danger_set", "pairs_pred")


def publicize(bundle: dict) -> dict:
    """Only what benchmark v2 already publishes: CV and the IDD-trained benchmark models (seed 0) on an IDD
    val scene. Removed, not hidden: AV2-trained models, the AutoBot reference, conformal bands, safety
    quantities, kinematic indicators and calibration details."""
    out = json.loads(json.dumps(bundle))
    out["public"] = True
    out["models"] = [m for m in out["models"] if m["key"] in PUBLIC_MODELS]
    for m in out["models"]:
        for k in ("conformal", "conformal_set", "caveat", "source"):
            m.pop(k, None)
    for w in out["windows"]:
        for k in WINDOW_PRIVATE:
            w.pop(k, None)
        w["preds"] = {k: {f: v for f, v in e.items() if f not in PRED_PRIVATE} for k, e in w["preds"].items() if k in PUBLIC_MODELS}
    out["calibration"] = None
    out["scene"]["split"] = "val"                                       # the halves are a private protocol detail
    out["definitions"] = {k: v for k, v in out["definitions"].items() if k in ("oracle", "fallback")}
    out["scene"]["notes"] = [n for n in out["scene"]["notes"] if not n.startswith("Development")] + [
        "Benchmark v2 IDD-trained models (seed 0) and CV on an IDD-3D val scene: results the benchmark already reports."]
    text = json.dumps(out)
    for leak in ("_av2", "autobot", "radii", "danger", "conformal", "chaos"):
        assert leak not in text.lower(), f"public bundle still mentions {leak}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a scene bundle for the trajectory scene viewer")
    ap.add_argument("--dataset", choices=["idd", "av2"], default="idd")
    ap.add_argument("--scene", default=None, help="IDD scene id, or the AV2 scenario uuid")
    ap.add_argument("--public", action="store_true", help="only already-public benchmark content (see publicize)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.dataset == "av2" and args.public:
        raise SystemExit("--public covers the IDD-3D benchmark only: every AV2 prediction is a chaos-gap result")
    scene = args.scene or ("20220118105838_seq_8" if args.dataset == "idd" else "5a0868d6-958c-42b4-aa77-a238010a5da7")
    bundle = build_bundle(build_idd(scene) if args.dataset == "idd" else build_av2(scene))
    if args.public:
        bundle = publicize(bundle)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{'public_' if args.public else ''}{args.dataset}_{scene[:8] if args.dataset == 'av2' else scene}.json"
    f.write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")
    avail = [m["key"] for m in bundle["models"] if m["available"]]
    missing = [m["key"] for m in bundle["models"] if not m["available"]]
    n = sum(len(w["agents"]) for w in bundle["windows"])
    print(f"{_rel(f)} ({f.stat().st_size / 1e6:.1f} MB): {len(bundle['windows'])} windows, {n} agent-windows, "
          f"{len(bundle['agents']['ids'])} agents | {len(avail)} models: {', '.join(avail)}")
    if missing:
        print(f"not available yet: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
