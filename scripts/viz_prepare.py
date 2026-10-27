"""Build replay packs for the side-by-side CARLA visualization.

One pack per scene, for IDD-3D (chaotic Indian traffic, ours) and Argoverse 2
(structured US traffic, Western comparison). Each pack holds everything the
CARLA replay needs, so rendering never touches a model:

  - every agent's world-frame track (NaN where absent), class, heading;
  - the ego vehicle's track (IDD-3D: annotation-based ego estimate; AV2: 'AV');
  - Trajectron++ (benchmark-v2 seed-0 model, trained on IDD-3D only) run at
    every frame: deployable most-likely path + a few sampled futures;
  - per-class 90% conformal tube radii, calibrated on the IDD-3D val split
    (deployment-faithful: the same IDD-calibrated tubes are applied to AV2, so
    the Western side is a zero-shot test of both the forecaster and its
    uncertainty);
  - lane boundaries and crosswalks (AV2 only; IDD-3D roads are lane-free and
    have no map).

The model sees both sources through the same path: tracks -> Trajectron++
point-mass PEDESTRIAN nodes built exactly as scripts/6_export_trajectronpp_world.py
does (dt = 0.1 s, 20-frame history, 60-frame horizon).

Usage:
    python scripts/viz_prepare.py \
        --idd 20220118103808_seq_18 20220118112945_seq_22 \
        --av2 01340461-8ff4-4d3a-bf5f-bc5d1e4e5d4b 014586b7-...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
TRAJECTRON = REPO / "third_party" / "trajectron-plus-plus" / "trajectron"
TPP_DIR = REPO / "outputs" / "trajectronpp_world"
TPP_MODEL = TPP_DIR / "logs" / "models_16_Jul_2026_13_20_31_idd3d_world_v2_s0"
AV2_DIR = REPO / "data_external" / "av2" / "val"
OUT = REPO / "outputs" / "viz" / "packs"

OBS_LEN, PRED_LEN = 19, 60          # Trajectron++ history (past steps) / horizon
AV2_CLASS = {"vehicle": "Car", "bus": "Bus", "pedestrian": "Pedestrian",
             "cyclist": "BicycleRider", "motorcyclist": "MotorcyleRider"}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp6 = _load_module("exp6", SCRIPTS / "6_export_trajectronpp_world.py")


# --------------------------------------------------------------------------
# Scene loading -> common layout: pos (T,A,2) NaN-padded, ids, classes, ego
# --------------------------------------------------------------------------
def load_idd(scene_id: str) -> dict:
    d = pd.read_csv(REPO / "outputs" / "parsed_positions_world" / f"{scene_id}.csv",
                    dtype={"frame_id": str, "obj_id": str})
    frames = sorted(d.frame_id.unique())
    ids = sorted(d.obj_id.unique(), key=lambda s: (len(s), s))
    fi = {f: i for i, f in enumerate(frames)}
    ai = {a: i for i, a in enumerate(ids)}
    pos = np.full((len(frames), len(ids), 2), np.nan)
    pos[d.frame_id.map(fi).values, d.obj_id.map(ai).values] = d[["x", "y"]].values
    cls = d.drop_duplicates("obj_id").set_index("obj_id").obj_type.reindex(ids).values
    e = pd.read_csv(REPO / "outputs" / "ego_trajectories.csv", dtype={"frame_id": str})
    e = e[e.scene_id == scene_id].set_index("frame_id").reindex(frames)
    return {"name": scene_id, "source": "IDD-3D", "place": "India (IDD-3D)",
            "pos": pos, "ids": np.asarray(ids, dtype=object),
            "classes": np.asarray(cls, dtype=object),
            "ego": e[["ego_x", "ego_y"]].values.astype(float),
            "lanes": [], "crossings": [], "heading": None}


def load_av2(scenario_id: str) -> dict:
    sdir = AV2_DIR / scenario_id
    d = pd.read_parquet(sdir / f"scenario_{scenario_id}.parquet")
    city = str(d.city.iloc[0])
    T = int(d.num_timestamps.iloc[0])
    ego = d[d.track_id == "AV"].sort_values("timestep")
    ego_xy = np.full((T, 2), np.nan)
    ego_xy[ego.timestep.values] = ego[["position_x", "position_y"]].values
    d = d[d.object_type.isin(AV2_CLASS) & (d.track_id != "AV")]
    ids = sorted(d.track_id.unique())
    ai = {a: i for i, a in enumerate(ids)}
    pos = np.full((T, len(ids), 2), np.nan)
    hdg = np.full((T, len(ids)), np.nan)
    rows, cols = d.timestep.values, d.track_id.map(ai).values
    pos[rows, cols] = d[["position_x", "position_y"]].values
    hdg[rows, cols] = d.heading.values
    cls = d.drop_duplicates("track_id").set_index("track_id").object_type.reindex(ids).map(AV2_CLASS).values
    m = json.loads((sdir / f"log_map_archive_{scenario_id}.json").read_text(encoding="utf-8"))
    lanes = []
    for seg in m["lane_segments"].values():
        for side in ("left", "right"):
            mark = seg.get(f"{side}_lane_mark_type", "NONE")
            pts = np.array([[p["x"], p["y"]] for p in seg[f"{side}_lane_boundary"]])
            lanes.append((pts, str(mark)))
    crossings = [np.array([[p["x"], p["y"]] for p in c["edge1"]] +
                          [[p["x"], p["y"]] for p in c["edge2"]][::-1])
                 for c in m["pedestrian_crossings"].values()]
    return {"name": scenario_id, "source": "Argoverse 2", "place": f"{city.title()}, USA (Argoverse 2)",
            "pos": pos, "ids": np.asarray(ids, dtype=object), "classes": np.asarray(cls, dtype=object),
            "ego": ego_xy, "lanes": lanes, "crossings": crossings, "heading": hdg}


def headings(pos: np.ndarray, given: np.ndarray | None) -> np.ndarray:
    """Per-frame heading: dataset yaw when given (AV2), else smoothed motion
    direction held through slow periods (IDD-3D has no yaw)."""
    if given is not None:
        return given
    T, A, _ = pos.shape
    out = np.zeros((T, A))
    for a in range(A):
        p = pos[:, a]
        ok = np.isfinite(p).all(axis=1)
        if ok.sum() < 2:
            continue
        v = np.gradient(np.where(ok[:, None], p, np.nan), axis=0) / 0.1
        v = pd.DataFrame(v).rolling(5, center=True, min_periods=1).mean().values
        out[:, a] = exp6.motion_heading(np.nan_to_num(v[:, 0]), np.nan_to_num(v[:, 1]))
    return out


# --------------------------------------------------------------------------
# Trajectron++ (same node construction as script 6, homogeneous mode)
# --------------------------------------------------------------------------
def load_trajectron(device: str):
    sys.path.insert(0, str(TRAJECTRON))
    import torch
    from model.model_registrar import ModelRegistrar
    from model.trajectron import Trajectron
    exp7 = _load_module("exp7", SCRIPTS / "7_train_eval_trajectronpp.py")
    hyper = json.loads((TPP_MODEL / "config.json").read_text(encoding="utf-8"))
    reg = ModelRegistrar(str(TPP_MODEL), device)
    reg.load_models(exp7.latest_checkpoint_iter(TPP_MODEL))
    torch.manual_seed(0)
    return Trajectron(reg, hyper, None, device), hyper


def tpp_env(name: str, pos: np.ndarray, ids: np.ndarray, attention_radius: float):
    from environment import Environment, Scene, Node, derivative_of
    env = Environment(node_type_list=["PEDESTRIAN"], standardization=exp6.STANDARDIZATION_PED_ONLY)
    env.attention_radius = {(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): attention_radius}
    cols = pd.MultiIndex.from_product([["position", "velocity", "acceleration"], ["x", "y"]])
    T, A, _ = pos.shape
    scene = Scene(timesteps=T, dt=0.1, name=name)
    for a in range(A):
        s, e = exp6.longest_contiguous_run(np.isfinite(pos[:, a]).all(axis=1))
        if e - s < 2:
            continue
        df = pd.DataFrame(exp6.pedestrian_frame(pos[s:e, a], 0.1, derivative_of), columns=cols)
        scene.nodes.append(Node(node_type=env.NodeType.PEDESTRIAN, node_id=str(ids[a]),
                                data=df, first_timestep=s))
    env.scenes = [scene]
    return env, scene


def predict_all(stg, hyper, env, scene, n_samples: int):
    """Most-likely (T,A,60,2) and samples (T,A,S,60,2), NaN where no prediction."""
    import torch
    scene.calculate_scene_graph(env.attention_radius, hyper["edge_addition_filter"],
                                hyper["edge_removal_filter"])
    stg.set_environment(env)
    stg.set_annealing_params()
    ts = np.arange(scene.timesteps)
    kw = dict(min_history_timesteps=OBS_LEN, min_future_timesteps=0)
    with torch.no_grad():
        ml = stg.predict(scene, ts, PRED_LEN, num_samples=1, z_mode=True, gmm_mode=True,
                         full_dist=False, **kw)
        sp = stg.predict(scene, ts, PRED_LEN, num_samples=n_samples, z_mode=False,
                         gmm_mode=False, full_dist=False, **kw)
    return ml, sp


# --------------------------------------------------------------------------
# Conformal radii (IDD-3D val calibration, Trajectron++ dump)
# --------------------------------------------------------------------------
def conformal_radii(alpha: float) -> dict:
    cw = _load_module("cw", SCRIPTS / "conformal_wrapper.py")
    p = np.load(REPO / "outputs" / "benchmark_v2" / "preds_trajectronpp_canonical.npz", allow_pickle=True)
    v = p["split"].astype(object) == "val"
    s, valid = cw.scores_and_valid(p["pred"][v], p["gt"][v], p["fut_mask"][v])
    marg = cw.radii_per_timestep(s, valid, alpha)
    per_class = cw.radii_per_class(s, valid, p["agent_type"][v].astype(object), alpha, marg, 30)
    return {"__marginal__": marg, **per_class}


def build_pack(sc: dict, stg, hyper, radii: dict, n_samples: int, att_radius: float) -> dict:
    pos, ids = sc["pos"], sc["ids"]
    T, A, _ = pos.shape
    env, scene = tpp_env(sc["name"], pos, ids, att_radius)
    ml, sp = predict_all(stg, hyper, env, scene, n_samples)
    col = {str(a): i for i, a in enumerate(ids)}
    ML = np.full((T, A, PRED_LEN, 2), np.nan, dtype=np.float32)
    SP = np.full((T, A, n_samples, PRED_LEN, 2), np.nan, dtype=np.float32)
    for t, nodes in ml.items():
        for node, arr in nodes.items():
            ML[t, col[str(node.id)]] = np.asarray(arr).reshape(-1, PRED_LEN, 2)[0]
    for t, nodes in sp.items():
        for node, arr in nodes.items():
            SP[t, col[str(node.id)]] = np.asarray(arr).reshape(-1, PRED_LEN, 2)
    # Whole-scene caption metrics over every (agent, t) with a prediction and >= 1
    # valid future step (most-likely ADE + tube coverage). This is every replayed
    # frame, not the benchmark's canonical windows, so it is not a benchmark number.
    ade, cov_hit, cov_n = [], 0, 0
    for t in range(T):
        for a in range(A):
            if not np.isfinite(ML[t, a, 0, 0]):
                continue
            fut = pos[t + 1:t + 1 + PRED_LEN, a]
            fut = np.vstack([fut, np.full((PRED_LEN - len(fut), 2), np.nan)])
            ok = np.isfinite(fut).all(axis=1)
            if not ok.any():
                continue
            err = np.linalg.norm(ML[t, a] - fut, axis=1)[ok]
            ade.append(float(err.mean()))
            r = radii.get(str(sc["classes"][a]), radii["__marginal__"])[ok]
            cov_hit += int((err <= r).sum()); cov_n += int(ok.sum())
    center = np.nanmean(sc["ego"], axis=0) if np.isfinite(sc["ego"]).any() else np.nanmean(pos.reshape(-1, 2), axis=0)
    return {
        "name": sc["name"], "source": sc["source"], "place": sc["place"],
        "center": center, "pos": pos.astype(np.float32), "ids": ids, "classes": sc["classes"],
        "heading": headings(pos, sc["heading"]).astype(np.float32),
        "ego": sc["ego"].astype(np.float32), "ml": ML, "samples": SP,
        "radii_classes": np.asarray(list(radii), dtype=object),
        "radii": np.stack([radii[k] for k in radii]).astype(np.float32),
        "lanes": np.asarray([l for l, _ in sc["lanes"]] + [None], dtype=object)[:-1],
        "lane_marks": np.asarray([m for _, m in sc["lanes"]], dtype=object),
        "crossings": np.asarray(list(sc["crossings"]) + [None], dtype=object)[:-1],
        "metrics": {"tpp_most_likely_ade_m": float(np.mean(ade)) if ade else None,
                    "n_predictions": len(ade),
                    "tube_coverage": cov_hit / cov_n if cov_n else None,
                    "tube_target": 0.9},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idd", nargs="*", default=[])
    ap.add_argument("--av2", nargs="*", default=[])
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    meta = json.loads((TPP_DIR / "export_meta.json").read_text(encoding="utf-8"))
    att = float(meta["attention_radius_m"])
    stg, hyper = load_trajectron(args.device)
    radii = conformal_radii(args.alpha)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [("idd", s, load_idd) for s in args.idd] + [("av2", s, load_av2) for s in args.av2]
    for tag, sid, loader in jobs:
        pack = build_pack(loader(sid), stg, hyper, radii, args.samples, att)
        f = out / f"{tag}_{sid if tag == 'idd' else sid[:8]}.npz"
        metrics = pack.pop("metrics")
        np.savez_compressed(f, **pack, metrics=json.dumps(metrics))
        print(f"{pack['source']:12s} {sid}: {pack['pos'].shape[1]} agents, {pack['pos'].shape[0]} frames | "
              f"T++ most-likely ADE {metrics['tpp_most_likely_ade_m']:.2f} m over {metrics['n_predictions']} "
              f"predictions | 90% tube coverage {metrics['tube_coverage']:.1%} -> {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
