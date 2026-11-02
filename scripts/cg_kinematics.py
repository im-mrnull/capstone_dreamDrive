"""Chaos-gap kinematics (development): per-class motion statistics of the Argoverse 2 and
IDD-3D agent-windows the models train and are scored on, to describe the shift before any
model is compared.

Per agent-window (the evaluation unit; IDD's overlapping windows count once each), at 10 Hz
over the 20 observed and the valid future frames, using only differences between valid frames.
Motion is measured over a lag of LAG frames (1 s): IDD-3D's per-frame position noise (about
8 cm per axis) makes one-frame velocities noisy by about 1.2 m/s, against about 0.12 m/s at 1 s.

    speed      mean speed over 1 s displacements (m/s)
    accel_p90  90th percentile of |change of the 1 s velocity over 1 s| (m/s^2)
    yaw_rate   mean |change of the 1 s heading over 1 s| where both speeds > 1 m/s (rad/s;
               NaN if never moving)
    jitter     RMS one-frame second difference |p[t+1] - 2 p[t] + p[t-1]| (m). For smooth
               motion it is about a dt^2 (0.01 m at 1 m/s^2); white position noise of sd s per
               axis adds sqrt(12) s, so it is a proxy for track noise (AV2: onboard tracker;
               IDD: annotated boxes and our ego compensation)
    stationary max distance from the first observed position < 1 m (reported as a fraction)
    fut_valid  fraction of the 60 future frames present
Each (source, class) gets the mean, median and 90th percentile of every statistic, and every IDD
class with an AV2 counterpart gets the Wasserstein-1 distance to it (AV2 train). IDD classes
absent from AV2 are listed as unseen. The frozen IDD-3D test split is never read.

Usage:
    python scripts/cg_kinematics.py --name av2_large_dev
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from dd_split import load_splits

REPO = Path(__file__).resolve().parent.parent
CG = REPO / "outputs" / "chaos_gap"
DT = 0.1
LAG = 10                                                                  # frames (1 s)
FEATURES = ("speed", "accel_p90", "yaw_rate", "jitter", "stationary", "fut_valid")
AV2_COUNTERPART = {"Car": "Car", "Bus": "Bus", "Pedestrian": "Pedestrian",
                   "MotorcyleRider": "MotorcyleRider", "BicycleRider": "BicycleRider"}


def lagged(x: np.ndarray, m: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """(x[t+lag] - x[t]) / (lag dt) along axis 1, valid where both ends are."""
    return (x[:, lag:] - x[:, :-lag]) / (lag * DT), m[:, lag:] & m[:, :-lag]


def agent_features(obs: np.ndarray, pred: np.ndarray, fut_mask: np.ndarray, lag: int = LAG) -> dict[str, np.ndarray]:
    """obs (N, 2, To), pred (N, 2, Tf), fut_mask (N, Tf) -> one value per agent per feature."""
    p = np.concatenate([obs, pred], axis=2).transpose(0, 2, 1)             # (N, T, 2)
    v = np.concatenate([np.ones(obs.shape[::2], bool), fut_mask.astype(bool)], axis=1)
    d, dv = lagged(p, v, lag)                                              # 1 s velocity
    sp = np.linalg.norm(d, axis=-1)
    acc, av = lagged(d, dv, lag)
    head = np.arctan2(d[..., 1], d[..., 0])
    dh = np.abs(np.angle(np.exp(1j * (head[:, lag:] - head[:, :-lag])))) / (lag * DT)
    hv = av & (sp[:, lag:] > 1.0) & (sp[:, :-lag] > 1.0)
    jit = np.linalg.norm(p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2], axis=-1)
    jv = v[:, 2:] & v[:, 1:-1] & v[:, :-2]

    def masked(x, m, f):                                                   # NaN where no valid step
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return f(np.where(m, x, np.nan))

    far = np.where(v, np.linalg.norm(p - p[:, :1], axis=-1), 0.0).max(1)
    return {"speed": masked(sp, dv, lambda z: np.nanmean(z, 1)),
            "accel_p90": masked(np.linalg.norm(acc, axis=-1), av, lambda z: np.nanpercentile(z, 90, axis=1)),
            "yaw_rate": masked(dh, hv, lambda z: np.nanmean(z, 1)),
            "jitter": masked(jit, jv, lambda z: np.sqrt(np.nanmean(z ** 2, 1))),
            "stationary": (far < 1.0).astype(float), "fut_valid": fut_mask.astype(float).mean(1)}


def split_rows(d, mapping: dict, split: str) -> np.ndarray:
    se, sid = d["seq_start_end"], d["scene_id"]
    keep = [i for i, s in enumerate(sid) if mapping.get(str(s)) == split]
    return np.concatenate([np.arange(se[i, 0], se[i, 1]) for i in keep]) if keep else np.zeros(0, int)


def source(d, rows) -> dict:
    f = agent_features(d["obs_traj"][rows], d["pred_traj"][rows], d["fut_mask"][rows])
    return {"type": d["agent_type"][rows].astype(str), **f}


def summarize(src: dict) -> dict:
    out = {}
    for t in sorted(set(src["type"])):
        m = src["type"] == t
        row = {"n": int(m.sum())}
        for k in FEATURES:
            x = src[k][m]; x = x[np.isfinite(x)]
            row[k] = ({"mean": float(np.mean(x)), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
                       "n": int(len(x))} if len(x) else None)
        out[t] = row
    return out


def w1(a: dict, b: dict, ta: str, tb: str) -> dict:
    out = {}
    for k in FEATURES:
        x, y = a[k][a["type"] == ta], b[k][b["type"] == tb]
        x, y = x[np.isfinite(x)], y[np.isfinite(y)]
        out[k] = float(wasserstein_distance(x, y)) if len(x) and len(y) else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="chaos-gap dev dataset, e.g. av2_large_dev")
    ap.add_argument("--idd", default=str(REPO / "outputs" / "canonical_world_obs20_pred60.npz"))
    ap.add_argument("--idd-splits", default=str(REPO / "splits" / "scene_splits.json"))
    ap.add_argument("--out", default=None, help="default outputs/chaos_gap/dev/kinematics_<name>.json")
    args = ap.parse_args()
    cg_splits = json.loads((REPO / "splits" / "chaos_gap" / f"{args.name}.json").read_text(encoding="utf-8"))
    assert "DEV ONLY" in cg_splits["note"], "development datasets only: the frozen IDD test stays unread"
    cg, idd = np.load(CG / f"{args.name}.npz", allow_pickle=True), np.load(args.idd, allow_pickle=True)
    cg_map, idd_map = load_splits(REPO / "splits" / "chaos_gap" / f"{args.name}.json"), load_splits(args.idd_splits)
    srcs = {"AV2 train": source(cg, split_rows(cg, cg_map, "train")),
            "AV2 val": source(cg, split_rows(cg, cg_map, "val")),
            "IDD train": source(idd, split_rows(idd, idd_map, "train")),
            "IDD val": source(idd, split_rows(idd, idd_map, "val"))}
    av2_types = set(srcs["AV2 train"]["type"])
    dist = {}
    for s in ("IDD train", "IDD val"):
        for t in sorted(set(srcs[s]["type"])):
            c = AV2_COUNTERPART.get(t)
            dist[f"{s} {t}"] = ({"av2_class": c, "w1_vs_av2_train": w1(srcs[s], srcs["AV2 train"], t, c)}
                                if c in av2_types else {"av2_class": None, "note": "unseen in AV2 training"})
    dist["AV2 val vs AV2 train (reference)"] = {t: w1(srcs["AV2 val"], srcs["AV2 train"], t, t) for t in sorted(av2_types)}
    res = {"label": "development (descriptive; no model outputs)", "dataset": args.name,
           "unit": "agent-window (IDD windows overlap in time)", "dt_s": DT,
           "summary": {s: summarize(v) for s, v in srcs.items()}, "distances": dist}
    out = Path(args.out) if args.out else CG / "dev" / f"kinematics_{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    stat = {k: "mean" if k in ("stationary", "fut_valid") else "median" for k in FEATURES}  # 0/1 -> a fraction
    print(f"{'source':10s} {'class':15s} {'n':>7s} " + " ".join(f"{k + ' ' + stat[k][:3]:>15s}" for k in FEATURES))
    for s, rows in res["summary"].items():
        for t, r in rows.items():
            print(f"{s:10s} {t:15s} {r['n']:7d} " + " ".join(
                f"{r[k][stat[k]]:15.3f}" if r[k] else f"{'-':>15s}" for k in FEATURES))
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
