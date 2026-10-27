"""Pre-registered safety scorecard: which predictor best supports safe navigation?

The rule (candidates, metrics, decision procedure) is fixed in
docs/wiki/safety-scorecard.md and was committed before this script produced any
number; the output records that commit as `rule_commit`.

For every package (8 single-trajectory predictors + 4 multi-future sets) and
seeds 0-2, on the canonical windows:

  * band: per-step split-conformal radius r_t (alpha = 0.1), per class where a
    class has >= 30 val rows at that step (conformal_wrapper.radii_per_class),
    calibrated on val and applied on test. A set's score is the min-over-samples
    error, so its region is the union of discs of radius r_t around its samples.
  * conflict: the agent's true position comes within 5 m of the ego's actual
    position (outputs/ego_trajectories.csv) at the same valid future step.
  * danger score D = min over valid steps of (d_t - 5 m) / r_t, with d_t the
    distance from the ego to the nearest predicted position; flagged at band
    scale lambda when D <= lambda (lambda = 1: the calibrated band).
  * primary P = ROC-AUC of D for conflicts, 3-seed mean, with a seed-pooled scene
    bootstrap (test scenes resampled and shared by every package, each package's
    seeds resampled with replacement); S1 = the same over vulnerable road users.

The decision rule is applied mechanically; latency (tie-break 2) is read from
outputs/benchmark_v2/safety_latency.json (scripts/safety_latency.py).

Inputs: outputs/benchmark_v2/seed_dumps/seed{0,1,2}/preds_*.npz (gitignored; see
dump_predictions.py --save-samples, model6/model7 --eval-only --save-samples and
eval_trajectronpp_canonical.py --dump-dir).

Usage:
    python scripts/safety_scorecard.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from conformal_wrapper import cv_predict, mlp_predict, radii_matrix, radii_per_class, radii_per_timestep
from dd_bootstrap import bootstrap_counts, ci, scene_codes
from dd_data import masked_ade_fde_np
from dd_split import load_splits

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "benchmark_v2"
DUMPS = OUT / "seed_dumps"
CANONICAL = REPO / "outputs" / "canonical_world_obs20_pred60.npz"
EGO = REPO / "outputs" / "ego_trajectories.csv"
SCENES = REPO / "outputs" / "multiagent_world_v1" / "scenes"
RULE_DOC = "docs/wiki/safety-scorecard.md"

OBS_LEN, PRED_LEN = 20, 60
ALPHA, MIN_CAL = 0.1, 30
CONFLICT_M = 5.0
K_SET = 10
SEEDS = (0, 1, 2)
VRU = {"Pedestrian", "Child", "MotorcyleRider", "ScooterRider", "BicycleRider", "BicycleGroup"}
REAL_TIME_MS = 100.0

# (package, kind, source): kind "single" uses the dump's most-likely `pred`,
# "set" its first K_SET `samples`; "cv"/"mlp" are reproduced in numpy.
PACKAGES = [
    ("CV (vel_k=5)", "single", "cv"),
    ("MLP", "single", "mlp"),
    ("ST-GAT (MHA)", "single", "preds_stgat.npz"),
    ("Map-free transformer", "single", "preds_transformer.npz"),
    ("K-mode (top-1)", "single", "preds_mfp.npz"),
    ("AgentFormer (most-likely)", "single", "preds_agentformer.npz"),
    ("LED (mean-denoised)", "single", "preds_led.npz"),
    ("Trajectron++ (most-likely)", "single", "preds_trajectronpp_canonical.npz"),
    ("K-mode set (10 modes)", "set", "preds_mfp.npz"),
    ("AgentFormer set (10 samples)", "set", "preds_agentformer.npz"),
    ("LED set (10 samples)", "set", "preds_led.npz"),
    ("Trajectron++ set (10 samples)", "set", "preds_trajectronpp_canonical.npz"),
]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), REPO / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_population(splits_file) -> dict:
    """Canonical val+test rows in dump order (val then test, NPZ order within a
    split) with ground truth, classes, scenes and the ego's future path."""
    c = np.load(CANONICAL, allow_pickle=True)
    mapping = load_splits(splits_file)
    ego = pd.read_csv(EGO, dtype={"frame_id": str})
    by_frame = {(r.scene_id, int(r.frame_id)): (r.ego_x, r.ego_y) for r in ego.itertuples()}
    ego_xy = {}   # per scene, indexed like the scene's frames (t0 indexes those)
    for sid in {str(x) for x in c["scene_id"]}:
        frames = np.load(SCENES / f"{sid}.npz", allow_pickle=True)["frame_id"]
        ego_xy[sid] = np.array([by_frame[(sid, int(f))] for f in frames], dtype=float)
    rows, split, ego_fut, ego_now = [], [], [], []
    for want in ("val", "test"):
        for q, (a, b) in enumerate(c["seq_start_end"]):
            sid = str(c["scene_id"][q])
            if mapping.get(sid) != want:
                continue
            t0 = int(c["t0"][q])
            e = ego_xy[sid][t0 + OBS_LEN:t0 + OBS_LEN + PRED_LEN]
            if len(e) < PRED_LEN:   # scene ends inside the horizon: agents are masked there too
                e = np.vstack([e, np.repeat(e[-1:], PRED_LEN - len(e), 0)])
            assert e.shape == (PRED_LEN, 2) and np.isfinite(e).all(), (sid, t0)
            for i in range(a, b):
                rows.append(i); split.append(want); ego_fut.append(e)
                ego_now.append(ego_xy[sid][t0 + OBS_LEN - 1])
    rows = np.asarray(rows)
    sid = np.concatenate([[str(c["scene_id"][q])] * int(b - a) for q, (a, b) in enumerate(c["seq_start_end"])])
    return {
        "obs": c["obs_traj"][rows].transpose(0, 2, 1).astype(np.float64),
        "gt": c["pred_traj"][rows].transpose(0, 2, 1).astype(np.float64),
        "fm": c["fut_mask"][rows].astype(bool),
        "cls": c["agent_type"][rows].astype(str),
        "scene": sid[rows],
        "split": np.asarray(split),
        "ego": np.stack(ego_fut),
        "ego_now": np.asarray(ego_now),
    }


def mlp_world(obs: np.ndarray, weights: dict) -> np.ndarray:
    """The MLP was trained on agent windows (centred on the last observation,
    rotated so the recent heading points along +x; script 9). Apply that
    transform, predict, and map the prediction back to the world frame."""
    m9 = _load_script("9_make_agent_windows_from_canonical.py")
    centre = obs[:, -1:, :]
    o = obs - centre
    k = m9.HEADING_K
    R = m9.rotation_matrices((o[:, -k:, :] - o[:, -k - 1:-1, :]).sum(axis=1)).astype(np.float64)
    o_rot = np.einsum("nij,ntj->nti", R, o)
    y_rot = mlp_predict(o_rot.astype(np.float32), weights).astype(np.float64)
    return np.einsum("nji,ntj->nti", R, y_rot) + centre


def package_predictions(kind: str, source: str, seed: int, pop: dict, k_set: int) -> np.ndarray:
    """(N, K, 60, 2) world-frame predictions over pop's rows (K = 1 for single)."""
    if source == "cv":
        return (pop["obs"][:, -1:, :] + cv_predict(pop["obs"], 5))[:, None]
    if source == "mlp":
        w = np.load(OUT / f"mlp_best_seed{seed}.npz")
        return mlp_world(pop["obs"], {k: w[k] for k in w.files})[:, None]
    d = np.load(DUMPS / f"seed{seed}" / source, allow_pickle=True)
    assert np.array_equal(d["split"].astype(str), pop["split"]), f"{source} seed {seed}: row order"
    assert np.array_equal(d["fut_mask"].astype(bool), pop["fm"]), f"{source} seed {seed}: fut_mask"
    if kind == "single":
        return d["pred"].astype(np.float64)[:, None]
    s = d["samples"]
    assert s.shape[1] >= k_set, f"{source}: only {s.shape[1]} samples"
    return s[:, :k_set].astype(np.float64)


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------
def min_sample_dist(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(N,K,T,2) vs (N,T,2) -> (N,T) distance from target to the nearest sample."""
    return np.linalg.norm(pred - target[:, None], axis=-1).min(axis=1)


def conformal_radii(pred, pop, alpha=ALPHA, min_cal=MIN_CAL) -> np.ndarray:
    """(N_test, T) per-step, per-class radii calibrated on val (min-over-samples score)."""
    val = pop["split"] == "val"
    s = min_sample_dist(pred[val], pop["gt"][val])
    marginal = radii_per_timestep(s, pop["fm"][val], alpha)
    per_class = radii_per_class(s, pop["fm"][val], pop["cls"][val], alpha, marginal, min_cal)
    r = radii_matrix(pop["cls"][~val], marginal, per_class)
    assert np.isfinite(r).all() and (r > 0).all(), "non-finite conformal radius"
    return np.asarray(r, dtype=np.float64)


def conflict_labels(gt, fm, ego, radius=CONFLICT_M, horizon=PRED_LEN) -> np.ndarray:
    d = np.linalg.norm(gt - ego, axis=-1)
    v = fm.copy(); v[:, horizon:] = False
    return ((d <= radius) & v).any(axis=1)


def danger_scores(pred, radii, fm, ego, radius=CONFLICT_M, horizon=PRED_LEN) -> np.ndarray:
    """D = min over valid steps of (d_t - radius) / r_t (lower = more dangerous)."""
    z = (min_sample_dist(pred, ego) - radius) / radii
    v = fm.copy(); v[:, horizon:] = False
    return np.where(v, z, np.inf).min(axis=1)


def weighted_auc(score: np.ndarray, label: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """P(score_pos < score_neg) + 0.5 P(tie), rows weighted. weights (B, N) or (N,)."""
    w = np.atleast_2d(weights).astype(np.float64)
    order = np.argsort(score, kind="stable")
    s, y, w = score[order], label[order].astype(bool), w[:, order]
    wp, wn = w * y, w * ~y
    _, start, count = np.unique(s, return_index=True, return_counts=True)
    end = start + count - 1
    grp = np.repeat(np.arange(len(start)), count)
    cn = np.cumsum(wn, axis=1)
    before = np.where(start > 0, cn[:, np.maximum(start - 1, 0)], 0.0)   # neg strictly lower
    upto = cn[:, end]                                                       # neg lower or tied
    neg_gt = cn[:, -1:] - upto[:, grp]
    neg_eq = (upto - before)[:, grp]
    num = (wp * (neg_gt + 0.5 * neg_eq)).sum(axis=1)
    den = wp.sum(axis=1) * wn.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return out if np.ndim(weights) == 2 else out[0]


# --------------------------------------------------------------------------
# Scorecard
# --------------------------------------------------------------------------
def accuracy(pred, gt, fm, near) -> dict:
    """Point (K=1) or oracle min-over-samples (K>1) accuracy readouts."""
    a = np.stack([masked_ade_fde_np(pred[:, k], gt, fm)[0] for k in range(pred.shape[1])], 1).min(1)
    f = np.stack([masked_ade_fde_np(pred[:, k], gt, fm)[1] for k in range(pred.shape[1])], 1).min(1)
    return {"ade": float(a.mean()), "fde": float(f.mean()), "fde_p95": float(np.percentile(f, 95)),
            "miss_rate_2m": float((f > 2.0).mean()),
            "near_ego_ade": float(a[near].mean()), "near_ego_fde": float(f[near].mean())}


def evaluate_package(kind, source, pop, k_set) -> dict:
    test = pop["split"] == "test"
    gt, fm, ego = pop["gt"][test], pop["fm"][test], pop["ego"][test]
    vru = np.isin(pop["cls"][test], list(VRU))
    near = np.where(fm, np.linalg.norm(gt - ego, axis=-1), np.inf).min(axis=1) <= 10.0
    y = conflict_labels(gt, fm, ego)
    per_seed = {k: [] for k in ("D", "auc", "auc_vru", "recall_at_band", "far_at_band",
                                "mean_radius_m", "auc_conflict_3m", "auc_conflict_10m",
                                "auc_horizon_3s", "auc_k20", "accuracy")}
    for seed in SEEDS:
        pred = package_predictions(kind, source, seed, pop, k_set)
        r = conformal_radii(pred, pop)
        pt = pred[test]
        D = danger_scores(pt, r, fm, ego)
        per_seed["D"].append(D)
        per_seed["auc"].append(float(weighted_auc(D, y, np.ones_like(D))))
        per_seed["auc_vru"].append(float(weighted_auc(D[vru], y[vru], np.ones(vru.sum()))))
        per_seed["recall_at_band"].append(float((D[y] <= 1.0).mean()))
        per_seed["far_at_band"].append(float((D[~y] <= 1.0).mean()))
        per_seed["mean_radius_m"].append(float(r[fm].mean()))
        for key, rad, hor in (("auc_conflict_3m", 3.0, PRED_LEN), ("auc_conflict_10m", 10.0, PRED_LEN),
                              ("auc_horizon_3s", CONFLICT_M, 30)):
            yy = conflict_labels(gt, fm, ego, rad, hor)
            dd = danger_scores(pt, r, fm, ego, rad, hor)
            per_seed[key].append(float(weighted_auc(dd, yy, np.ones_like(dd))))
        if kind == "set":
            p20 = package_predictions(kind, source, seed, pop, k_set=20) if source != "preds_mfp.npz" else pred
            r20 = conformal_radii(p20, pop)
            d20 = danger_scores(p20[test], r20, fm, ego)
            per_seed["auc_k20"].append(float(weighted_auc(d20, y, np.ones_like(d20))))
        per_seed["accuracy"].append(accuracy(pt, gt, fm, near))
    mean = lambda k: float(np.mean(per_seed[k])) if per_seed[k] else None   # noqa: E731
    acc = {k: float(np.mean([a[k] for a in per_seed["accuracy"]])) for k in per_seed["accuracy"][0]}
    return {
        "kind": kind, "k": 1 if kind == "single" else k_set, "source": source,
        "auc": mean("auc"), "auc_vru": mean("auc_vru"),
        "recall_at_band": mean("recall_at_band"), "far_at_band": mean("far_at_band"),
        "mean_radius_m": mean("mean_radius_m"),
        "sensitivity": {k: mean(k) for k in ("auc_conflict_3m", "auc_conflict_10m",
                                             "auc_horizon_3s", "auc_k20")},
        "accuracy": acc, "accuracy_is_oracle": kind == "set",
        "per_seed": {k: v for k, v in per_seed.items() if k not in ("D", "accuracy")},
        "_D": per_seed["D"],
    }


def seed_pooled_replicates(D_seeds, y, mask, counts, codes, rng) -> np.ndarray:
    """(B,) bootstrap replicates of the 3-seed-mean AUC over rows `mask`."""
    w = counts[:, codes[mask]].astype(np.float64)                     # scene weights per row
    aucs = np.stack([weighted_auc(D[mask], y[mask], w) for D in D_seeds], 1)   # (B, seeds)
    k = len(D_seeds)
    sw = rng.multinomial(k, np.full(k, 1.0 / k), size=counts.shape[0]) / k
    return (aucs * sw).sum(axis=1)


def paired(reps: dict, a: str, b: str, point: dict) -> dict:
    d = reps[a] - reps[b]
    d = d[np.isfinite(d)]
    lo, hi = ci(d)
    return {"a": a, "b": b, "diff": point[a] - point[b], "ci95": [lo, hi],
            "tied": bool(lo <= 0.0 <= hi)}


def decide(packages: dict, reps_p: dict, reps_s1: dict, latency: dict | None) -> dict:
    p = {n: v["auc"] for n, v in packages.items()}
    s1 = {n: v["auc_vru"] for n, v in packages.items()}
    leader = max(p, key=p.get)
    vs_leader = [paired(reps_p, leader, n, p) for n in packages if n != leader]
    tied = [leader] + [c["b"] for c in vs_leader if c["tied"]]
    steps = [f"Leader on P (conflict AUC): {leader} ({p[leader]:.3f})."]
    if len(tied) == 1:
        steps.append("No package is statistically tied with the leader: it is the pick.")
        return {"leader": leader, "vs_leader": vs_leader, "tied_on_p": tied, "s1_best": None,
                "s1_pairs": [], "tied_on_s1": tied, "pick": leader, "steps": steps}
    steps.append(f"Tied with the leader on P (paired 95% CI includes 0): {', '.join(tied[1:])}.")
    s1_best = max(tied, key=s1.get)
    s1_pairs = [paired(reps_s1, s1_best, n, s1) for n in tied if n != s1_best]
    keep = [s1_best] + [c["b"] for c in s1_pairs if c["tied"]]
    steps.append(f"Tie-break 1 (vulnerable road users): best {s1_best} ({s1[s1_best]:.3f}); "
                 f"still tied: {', '.join(keep[1:]) or 'none'}.")
    pick = keep[0]
    if len(keep) > 1:
        if latency is None:
            steps.append("Tie-break 2 needs outputs/benchmark_v2/safety_latency.json "
                         "(python scripts/safety_latency.py): pick pending.")
            pick = None
        else:
            pick = min(keep, key=lambda n: latency[n]["median_ms"])
            steps.append(f"Tie-break 2 (latency): {pick} at {latency[pick]['median_ms']:.1f} ms.")
    return {"leader": leader, "vs_leader": vs_leader, "tied_on_p": tied, "s1_best": s1_best,
            "s1_pairs": s1_pairs, "tied_on_s1": keep, "pick": pick, "steps": steps}


def exploratory(pop, D_by_package: dict, counts, codes, rng) -> dict:
    """NOT pre-registered, not used for the pick: why the primary metric saturates.
    Emerging conflicts = conflict agents more than 10 m from the ego at the prediction
    time, compared with non-conflict agents that are also more than 10 m away."""
    test = pop["split"] == "test"
    y = conflict_labels(pop["gt"][test], pop["fm"][test], pop["ego"][test])
    d_now = np.linalg.norm(pop["obs"][test][:, -1] - pop["ego_now"][test], axis=1)
    far = d_now > 10.0
    sub = far
    point, reps = {}, {}
    for name, Ds in D_by_package.items():
        point[name] = float(np.mean([weighted_auc(D[sub], y[sub], np.ones(sub.sum())) for D in Ds]))
        reps[name] = seed_pooled_replicates(Ds, y, sub, counts, codes, rng)
    best = max(point, key=point.get)
    return {
        "note": "Exploratory, added after the pre-registered result; not used for the pick.",
        "share_of_conflicts_within_5m_now": float((d_now[y] <= 5.0).mean()),
        "share_of_conflicts_within_10m_now": float((d_now[y] <= 10.0).mean()),
        "auc_current_distance_only": float(weighted_auc(d_now, y, np.ones_like(d_now))),
        "emerging_conflicts": {
            "n_pos": int((y & far).sum()), "n_neg": int((~y & far).sum()),
            "auc": {n: {"auc": point[n], "ci95": ci(reps[n])} for n in point},
            "best": best,
            "vs_best": [paired(reps, best, n, point) for n in point if n != best],
        },
    }


def rule_commit() -> str | None:
    try:
        return subprocess.run(["git", "log", "-1", "--format=%H", "--", RULE_DOC], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-registered safety scorecard")
    ap.add_argument("--splits-file", default=str(REPO / "splits" / "scene_splits.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-set", type=int, default=K_SET)
    ap.add_argument("--save-json", default=str(OUT / "safety_scorecard.json"))
    args = ap.parse_args()

    pop = load_population(args.splits_file)
    test = pop["split"] == "test"
    gt, fm, ego, cls = pop["gt"][test], pop["fm"][test], pop["ego"][test], pop["cls"][test]
    y = conflict_labels(gt, fm, ego)
    vru = np.isin(cls, list(VRU))
    scenes = np.asarray(sorted(set(pop["scene"][test])), dtype=object)
    codes, _ = scene_codes(pop["scene"][test], scenes)
    counts = bootstrap_counts(len(scenes), args.n_boot, args.seed)
    rng = np.random.default_rng(args.seed + 1)

    packages, reps_p, reps_s1, D_by_package = {}, {}, {}, {}
    for name, kind, source in PACKAGES:
        res = evaluate_package(kind, source, pop, args.k_set)
        D = D_by_package[name] = res.pop("_D")
        reps_p[name] = seed_pooled_replicates(D, y, np.ones_like(y), counts, codes, rng)
        reps_s1[name] = seed_pooled_replicates(D, y, vru, counts, codes, rng)
        res["auc_ci95"], res["auc_vru_ci95"] = ci(reps_p[name]), ci(reps_s1[name])
        packages[name] = res
        print(f"{name:30s} AUC {res['auc']:.3f} [{res['auc_ci95'][0]:.3f}, {res['auc_ci95'][1]:.3f}]  "
              f"VRU {res['auc_vru']:.3f}  recall@band {res['recall_at_band']:.3f}  "
              f"FAR@band {res['far_at_band']:.3f}  r {res['mean_radius_m']:.1f} m", flush=True)

    lat_path = OUT / "safety_latency.json"
    latency = json.loads(lat_path.read_text(encoding="utf-8"))["packages"] if lat_path.exists() else None
    decision = decide(packages, reps_p, reps_s1, latency)
    if decision["pick"] and latency:
        decision["pick_real_time"] = bool(latency[decision["pick"]]["median_ms"] <= REAL_TIME_MS)
    for s in decision["steps"]:
        print(" ->", s)

    payload = {
        "rule_doc": RULE_DOC, "rule_commit": rule_commit(),
        "settings": {"alpha": ALPHA, "min_cal": MIN_CAL, "conflict_m": CONFLICT_M,
                     "k_set": args.k_set, "seeds": list(SEEDS), "n_boot": args.n_boot,
                     "vru_classes": sorted(VRU), "real_time_ms": REAL_TIME_MS},
        "population": {"n_test": int(test.sum()), "n_conflicts": int(y.sum()),
                       "n_vru": int(vru.sum()), "n_vru_conflicts": int((y & vru).sum()),
                       "n_conflicts_3m": int(conflict_labels(gt, fm, ego, 3.0).sum()),
                       "n_conflicts_10m": int(conflict_labels(gt, fm, ego, 10.0).sum()),
                       "n_conflicts_3s": int(conflict_labels(gt, fm, ego, CONFLICT_M, 30).sum())},
        "packages": packages,
        "decision": decision,
        "exploratory": exploratory(pop, D_by_package, counts, codes, rng),
        "latency_source": str(lat_path.relative_to(REPO)).replace("\\", "/") if latency else None,
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": "Open-loop proxy: the ego path is what the human driver did and nearby agents "
                "reacted to it. Set packages' accuracy readouts are oracle (min over samples).",
    }
    Path(args.save_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {args.save_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
