"""Split-conformal uncertainty bands over the deployable predictors (v2).

Post-hoc, no retraining: wraps a point predictor with a per-timestep conformal
radius so the true future position falls inside the band with a guaranteed
(1 - alpha) marginal probability. The frozen splits (splits/scene_splits.json)
make this statistically clean.

Nonconformity score: per-timestep Euclidean error ||pred_t - gt_t|| at valid
future steps (fut_mask). Split-conformal radius at step t is the finite-sample
corrected (1 - alpha) quantile of the calibration scores at t.

Two variants:
  - marginal:          one radius profile r_t for every agent.
  - class-conditional: a separate r_t^c per IDD-3D class (novelty: the 16x
                       per-class error disparity means one radius is far too
                       loose for pedestrians and too tight for cars/riders);
                       classes with too few calibration samples fall back to
                       the marginal radius.

Two regimes:
  - A (deployment):  calibrate on val, evaluate coverage on test. Exposes the
                     val->test distribution shift (test is 1.2-1.8x harder), so
                     naive conformal UNDER-covers -- an honest, important result.
  - B (validity):    split-conformal within test (seeded 50/50 agent split);
                     calibration matches deployment, so coverage is nominal --
                     confirms the implementation is correct.

Predictors are reproduced in pure numpy from committed artifacts (agent windows
NPZ + the MLP's saved numpy weights), so this runs on CPU in seconds.

Usage:
    python scripts/conformal_wrapper.py --alpha 0.1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from dd_split import load_splits

REPO = Path(__file__).resolve().parent.parent
WINDOWS = REPO / "outputs" / "agent_windows_canonical_obs20_pred60.npz"
MLP_WEIGHTS = REPO / "outputs" / "benchmark_v2" / "mlp_best_seed0.npz"


# --------------------------------------------------------------------------
# Predictors (numpy; reproduced from committed artifacts)
# --------------------------------------------------------------------------
def cv_predict(obs: np.ndarray, vel_k: int = 5) -> np.ndarray:
    """Constant-velocity extrapolation. obs (N,obs_len,2) -> (N,pred_len=60,2)."""
    obs_len = obs.shape[1]
    k = min(vel_k, obs_len - 1)
    v = (obs[:, -k:, :] - obs[:, -k - 1:-1, :]).mean(axis=1)
    steps = np.arange(1, 61, dtype=obs.dtype).reshape(1, 60, 1)
    return steps * v[:, None, :]


def mlp_predict(obs: np.ndarray, weights: dict) -> np.ndarray:
    """2-layer numpy MLP with the training standardizer. (N,20,2) -> (N,60,2)."""
    n = obs.shape[0]
    x = obs.reshape(n, -1).astype(np.float32)
    h = (x - weights["x_mean"]) / weights["x_std"]
    h = np.maximum(h @ weights["W1"] + weights["b1"], 0.0)
    h = np.maximum(h @ weights["W2"] + weights["b2"], 0.0)
    y = h @ weights["W_out"] + weights["b_out"]
    y = y * weights["y_std"] + weights["y_mean"]
    return y.reshape(n, 60, 2)


# --------------------------------------------------------------------------
# Conformal core
# --------------------------------------------------------------------------
def scores_and_valid(pred: np.ndarray, gt: np.ndarray, fut_mask: np.ndarray):
    """Per-(agent,timestep) nonconformity ||pred-gt|| and the valid mask."""
    s = np.linalg.norm(pred - gt, axis=-1)          # (N,T)
    return s, fut_mask.astype(bool)


def conformal_radius(cal_scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal radius: the ceil((n+1)(1-a))-th order stat."""
    s = np.sort(cal_scores)
    n = s.shape[0]
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    return float("inf") if rank > n else float(s[rank - 1])


def radii_per_timestep(scores: np.ndarray, valid: np.ndarray, alpha: float) -> np.ndarray:
    """Marginal radius r_t for each future step from calibration scores."""
    T = scores.shape[1]
    return np.array([conformal_radius(scores[valid[:, t], t], alpha) for t in range(T)])


def radii_per_class(scores: np.ndarray, valid: np.ndarray, classes: np.ndarray,
                    alpha: float, marginal: np.ndarray, min_cal: int) -> dict:
    """Per-class radius profiles with PER-STEP fallback: a step uses the class's
    own conformal radius where it has >= min_cal calibration rows, else the
    marginal radius for that step (so short-future classes still get their own
    tighter radius at the horizons where data exists)."""
    T = scores.shape[1]
    out = {}
    for c in np.unique(classes):
        m = classes == c
        r = marginal.copy()
        for t in range(T):
            cs = scores[m][valid[m][:, t], t]
            if cs.shape[0] >= min_cal:
                r[t] = conformal_radius(cs, alpha)
        out[str(c)] = r
    return out


def coverage_report(scores: np.ndarray, valid: np.ndarray, classes: np.ndarray,
                    radii_fn) -> dict:
    """Empirical coverage + efficiency, pooled / final-step / per-class.

    radii_fn(agent_class) -> per-timestep radius array for that agent."""
    T = scores.shape[1]
    covered_all, valid_all, radius_all = [], [], []
    final_cov, final_n = 0, 0
    per_class = {}
    for i in range(scores.shape[0]):
        r = radii_fn(str(classes[i]))
        v = valid[i]
        cov = (scores[i] <= r) & v
        covered_all.append(int(cov.sum())); valid_all.append(int(v.sum()))
        radius_all.append(r[v])
        last = np.max(np.where(v)[0]) if v.any() else None
        if last is not None:
            final_cov += int(scores[i, last] <= r[last]); final_n += 1
        pc = per_class.setdefault(str(classes[i]), {"cov": 0, "n": 0, "rad": []})
        pc["cov"] += int(cov.sum()); pc["n"] += int(v.sum()); pc["rad"].append(r[v])
    tot_cov, tot_n = sum(covered_all), sum(valid_all)
    return {
        "coverage": tot_cov / tot_n if tot_n else None,
        "coverage_final_step": final_cov / final_n if final_n else None,
        "mean_radius_m": float(np.concatenate(radius_all).mean()) if radius_all else None,
        "n_valid_steps": tot_n,
        "per_class_coverage": {c: v["cov"] / v["n"] for c, v in per_class.items() if v["n"]},
        "per_class_mean_radius_m": {
            c: float(np.concatenate(v["rad"]).mean()) for c, v in per_class.items() if v["rad"]},
        "per_class_n": {c: v["n"] for c, v in per_class.items()},
    }


def make_radii_fn(marginal: np.ndarray, per_class: dict | None):
    if per_class is None:
        return lambda c: marginal
    return lambda c: per_class.get(c, marginal)


def evaluate_split(cal_scores, cal_valid, cal_classes,
                   ev_scores, ev_valid, ev_classes, alpha, min_cal) -> dict:
    """Calibrate on the cal set, evaluate marginal + class-conditional on ev."""
    marg = radii_per_timestep(cal_scores, cal_valid, alpha)
    pc = radii_per_class(cal_scores, cal_valid, cal_classes, alpha, marg, min_cal)
    return {
        "marginal": coverage_report(ev_scores, ev_valid, ev_classes,
                                    make_radii_fn(marg, None)),
        "class_conditional": coverage_report(ev_scores, ev_valid, ev_classes,
                                             make_radii_fn(marg, pc)),
        "marginal_radius_by_step": [float(x) for x in marg],
    }


def run_one(s, v, types, is_val, is_test, alpha, min_cal, rng) -> dict:
    """Both regimes for one model's per-(agent,step) scores + masks at one alpha.
    Regime A calibrates on val and evaluates on test (exposes the shift); regime B
    is an exchangeable seeded 50/50 test fold (valid)."""
    test_i = np.where(is_test)[0]
    perm = rng.permutation(len(test_i)); half = len(test_i) // 2
    fc, fe = test_i[perm[:half]], test_i[perm[half:]]
    A = evaluate_split(s[is_val], v[is_val], types[is_val],
                       s[is_test], v[is_test], types[is_test], alpha, min_cal)
    B = evaluate_split(s[fc], v[fc], types[fc], s[fe], v[fe], types[fe], alpha, min_cal)
    return {"regime_A_val_to_test": A, "regime_B_test_fold": B}


def assemble_model_scores(windows_path, splits_file, preds_dir) -> dict:
    """Per-model (scores, valid, agent_type, is_val, is_test), independent of alpha.

    CV and MLP are reproduced in numpy from the committed agent windows; the neural
    models are read from the preds_*.npz dumps in preds_dir (dump_predictions.py /
    7_..._--dump-preds). The nonconformity scores do not depend on alpha, so a caller
    can re-evaluate coverage at any number of targets without recomputing them."""
    d = np.load(windows_path, allow_pickle=True)
    obs = d["obs"].astype(np.float32); fut = d["fut"].astype(np.float32)
    fut_mask = d["fut_mask"].astype(bool)
    scene_id = d["scene_id"].astype(object); agent_type = d["agent_type"].astype(object)
    split_of = np.array([load_splits(splits_file).get(str(s), "?") for s in scene_id])
    is_val, is_test = split_of == "val", split_of == "test"
    weights = {k: np.load(MLP_WEIGHTS)[k] for k in np.load(MLP_WEIGHTS).files}

    out = {}
    # CV and MLP: reproduced in numpy from committed artifacts (agent windows).
    for name, predict in (("CV (vel_k=5)", lambda o: cv_predict(o, 5)),
                          ("MLP", lambda o: mlp_predict(o, weights))):
        s, v = scores_and_valid(predict(obs), fut, fut_mask)
        out[name] = (s, v, agent_type, is_val, is_test)
    # Neural models: consume dumped prediction NPZs if present.
    for npz in sorted(Path(preds_dir).glob("preds_*.npz")):
        p = np.load(npz, allow_pickle=True)
        sp = p["split"].astype(object)
        s, v = scores_and_valid(p["pred"], p["gt"], p["fut_mask"])
        out[str(p["model"])] = (s, v, p["agent_type"].astype(object),
                                sp == "val", sp == "test")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Split-conformal wrapper (benchmark v2)")
    ap.add_argument("--windows", default=str(WINDOWS))
    ap.add_argument("--splits-file", default="splits/scene_splits.json")
    ap.add_argument("--alpha", type=float, default=0.1, help="miscoverage; 0.1 -> 90%")
    ap.add_argument("--min-cal", type=int, default=30,
                    help="min per-step calibration rows for a class-conditional radius")
    ap.add_argument("--fold-seed", type=int, default=0, help="regime-B test split")
    ap.add_argument("--save-metrics-json",
                    default="outputs/benchmark_v2/conformal_metrics.json")
    args = ap.parse_args()

    data = assemble_model_scores(args.windows, args.splits_file,
                                 Path(args.save_metrics_json).parent)
    results = {
        name: run_one(s, v, types, is_val, is_test, args.alpha, args.min_cal,
                      np.random.default_rng(int(args.fold_seed)))
        for name, (s, v, types, is_val, is_test) in data.items()
    }

    for name, r in results.items():
        cA = r["regime_A_val_to_test"]["marginal"]["coverage"]
        cB = r["regime_B_test_fold"]["marginal"]["coverage"]
        print(f"{name}: marginal coverage  A(val->test)={cA:.3f}  B(test-fold)={cB:.3f}  "
              f"(target {1 - args.alpha:.2f})")

    payload = {
        "alpha": args.alpha, "target_coverage": 1 - args.alpha,
        "min_cal": args.min_cal, "fold_seed": args.fold_seed,
        "windows": str(Path(args.windows).name),
        "mlp_weights": str(MLP_WEIGHTS.name),
        "models": results,
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": "Split-conformal per-timestep radii. Regime A (val->test) exposes the "
                "distribution shift; Regime B (test-fold) is exchangeable and valid. "
                "class_conditional uses per-IDD-3D-class radii (marginal fallback below "
                "min_cal). Predictors reproduced in numpy from committed artifacts.",
    }
    out = Path(args.save_metrics_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved metrics: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
