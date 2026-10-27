"""Split-conformal uncertainty bands over the deployable predictors (v2).

Post-hoc, no retraining: wraps a point predictor with a per-timestep conformal
radius so the true future position falls inside the band with a guaranteed
(1 - alpha) marginal probability. The frozen splits (splits/scene_splits.json)
make this statistically clean.

Nonconformity score: per-timestep Euclidean error ||pred_t - gt_t|| at valid
future steps (fut_mask). Split-conformal radius at step t is the finite-sample
corrected (1 - alpha) quantile of the calibration scores at t.

Two kinds of band:
  - per-step:      radius r_t from step t's scores alone. Guarantees each STEP
                   is covered w.p. >= 1 - alpha, but NOT that the whole future
                   stays inside (reported as `trajectory_coverage`, which falls
                   well below target -- the honest per-step limitation).
  - simultaneous:  one band r_t = q * sigma_t whose guarantee is for the WHOLE
                   valid horizon of an agent (what a planner needs). sigma_t is a
                   per-step scale fitted on one half of the calibration scenes;
                   q is the conformal quantile of each agent's max_t s_t/sigma_t
                   on the other half, so the score function is fixed before q is
                   calibrated and the guarantee is the standard split-conformal one.

Each kind comes marginal (one band for every agent) and class-conditional
(Mondrian conformal: a separate band per IDD-3D class, motivated by the 16x
per-class error disparity; classes with too few calibration rows fall back to
the marginal band).

Two regimes:
  - A (deployment):  calibrate on val, evaluate coverage on test. Exposes the
                     val->test distribution shift (test is 1.2-1.8x harder), so
                     naive conformal UNDER-covers. Coverage carries a scene-level
                     bootstrap 95% CI over the test scenes (calibration held fixed).
  - B (validity):    split-conformal within test on a seeded 50/50 split of test
                     SCENES (not agents: agents of one scene -- and the up to 7
                     overlapping windows of one agent -- must never sit on both
                     sides). `regime_B_test_fold` is fold `--fold-seed`;
                     `regime_B_scene_folds` summarizes `--n-folds` random scene
                     partitions, since 11-vs-11 scenes is a noisy single draw.

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

from dd_bootstrap import bootstrap_counts, ratio_ci, scene_codes
from dd_split import load_splits

REPO = Path(__file__).resolve().parent.parent
WINDOWS = REPO / "outputs" / "agent_windows_canonical_obs20_pred60.npz"
MLP_WEIGHTS = REPO / "outputs" / "benchmark_v2" / "mlp_best_seed0.npz"
# Trajectron++'s own dump (script 7 --dump-preds) covers every valid node-timestep,
# a different and easier population than the canonical windows of every other
# dump; scripts/eval_trajectronpp_canonical.py writes the matched one.
DENSE_PROTOCOL_DUMPS = {"preds_trajectronpp.npz"}


def canonical_dumps(preds_dir) -> list[Path]:
    """Prediction dumps over the canonical windows (dense-protocol dumps skipped)."""
    return [p for p in sorted(Path(preds_dir).glob("preds_*.npz"))
            if p.name not in DENSE_PROTOCOL_DUMPS]


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


def radii_matrix(classes: np.ndarray, marginal: np.ndarray, per_class: dict | None) -> np.ndarray:
    """(N,T) radius per evaluated row: the class's profile, else the marginal one."""
    if per_class is None:
        return np.broadcast_to(marginal, (classes.shape[0], marginal.shape[0]))
    return np.stack([per_class.get(str(c), marginal) for c in classes]) if len(classes) \
        else np.zeros((0, marginal.shape[0]))


def coverage_report(scores: np.ndarray, valid: np.ndarray, classes: np.ndarray,
                    radii: np.ndarray) -> dict:
    """Empirical coverage + efficiency, pooled / final-step / whole-trajectory /
    per-class. radii (N,T) is each row's band.

    coverage            -- fraction of valid (agent, step) positions inside the band
    trajectory_coverage -- fraction of agents whose EVERY valid step is inside"""
    T = scores.shape[1]
    cov = (scores <= radii) & valid
    n_valid = valid.sum(axis=1)
    has = n_valid > 0
    last = T - 1 - np.argmax(valid[:, ::-1], axis=1)
    rows = np.arange(scores.shape[0])
    final_ok = cov[rows, last][has]
    traj_ok = (cov.sum(axis=1) == n_valid)[has]
    rad_sum = np.where(valid, radii, 0.0)

    per_class = {}
    for c in sorted(set(map(str, classes))):
        m = classes.astype(str) == c
        nv = int(valid[m].sum())
        if nv == 0:
            continue
        per_class[c] = (int(cov[m].sum()), nv, float(rad_sum[m].sum() / nv),
                        float(traj_ok[m[has]].mean()) if (m & has).any() else None)
    tot_n = int(valid.sum())
    return {
        "coverage": float(cov.sum() / tot_n) if tot_n else None,
        "coverage_final_step": float(final_ok.mean()) if final_ok.size else None,
        "trajectory_coverage": float(traj_ok.mean()) if traj_ok.size else None,
        "mean_radius_m": float(rad_sum.sum() / tot_n) if tot_n else None,
        "n_valid_steps": tot_n,
        "n_agents": int(has.sum()),
        "per_class_coverage": {c: v[0] / v[1] for c, v in per_class.items()},
        "per_class_trajectory_coverage": {c: v[3] for c, v in per_class.items()},
        "per_class_mean_radius_m": {c: v[2] for c, v in per_class.items()},
        "per_class_n": {c: v[1] for c, v in per_class.items()},
    }


def _halves(cal_scenes: np.ndarray | None, n: int, rng: np.random.Generator):
    """Split calibration rows into (scale, quantile) halves -- by scene when scene
    ids are given (scenes are the exchangeable unit), else by alternating rows."""
    if cal_scenes is None:
        idx = np.arange(n)
        return idx % 2 == 0, idx % 2 == 1
    scenes = np.asarray(sorted(set(map(str, cal_scenes))), dtype=object)
    first = set(scenes[rng.permutation(len(scenes))[: len(scenes) // 2]])
    a = np.asarray([str(s) in first for s in cal_scenes])
    return a, ~a


def step_scale(scores: np.ndarray, valid: np.ndarray, alpha: float) -> np.ndarray:
    """Per-step scale sigma_t (> 0): the (1-alpha) empirical quantile of the
    step-t scores, gaps filled from the nearest step that has data."""
    T = scores.shape[1]
    sig = np.full(T, np.nan)
    for t in range(T):
        st = scores[valid[:, t], t]
        if st.size:
            sig[t] = np.quantile(st, 1.0 - alpha)
    ok = np.isfinite(sig)
    if not ok.any():
        return np.ones(T)
    sig = np.interp(np.arange(T), np.where(ok)[0], sig[ok])
    return np.maximum(sig, 1e-6)


def max_normalized(scores: np.ndarray, valid: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Per-agent max over valid steps of s_t / sigma_t (rows with no valid step
    get -inf and are dropped by the caller)."""
    z = np.where(valid, scores / sigma, -np.inf)
    return z.max(axis=1)


def simultaneous_bands(cal_scores, cal_valid, cal_classes, cal_scenes, alpha, min_cal, rng):
    """Horizon-wide bands: sigma from one half of the calibration scenes, q (and
    per-class q_c, Mondrian) from the other. Returns (marginal r_t, per-class dict)."""
    a, b = _halves(cal_scenes, cal_scores.shape[0], rng)
    sigma = step_scale(cal_scores[a], cal_valid[a], alpha)
    m = max_normalized(cal_scores[b], cal_valid[b], sigma)
    keep = np.isfinite(m)
    m, cls = m[keep], cal_classes[b][keep]
    q = conformal_radius(m, alpha)
    marginal = q * sigma
    per_class = {}
    for c in np.unique(cls):
        mc = m[cls == c]
        per_class[str(c)] = (conformal_radius(mc, alpha) * sigma
                             if mc.shape[0] >= min_cal else marginal)
    return marginal, per_class


def evaluate_split(cal_scores, cal_valid, cal_classes,
                   ev_scores, ev_valid, ev_classes, alpha, min_cal,
                   cal_scenes=None, rng=None) -> dict:
    """Calibrate on the cal set; evaluate per-step and simultaneous bands, each
    marginal + class-conditional, on the ev set."""
    rng = rng if rng is not None else np.random.default_rng(0)
    marg = radii_per_timestep(cal_scores, cal_valid, alpha)
    pc = radii_per_class(cal_scores, cal_valid, cal_classes, alpha, marg, min_cal)
    s_marg, s_pc = simultaneous_bands(cal_scores, cal_valid, cal_classes, cal_scenes,
                                      alpha, min_cal, rng)
    return {
        "marginal": coverage_report(ev_scores, ev_valid, ev_classes,
                                    radii_matrix(ev_classes, marg, None)),
        "class_conditional": coverage_report(ev_scores, ev_valid, ev_classes,
                                             radii_matrix(ev_classes, marg, pc)),
        "simultaneous": {
            "marginal": coverage_report(ev_scores, ev_valid, ev_classes,
                                        radii_matrix(ev_classes, s_marg, None)),
            "class_conditional": coverage_report(ev_scores, ev_valid, ev_classes,
                                                 radii_matrix(ev_classes, s_marg, s_pc)),
        },
        "marginal_radius_by_step": [float(x) for x in marg],
        "simultaneous_radius_by_step": [float(x) for x in s_marg],
    }


def scene_fold(is_test: np.ndarray, scenes: np.ndarray, rng: np.random.Generator):
    """Seeded 50/50 split of the TEST SCENES -> (cal rows, eval rows)."""
    test_i = np.where(is_test)[0]
    ts = np.asarray(sorted(set(map(str, scenes[test_i]))), dtype=object)
    cal_set = set(ts[rng.permutation(len(ts))[: len(ts) // 2]])
    on_cal = np.asarray([str(s) in cal_set for s in scenes[test_i]])
    return test_i[on_cal], test_i[~on_cal]


def bootstrap_report_ci(ev_scores, ev_valid, ev_scenes, radii, n_boot, seed) -> dict:
    """Scene-bootstrap 95% CIs (eval scenes resampled, band held fixed) for step
    and trajectory coverage of one band."""
    codes, scenes = scene_codes(ev_scenes)
    counts = bootstrap_counts(len(scenes), n_boot, seed)
    cov = (ev_scores <= radii) & ev_valid
    n_valid = ev_valid.sum(axis=1)
    has = (n_valid > 0).astype(float)
    traj = ((cov.sum(axis=1) == n_valid) & (n_valid > 0)).astype(float)
    step = ratio_ci(codes, len(scenes), cov.sum(axis=1), n_valid, counts)
    trj = ratio_ci(codes, len(scenes), traj, has, counts)
    return {"coverage_ci95": step["ci95"], "trajectory_coverage_ci95": trj["ci95"],
            "n_scenes": int(len(scenes)), "n_boot": int(n_boot)}


def _summ(x) -> dict:
    x = np.asarray([v for v in x if v is not None], dtype=float)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95))}


def run_one(s, v, types, scenes, is_val, is_test, alpha, min_cal, fold_seed=0,
            n_folds=100, n_boot=2000) -> dict:
    """Both regimes for one model's per-(agent,step) scores + masks at one alpha.
    Regime A calibrates on val and evaluates on test (exposes the shift); regime B
    is an exchangeable seeded 50/50 split of the test SCENES (valid)."""
    A = evaluate_split(s[is_val], v[is_val], types[is_val],
                       s[is_test], v[is_test], types[is_test], alpha, min_cal,
                       cal_scenes=scenes[is_val], rng=np.random.default_rng(fold_seed))
    marg_t = np.asarray(A["marginal_radius_by_step"])
    A["marginal"].update(bootstrap_report_ci(
        s[is_test], v[is_test], scenes[is_test],
        radii_matrix(types[is_test], marg_t, None), n_boot, fold_seed))
    simt = np.asarray(A["simultaneous_radius_by_step"])
    A["simultaneous"]["marginal"].update(bootstrap_report_ci(
        s[is_test], v[is_test], scenes[is_test],
        radii_matrix(types[is_test], simt, None), n_boot, fold_seed))

    def fold(seed):
        rng = np.random.default_rng(seed)
        fc, fe = scene_fold(is_test, scenes, rng)
        return evaluate_split(s[fc], v[fc], types[fc], s[fe], v[fe], types[fe],
                              alpha, min_cal, cal_scenes=scenes[fc], rng=rng)

    B = fold(fold_seed)
    reps = [B] + [fold(fold_seed + 1 + i) for i in range(max(n_folds - 1, 0))]
    a_cov = A["marginal"]["coverage"]
    folds = {
        "n_folds": len(reps),
        # Where regime A's coverage falls among no-shift scene folds (similar
        # calibration size: ~10 val vs 11 test scenes). A small value means the
        # val->test drop exceeds what scene-sampling variability alone produces.
        "regime_A_coverage_rank_in_folds": float(np.mean(
            [r["marginal"]["coverage"] <= a_cov for r in reps])),
        # Per-class step coverage averaged over the folds where the class is
        # evaluated (one fold's per-class numbers rest on ~11 scenes).
        "per_class_coverage_mean": {
            key: {c: float(np.mean(x)) for c in sorted({c for r in reps for c in r[key]["per_class_coverage"]})
                  for x in [[r[key]["per_class_coverage"][c] for r in reps
                             if c in r[key]["per_class_coverage"]]]}
            for key in ("marginal", "class_conditional")},
        "per_class_n_folds": {c: int(sum(c in r["marginal"]["per_class_coverage"] for r in reps))
                              for c in sorted({c for r in reps for c in r["marginal"]["per_class_coverage"]})},
        "per_step_marginal": {
            "coverage": _summ([r["marginal"]["coverage"] for r in reps]),
            "trajectory_coverage": _summ([r["marginal"]["trajectory_coverage"] for r in reps]),
            "mean_radius_m": _summ([r["marginal"]["mean_radius_m"] for r in reps]),
        },
        "per_step_class_conditional": {
            "coverage": _summ([r["class_conditional"]["coverage"] for r in reps]),
            "mean_radius_m": _summ([r["class_conditional"]["mean_radius_m"] for r in reps]),
        },
        "simultaneous_marginal": {
            "trajectory_coverage": _summ([r["simultaneous"]["marginal"]["trajectory_coverage"]
                                          for r in reps]),
            "mean_radius_m": _summ([r["simultaneous"]["marginal"]["mean_radius_m"] for r in reps]),
        },
        "simultaneous_class_conditional": {
            "trajectory_coverage": _summ([r["simultaneous"]["class_conditional"]["trajectory_coverage"]
                                          for r in reps]),
            "mean_radius_m": _summ([r["simultaneous"]["class_conditional"]["mean_radius_m"]
                                    for r in reps]),
        },
    }
    return {"regime_A_val_to_test": A, "regime_B_test_fold": B, "regime_B_scene_folds": folds}


def assemble_model_scores(windows_path, splits_file, preds_dir) -> dict:
    """Per-model (scores, valid, agent_type, scene_id, is_val, is_test), independent
    of alpha.

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
        out[name] = (s, v, agent_type, scene_id, is_val, is_test)
    # Neural models: consume dumped prediction NPZs if present.
    for npz in canonical_dumps(preds_dir):
        p = np.load(npz, allow_pickle=True)
        if "scene_id" not in p.files:
            raise SystemExit(f"{npz.name} has no scene_id (needed for scene-level folds); "
                             "regenerate it with dump_predictions.py / script 7 --dump-preds")
        sp = p["split"].astype(object)
        s, v = scores_and_valid(p["pred"], p["gt"], p["fut_mask"])
        out[str(p["model"])] = (s, v, p["agent_type"].astype(object),
                                p["scene_id"].astype(object), sp == "val", sp == "test")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Split-conformal wrapper (benchmark v2)")
    ap.add_argument("--windows", default=str(WINDOWS))
    ap.add_argument("--splits-file", default="splits/scene_splits.json")
    ap.add_argument("--alpha", type=float, default=0.1, help="miscoverage; 0.1 -> 90%")
    ap.add_argument("--min-cal", type=int, default=30,
                    help="min per-step calibration rows for a class-conditional radius")
    ap.add_argument("--fold-seed", type=int, default=0, help="regime-B scene split")
    ap.add_argument("--n-folds", type=int, default=100,
                    help="random test-scene partitions summarized for regime B")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="scene-bootstrap replicates for regime-A coverage CIs")
    ap.add_argument("--save-metrics-json",
                    default="outputs/benchmark_v2/conformal_metrics.json")
    args = ap.parse_args()

    data = assemble_model_scores(args.windows, args.splits_file,
                                 Path(args.save_metrics_json).parent)
    results = {
        name: run_one(s, v, types, scenes, is_val, is_test, args.alpha, args.min_cal,
                      fold_seed=int(args.fold_seed), n_folds=int(args.n_folds),
                      n_boot=int(args.n_boot))
        for name, (s, v, types, scenes, is_val, is_test) in data.items()
    }

    for name, r in results.items():
        A, F = r["regime_A_val_to_test"], r["regime_B_scene_folds"]
        print(f"{name}: step cov A(val->test)={A['marginal']['coverage']:.3f} "
              f"B(scene folds)={F['per_step_marginal']['coverage']['mean']:.3f} | "
              f"traj cov per-step band B={F['per_step_marginal']['trajectory_coverage']['mean']:.3f} "
              f"simultaneous B={F['simultaneous_marginal']['trajectory_coverage']['mean']:.3f} "
              f"(target {1 - args.alpha:.2f})")

    payload = {
        "alpha": args.alpha, "target_coverage": 1 - args.alpha,
        "min_cal": args.min_cal, "fold_seed": args.fold_seed,
        "n_folds": args.n_folds, "n_boot": args.n_boot,
        "windows": str(Path(args.windows).name),
        "mlp_weights": str(MLP_WEIGHTS.name),
        "models": results,
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": "Split-conformal radii. Regime A (val->test) exposes the distribution "
                "shift, with scene-bootstrap CIs over test scenes. Regime B splits the "
                "test SCENES 50/50 (exchangeable unit); regime_B_scene_folds summarizes "
                "n_folds random partitions. Per-step bands guarantee each step; "
                "trajectory_coverage = all valid steps inside. 'simultaneous' bands "
                "(r_t = q*sigma_t, max-normalized score, sigma from half the calibration "
                "scenes) target trajectory-level coverage. class_conditional = per-IDD-3D-"
                "class (Mondrian) radii with marginal fallback below min_cal. Predictors "
                "reproduced in numpy from committed artifacts.",
    }
    out = Path(args.save_metrics_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved metrics: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
