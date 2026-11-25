"""Chaos-gap Phase 2 seed table (development): AV2-matched models, seeds 0-2, from the prediction dumps.

For each model on AV2 val (in-domain) and IDD val (zero-shot):
  - per-seed most-likely ADE and best-of-6 (the Phase 1 contract's definitions, cg_phase1_table.metrics);
  - mean and sample SD over the 3 seeds;
  - a scene-bootstrap 95% CI of the seed-averaged ADE (each agent's ADE averaged over seeds, scenes resampled);
  - on IDD val, the paired scene-bootstrap CI of (model - CV) ADE: below 0 means the model beats CV.
Development figures only; the frozen IDD test is not read.

Usage:
    python scripts/cg_seeds_table.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("p1", REPO / "scripts" / "cg_phase1_table.py")
p1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1)
spec = importlib.util.spec_from_file_location("bs", REPO / "scripts" / "dd_bootstrap.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)

DEV = p1.DEV
MODELS = [  # (label, kind, [seed 0, seed 1, seed 2] dumps)
    ("MLP", "point", ["mlp_matched_eval_preds.npz", "mlp_matched_s1_preds.npz", "mlp_matched_s2_preds.npz"]),
    ("ST-GAT", "point", ["stgat_matched_eval_preds.npz", "stgat_matched_s1_eval_preds.npz", "stgat_matched_s2_eval_preds.npz"]),
    ("Transformer", "point", ["transformer_matched_eval_preds.npz", "transformer_matched_s1_eval_preds.npz",
                              "transformer_matched_s2_eval_preds.npz"]),
    ("MFP", "modes", ["mfp_matched_preds.npz", "mfp_matched_s1_preds.npz", "mfp_matched_s2_preds.npz"]),
    ("LED", "samples", ["led_matched_eval_preds.npz", "led_matched_s1_preds.npz", "led_matched_s2_preds.npz"]),
    ("AgentFormer", "samples", ["agentformer_matched_eval_preds.npz", "agentformer_matched_s1_preds.npz",
                                "agentformer_matched_s2_preds.npz"]),
    ("AutoBot", "modes", ["autobot_matched_ent8_x4_eval_preds.npz", "autobot_matched_s1_preds.npz", "autobot_matched_s2_preds.npz"]),
]
N_BOOT = 2000


def agent_ade(pred, gt, fm) -> np.ndarray:
    err = np.linalg.norm(pred - gt, axis=-1)
    return np.array([e[f].mean() for e, f in zip(err, fm)])


def main() -> int:
    data, splits = p1.vb.CG / "av2_matched_dev.npz", p1.vb.CG_SPLITS
    cv = p1.load(DEV / "cv_matched_preds.npz", data, splits, True)
    keys, split = cv[0], cv[1]
    rows = []
    for s, name in (("val", "AV2 val"), ("test", "IDD val")):
        sel = split == s
        codes, scenes = bs.scene_codes(np.array([k[0] for k in keys])[sel])
        counts = bs.bootstrap_counts(len(scenes), N_BOOT, seed=0)
        ones = np.ones(sel.sum())
        cv_ade = agent_ade(cv[2][sel], cv[3][sel], cv[4][sel])
        cv_ci = bs.ratio_ci(codes, len(scenes), cv_ade, ones, counts)
        rows.append({"model": "CV (vel_k = 5)", "evaluated_on": name, "n": int(sel.sum()), "n_scenes": len(scenes),
                     "ade_mean": cv_ci["estimate"], "ade_ci95": cv_ci["ci95"]})
        for label, kind, files in MODELS:
            per_seed, ades = [], []
            for f in files:
                k2, _, pred, gt, fm, sets, probs = p1.load(DEV / f, data, splits, False)
                assert k2 == keys, f"{f}: row order differs from CV"
                m = p1.metrics(pred[sel], gt[sel], fm[sel], None if sets is None else sets[sel],
                               None if probs is None else probs[sel], kind)
                per_seed.append({"file": f, "ade": m["ade"], "fde": m["fde"], "best_of_6": m.get("best_of_6")})
                ades.append(agent_ade(pred[sel], gt[sel], fm[sel]))
            mean_ade = np.mean(ades, axis=0)
            c = bs.ratio_ci(codes, len(scenes), mean_ade, ones, counts)
            a = np.array([r["ade"] for r in per_seed])
            row = {"model": label, "evaluated_on": name, "n": int(sel.sum()), "n_scenes": len(scenes), "seeds": per_seed,
                   "ade_mean": float(a.mean()), "ade_sd": float(a.std(ddof=1)), "ade_ci95": c["ci95"]}
            b6 = [r["best_of_6"] for r in per_seed if r["best_of_6"] is not None]
            if len(b6) == 3:
                row["best_of_6_mean"], row["best_of_6_sd"] = float(np.mean(b6)), float(np.std(b6, ddof=1))
            d = bs.ratio_ci(codes, len(scenes), mean_ade - cv_ade, ones, counts)
            row["minus_cv"], row["minus_cv_ci95"] = d["estimate"], d["ci95"]
            rows.append(row)
    out = DEV / "phase2_seeds_table.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{'model':15s} {'eval':8s} {'s0':>6s} {'s1':>6s} {'s2':>6s} {'mean':>6s} {'sd':>5s} {'CI95':>15s} {'bo6':>6s} {'-CV':>7s} {'CI95':>17s}")
    for r in rows:
        s = r.get("seeds", [])
        print(f"{r['model']:15s} {r['evaluated_on']:8s} " + " ".join(f"{x['ade']:6.3f}" for x in s).ljust(20)
              + f" {r['ade_mean']:6.3f} " + (f"{r['ade_sd']:5.3f}" if "ade_sd" in r else "     ")
              + f" [{r['ade_ci95'][0]:.2f}, {r['ade_ci95'][1]:.2f}] "
              + (f"{r['best_of_6_mean']:6.3f}" if "best_of_6_mean" in r else "     -")
              + (f" {r['minus_cv']:+7.3f} [{r['minus_cv_ci95'][0]:+.2f}, {r['minus_cv_ci95'][1]:+.2f}]" if "minus_cv" in r else ""))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
