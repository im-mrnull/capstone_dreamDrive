"""Scene-level bootstrap confidence intervals for the benchmark-v2 test numbers.

The test split is 22 scenes. Agents within a scene are correlated (shared ego
drive, layout, interactions), so the honest resampling unit is the scene. For
every deployable predictor this reports, on the test split:

  * ADE / FDE with a 95% percentile CI from resampling test scenes;
  * paired differences between models adjacent in the ADE ranking (and vs the
    best model): the same scene resample is applied to both models, so the CI
    answers "is model A really better than model B on this test set?";
  * per-class ADE CIs and a CI on the worst/best class ADE disparity (the
    fairness-audit headline).

Point estimates are recomputed from per-agent predictions (seed 0: CV/MLP in
numpy from committed artifacts, neural models from the preds_*.npz dumps) and
cross-checked against the seed-0 metrics JSONs that feed the README table. The
CIs cover scene-sampling uncertainty for one trained model; seed-to-seed spread is
the separate +/- std already in the benchmark table.

Seed-pooled comparisons: for models that also have seed-1/2 dumps
(<preds_dir>/seed_dumps/seed{1,2}/, same file names; see dump_predictions.py
--seed and model7 --eval-only), the comparison is between 3-seed means, and each
replicate resamples the test scenes (shared, so still paired) AND each model's
seeds with replacement, so the CI carries training-seed variation too.

Usage:
    python scripts/bootstrap_ci.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from conformal_wrapper import MLP_WEIGHTS, WINDOWS, canonical_dumps, cv_predict, mlp_predict
from dd_bootstrap import bootstrap_counts, ci, per_scene_sums, ratio_replicates, scene_codes
from dd_data import masked_ade_fde_np
from dd_split import load_splits
from run_benchmark_v2 import OUT, VARIANTS, _dig

SEED_DUMPS = "seed_dumps"   # <preds_dir>/seed_dumps/seed{s}/preds_*.npz for seeds != 0

# preds_*.npz / numpy model name -> VARIANTS label of its seed-0 deployable row.
VARIANT_OF = {
    "CV (vel_k=5)": "CV (vel_k=5)",
    "MLP": "MLP",
    "ST-GAT-inspired (MHA)": "ST-GAT-inspired (MHA)",
    "K-mode decoder (top-1-sel)": "K-mode decoder (top-1-selected) — learned top-1",
    "map-free transformer": "Map-free transformer",
    "Trajectron++ (most-likely, canonical windows)": "Trajectron++ — most-likely",
    "AgentFormer (most-likely)": "AgentFormer (upstream, map-free, 2 Hz) — most-likely",
    "LED (most-likely)": "LED (leapfrog diffusion) — mean-denoised",
}


def seed0_test_ade(name: str):
    """Test ADE of the seed-0 (or seedless) metrics JSON for this model, if any."""
    for label, pattern, subpath, _, _ in VARIANTS:
        if label == VARIANT_OF.get(name):
            f = OUT / pattern.format(seed=0)
            if f.exists():
                return float(_dig(json.loads(f.read_text(encoding="utf-8"))["test"], subpath)["ade"])
    return None


def test_rows(windows_path, splits_file, preds_dir) -> dict:
    """model -> (ade, fde, path, agent_type, scene_id) per test row."""
    d = np.load(windows_path, allow_pickle=True)
    obs = d["obs"].astype(np.float32); fut = d["fut"].astype(np.float32)
    fm = d["fut_mask"].astype(bool)
    scene = d["scene_id"].astype(object); typ = d["agent_type"].astype(object)
    split = np.array([load_splits(splits_file).get(str(s), "?") for s in scene])
    t = split == "test"
    w = np.load(MLP_WEIGHTS)
    weights = {k: w[k] for k in w.files}
    out = {}
    for name, pred in (("CV (vel_k=5)", cv_predict(obs[t], 5)),
                       ("MLP", mlp_predict(obs[t], weights))):
        out[name] = (*masked_ade_fde_np(pred, fut[t], fm[t]), typ[t], scene[t])
    for npz in canonical_dumps(preds_dir):
        p = np.load(npz, allow_pickle=True)
        if "scene_id" not in p.files:
            raise SystemExit(f"{npz.name} has no scene_id; regenerate the dump")
        t = p["split"].astype(object) == "test"
        out[str(p["model"])] = (*masked_ade_fde_np(p["pred"][t], p["gt"][t], p["fut_mask"][t]),
                                p["agent_type"][t].astype(object), p["scene_id"][t].astype(object))
    return out


def seed_pooled(preds_dir: Path, scenes: np.ndarray, counts: np.ndarray, rng_seed: int,
                seeds=(0, 1, 2)) -> dict | None:
    """Paired comparisons of 3-seed means for every model with a test dump per seed.

    Seed 0 is the dump in preds_dir, other seeds live in preds_dir/seed_dumps/seed{s}/
    under the same file name. Each replicate reuses the shared scene counts and draws
    each model's seeds with replacement (independently per model)."""
    S, B, k = len(scenes), counts.shape[0], len(seeds)
    rng = np.random.default_rng(rng_seed + 1)
    per_model = {}
    for npz0 in canonical_dumps(preds_dir):
        files = [npz0] + [preds_dir / SEED_DUMPS / f"seed{s}" / npz0.name for s in seeds[1:]]
        if not all(f.exists() for f in files):
            continue
        num, den, name = [], None, None
        for f in files:
            p = np.load(f, allow_pickle=True)
            t = p["split"].astype(object) == "test"
            ade = masked_ade_fde_np(p["pred"][t], p["gt"][t], p["fut_mask"][t])[0]
            codes, _ = scene_codes(p["scene_id"][t].astype(object), scenes)
            n = per_scene_sums(codes, S, np.ones_like(ade))
            assert den is None or np.array_equal(n, den), f"{f}: rows differ across seeds"
            den, name = n, str(p["model"])
            num.append(per_scene_sums(codes, S, ade))
        num = np.stack(num)                                          # (k, S)
        w = rng.multinomial(k, np.full(k, 1.0 / k), size=B) / k      # (B, k) seed resample
        per_model[name] = {
            "per_seed_ade": [float(x.sum() / den.sum()) for x in num],
            "reps_scene_seed": (counts * (w @ num)).sum(1) / (counts @ den),
            "reps_scene": counts @ num.mean(0) / (counts @ den),
        }
    if len(per_model) < 2:
        return None
    for v in per_model.values():
        v["ade"] = float(np.mean(v["per_seed_ade"]))
    ranked = sorted(per_model, key=lambda m: per_model[m]["ade"])
    pairs = []
    for i, a in enumerate(ranked):
        for b in ranked[i + 1:]:
            ma, mb = per_model[a], per_model[b]
            d = ma["reps_scene_seed"] - mb["reps_scene_seed"]
            pairs.append({
                "a": a, "b": b, "ade_diff_a_minus_b": ma["ade"] - mb["ade"],
                "ci95": ci(d), "p_a_not_better": float((d[np.isfinite(d)] >= 0).mean()),
                "ci95_scenes_only": ci(ma["reps_scene"] - mb["reps_scene"]),
                "seed_pairings_a_better": int(sum(x < y for x in ma["per_seed_ade"]
                                                  for y in mb["per_seed_ade"])),
                "seed_pairings": k * k,
            })
    return {"seeds": list(seeds), "ranking_by_ade": ranked,
            "models": {m: {"ade_seed_mean": per_model[m]["ade"],
                           "per_seed_ade": per_model[m]["per_seed_ade"],
                           "ade_ci95_scene_seed": ci(per_model[m]["reps_scene_seed"])}
                       for m in ranked},
            "pairs": pairs}


def main() -> int:
    ap = argparse.ArgumentParser(description="Scene-level bootstrap CIs (benchmark v2, test)")
    ap.add_argument("--windows", default=str(WINDOWS))
    ap.add_argument("--splits-file", default="splits/scene_splits.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-class-n", type=int, default=50)
    ap.add_argument("--save-json", default="outputs/benchmark_v2/bootstrap_ci.json")
    args = ap.parse_args()

    rows = test_rows(args.windows, args.splits_file, Path(args.save_json).parent)
    scenes = np.asarray(sorted({str(s) for r in rows.values() for s in r[4]}), dtype=object)
    S = len(scenes)
    counts = bootstrap_counts(S, args.n_boot, args.seed)   # shared -> paired comparisons

    models, reps = {}, {}
    for name, (ade, fde, path, typ, scn) in rows.items():
        codes, _ = scene_codes(scn, scenes)
        n = per_scene_sums(codes, S, np.ones_like(ade))
        r_ade = ratio_replicates(counts, per_scene_sums(codes, S, ade), n)
        r_fde = ratio_replicates(counts, per_scene_sums(codes, S, fde), n)
        reps[name] = r_ade

        per_class, cls_reps = {}, {}
        for c in sorted(set(map(str, typ))):
            m = typ.astype(str) == c
            if m.sum() < args.min_class_n:
                continue
            nc = per_scene_sums(codes[m], S, np.ones(m.sum()))
            rc = ratio_replicates(counts, per_scene_sums(codes[m], S, ade[m]), nc)
            cls_reps[c] = rc
            per_class[c] = {"n": int(m.sum()), "ade": float(ade[m].mean()), "ade_ci95": ci(rc),
                            "n_scenes": int((nc > 0).sum())}
        disp = None
        if len(per_class) >= 2:
            worst = max(per_class, key=lambda c: per_class[c]["ade"])
            best = min(per_class, key=lambda c: per_class[c]["ade"])
            disp = {"worst": worst, "best": best,
                    "ratio": per_class[worst]["ade"] / per_class[best]["ade"],
                    "ratio_ci95": ci(cls_reps[worst] / cls_reps[best])}
        ref = seed0_test_ade(name)
        models[name] = {
            "n_rows": int(ade.shape[0]), "n_scenes": int((n > 0).sum()),
            "ade": float(ade.mean()), "ade_ci95": ci(r_ade),
            "fde": float(fde.mean()), "fde_ci95": ci(r_fde),
            "seed0_artifact_test_ade": ref,
            "matches_seed0_artifact": None if ref is None else bool(abs(ref - ade.mean()) < 1e-3),
            "per_class": per_class, "ade_disparity": disp,
        }

    ranked = sorted(models, key=lambda k: models[k]["ade"])
    pooled = seed_pooled(Path(args.save_json).parent, scenes, counts, args.seed)

    def paired(a, b):
        diff = reps[a] - reps[b]
        diff = diff[np.isfinite(diff)]
        return {"a": a, "b": b, "ade_diff_a_minus_b": models[a]["ade"] - models[b]["ade"],
                "ci95": ci(diff), "p_a_not_better": float((diff >= 0).mean())}

    payload = {
        "split": "test", "n_scenes": S, "n_boot": args.n_boot, "seed": args.seed,
        "level": 0.95, "ranking_by_ade": ranked, "models": models,
        "adjacent_pairs": [paired(ranked[i], ranked[i + 1]) for i in range(len(ranked) - 1)],
        "vs_best": [paired(ranked[0], m) for m in ranked[1:]],
        "seed_pooled": pooled,
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": "Scene-level percentile bootstrap over the test scenes (resampled with "
                "replacement; statistics are ratios of per-scene sums). One trained model per "
                "row (seed 0); seed spread is reported separately in the benchmark table. "
                "Paired rows reuse the same scene resample for both models; p_a_not_better = "
                "share of replicates where a's ADE >= b's. Every row is scored on the same "
                "canonical test windows. seed_pooled compares 3-seed means, resampling scenes "
                "(shared) and each model's seeds (with replacement) in every replicate.",
    }
    out = Path(args.save_json)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name in ranked:
        m = models[name]
        print(f"{name:32s} ADE {m['ade']:.3f} [{m['ade_ci95'][0]:.3f}, {m['ade_ci95'][1]:.3f}]  "
              f"FDE {m['fde']:.3f} [{m['fde_ci95'][0]:.3f}, {m['fde_ci95'][1]:.3f}]  "
              f"seed0-artifact match: {m['matches_seed0_artifact']}")
    for p in payload["adjacent_pairs"]:
        print(f"  {p['a']} vs {p['b']}: dADE {p['ade_diff_a_minus_b']:+.3f} "
              f"CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  P(a not better)={p['p_a_not_better']:.3f}")
    if pooled:
        print("seed-pooled (3-seed means; scenes + seeds resampled):")
        for p in pooled["pairs"]:
            print(f"  {p['a']} vs {p['b']}: dADE {p['ade_diff_a_minus_b']:+.3f} "
                  f"CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  P(a not better)="
                  f"{p['p_a_not_better']:.3f}  seed pairings a better: "
                  f"{p['seed_pairings_a_better']}/{p['seed_pairings']}")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
