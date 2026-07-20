"""Benchmark v2 orchestrator: run all models x seeds, then aggregate.

Runs against the canonical dataset + frozen splits. Per-run metrics land in
outputs/benchmark_v2/<model>_seed<k>.json (CV is deterministic: one run).
Existing metric files are SKIPPED, so an interrupted overnight run resumes
where it left off. Aggregation writes outputs/benchmark_v2/summary.json with
mean +/- std over seeds for every headline metric plus pooled per-class stats.

Usage:
    python scripts/run_benchmark_v2.py --smoke     # reduced epochs, plumbing check
    python scripts/run_benchmark_v2.py             # full 3-seed benchmark (~7 h GPU)
    python scripts/run_benchmark_v2.py --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "benchmark_v2"
SEEDS = [0, 1, 2]

# Trajectron++ heterogeneous variant (VEHICLE=Unicycle for wheeled classes).
# Runs only if the hetero export exists (scripts/6_... --hetero).
# TPP_HETERO_AUGMENT is the winner of the 50-epoch val-only grid
# {no-aug, rotation-aug x8}; see outputs/benchmark_v2/tpp_hetero_grid/.
TPP_HETERO_DIR = "outputs/trajectronpp_world_hetero"
TPP_HETERO_AUGMENT = True


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def launch_all(smoke: bool) -> None:
    py = sys.executable
    OUT.mkdir(parents=True, exist_ok=True)

    mlp_epochs = "20" if smoke else "300"
    gnn_epochs = "20" if smoke else "300"
    tpp_epochs = "2" if smoke else "100"

    # CV: deterministic, single run.
    cv_json = OUT / "cv_velk5.json"
    if not cv_json.exists():
        run([py, "scripts/model1_baseline_cv_eval.py", "--vel-k", "5",
             "--save-metrics-json", str(cv_json)])

    for seed in SEEDS:
        s = str(seed)

        mlp_json = OUT / f"mlp_seed{seed}.json"
        if not mlp_json.exists():
            run([py, "scripts/model2_baseline_mlp_train_eval.py",
                 "--seed", s, "--epochs", mlp_epochs,
                 "--save-metrics-json", str(mlp_json),
                 "--save-best", str(OUT / f"mlp_best_seed{seed}.npz")])

        stgat_json = OUT / f"stgat_seed{seed}.json"
        if not stgat_json.exists():
            run([py, "scripts/model3_stgat_train_eval.py",
                 "--device", "cuda", "--seed", s, "--epochs", gnn_epochs,
                 "--save-metrics-json", str(stgat_json),
                 "--save-ckpt", str(OUT / f"stgat_ckpt_seed{seed}.pt")])

        mfp_json = OUT / f"mfp_seed{seed}.json"
        if not mfp_json.exists():
            run([py, "scripts/model4_mfp_train_eval.py",
                 "--device", "cuda", "--seed", s, "--epochs", gnn_epochs,
                 "--k", "10",
                 "--save-metrics-json", str(mfp_json),
                 "--save-ckpt", str(OUT / f"mfp_ckpt_seed{seed}.pt")])

        tpp_json = OUT / f"trajectronpp_seed{seed}.json"
        if not tpp_json.exists():
            run([py, "scripts/7_train_eval_trajectronpp.py",
                 "--seed", s, "--epochs", tpp_epochs,
                 "--log-tag", f"_idd3d_world_v2_s{seed}",
                 "--save-metrics-json", str(tpp_json)])

        # K-mode with checkpoint selection on val top-1 (deployable) rather than
        # val oracle. Winner of the EWTA grid (outputs/benchmark_v2/mfp_ewta_grid/):
        # EWTA annealing lost on both metrics; selection criterion was the fix.
        mfp_t1_json = OUT / f"mfp_top1sel_seed{seed}.json"
        if not mfp_t1_json.exists():
            run([py, "scripts/model4_mfp_train_eval.py",
                 "--device", "cuda", "--seed", s, "--epochs", gnn_epochs,
                 "--k", "10", "--select", "top1",
                 "--save-metrics-json", str(mfp_t1_json),
                 "--save-ckpt", str(OUT / f"mfp_top1sel_ckpt_seed{seed}.pt")])

        tpp_h_json = OUT / f"trajectronpp_hetero_seed{seed}.json"
        if (REPO / TPP_HETERO_DIR / "train.pkl").exists() and not tpp_h_json.exists():
            cmd = [py, "scripts/7_train_eval_trajectronpp.py",
                   "--data-dir", TPP_HETERO_DIR,
                   "--seed", s, "--epochs", tpp_epochs,
                   "--log-tag", f"_idd3d_hetero_v2_s{seed}",
                   "--save-metrics-json", str(tpp_h_json)]
            if TPP_HETERO_AUGMENT:
                cmd.append("--augment")
            run(cmd)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

# label -> (file pattern, [path to metric dict inside the JSON], K, deployable)
VARIANTS = [
    ("CV (vel_k=5)", "cv_velk5.json", [], 1, True),
    ("MLP", "mlp_seed{seed}.json", [], 1, True),
    ("ST-GAT-inspired (MHA)", "stgat_seed{seed}.json", [], 1, True),
    ("K-mode decoder — oracle best-of-10", "mfp_seed{seed}.json", ["oracle_best_of_k"], 10, False),
    ("K-mode decoder — learned top-1", "mfp_seed{seed}.json", ["top1"], 10, True),
    ("K-mode decoder (top-1-selected) — oracle best-of-10", "mfp_top1sel_seed{seed}.json", ["oracle_best_of_k"], 10, False),
    ("K-mode decoder (top-1-selected) — learned top-1", "mfp_top1sel_seed{seed}.json", ["top1"], 10, True),
    ("Trajectron++ — most-likely", "trajectronpp_seed{seed}.json", ["most_likely"], 1, True),
    ("Trajectron++ — oracle best-of-20", "trajectronpp_seed{seed}.json", ["best_of_k"], 20, False),
    ("Trajectron++ hetero (unicycle) — most-likely", "trajectronpp_hetero_seed{seed}.json", ["most_likely"], 1, True),
    ("Trajectron++ hetero (unicycle) — oracle best-of-20", "trajectronpp_hetero_seed{seed}.json", ["best_of_k"], 20, False),
]


def _dig(d: dict, keys: list[str]) -> dict:
    for k in keys:
        d = d[k]
    return d


def aggregate() -> dict:
    models = []
    for label, pattern, subpath, k, deployable in VARIANTS:
        seeds_present = []
        per_split: dict[str, dict[str, list[float]]] = {
            "val": {"ade": [], "fde": []}, "test": {"ade": [], "fde": []}}
        per_class_last = None
        seed_list = [None] if "{seed}" not in pattern else SEEDS
        for seed in seed_list:
            f = OUT / (pattern if seed is None else pattern.format(seed=seed))
            if not f.exists():
                continue
            m = json.loads(f.read_text(encoding="utf-8"))
            try:
                for split in ("val", "test"):
                    node = _dig(m[split], subpath)
                    per_split[split]["ade"].append(float(node["ade"]))
                    per_split[split]["fde"].append(float(node["fde"]))
                node_test = _dig(m["test"], subpath)
                per_class_last = node_test.get("per_class")
                seeds_present.append(0 if seed is None else seed)
            except (KeyError, TypeError) as e:
                print(f"warning: {f.name} missing fields for {label}: {e}")
        if not seeds_present:
            models.append({"label": label, "k": k, "deployable": deployable,
                           "status": "pending"})
            continue
        entry = {"label": label, "k": k, "deployable": deployable,
                 "seeds": seeds_present, "status": "ok"}
        for split in ("val", "test"):
            for metric in ("ade", "fde"):
                vals = np.asarray(per_split[split][metric])
                entry[f"{split}_{metric}_mean"] = float(vals.mean())
                entry[f"{split}_{metric}_std"] = float(vals.std(ddof=0))
        entry["per_class_test_seed_last"] = per_class_last
        models.append(entry)

    summary = {"models": models, "seeds": SEEDS,
               "splits_file": "splits/scene_splits.json",
               "data": "outputs/canonical_world_obs20_pred60.npz"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'summary.json'}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="reduced epochs, plumbing check")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    if not args.aggregate_only:
        launch_all(smoke=bool(args.smoke))
    aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
