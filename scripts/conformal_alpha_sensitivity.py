"""Alpha-sensitivity of the split-conformal findings across coverage targets.

The headline conformal analysis reports a single operating point (alpha = 0.1,
90% target). That target is the conventional reporting default and a tunable knob,
not a safety-derived spec -- so this script re-runs the wrapper's per-model
evaluation at several targets (default 80/90/95%, alpha 0.2/0.1/0.05) and shows the
four conformal findings replicate across alpha. The nonconformity scores are
alpha-free, so they are computed once (conformal_wrapper.assemble_model_scores) and
re-evaluated per target. Regime B is the mean over --n-folds random 50/50 splits of
the test SCENES (with its 5-95% spread), not a single agent-level fold.

Writes two artifacts to outputs/benchmark_v2/ (the headline conformal_metrics.json
is left untouched):
  conformal_sensitivity.json       -- per-target marginal coverage + mean radius,
                                      per model, both regimes (the source of truth).
  conformal_alpha_sensitivity.png  -- two panels: regime B (scene-fold mean) and
                                      regime A (val->test), coverage vs target.

Usage:
    python scripts/conformal_alpha_sensitivity.py --alphas 0.2 0.1 0.05
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conformal_wrapper import WINDOWS, assemble_model_scores, run_one

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "outputs" / "benchmark_v2"

# Validated light-surface palette (dataviz skill; CVD-checked), matching the other
# benchmark figures. The six models take the first six categorical slots (fixed
# reference order — the map-free transformer is appended as slot 6 green, kept last
# so the color sequence stays the validated reference *prefix*: validate_palette.js
# passes on #fcfcfb, worst adjacent CVD ΔE 9.1; the ADE-position ordering fails
# green↔orange, so entity-color is decoupled from the reading order here). Hue
# carries identity per model; the plot's vertical position carries the ordering.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
REF = "#b9b7b0"

# (conformal_metrics.json key, short display name, categorical hue), model order
# best -> worst by test ADE (order affects legend/reading only, not the encoding).
MODELS = [
    ("Trajectron++ (most-likely, canonical windows)", "Trajectron++", "#2a78d6"),
    ("ST-GAT-inspired (MHA)", "ST-GAT", "#eb6834"),
    ("MLP", "MLP", "#1baf7a"),
    ("K-mode decoder (top-1-sel)", "K-mode", "#eda100"),
    ("CV (vel_k=5)", "CV", "#e87ba4"),
    ("map-free transformer", "Map-free", "#008300"),
    # slots 7-8 appended in fixed order (validate_palette.js on #fcfcfb: all hard
    # checks pass for the 8-slot list; worst adjacent CVD ΔE 9.1; contrast WARN on
    # aqua/yellow/magenta is relieved by the legend + the README table).
    ("AgentFormer (most-likely)", "AgentFormer", "#6250d6"),
    ("LED (most-likely)", "LED", "#e34948"),
]


def compute(alphas, min_cal, fold_seed, n_folds, windows, splits_file, preds_dir) -> dict:
    data = assemble_model_scores(windows, splits_file, preds_dir)
    models = {}
    for name, (s, v, types, scenes, is_val, is_test) in data.items():
        A_cov, A_rad, B_cov, B_lo, B_hi, B_rad = [], [], [], [], [], []
        for a in alphas:
            r = run_one(s, v, types, scenes, is_val, is_test, a, min_cal,
                        fold_seed=int(fold_seed), n_folds=int(n_folds), n_boot=200)
            A = r["regime_A_val_to_test"]["marginal"]
            B = r["regime_B_scene_folds"]["per_step_marginal"]
            A_cov.append(A["coverage"]); A_rad.append(A["mean_radius_m"])
            B_cov.append(B["coverage"]["mean"]); B_rad.append(B["mean_radius_m"]["mean"])
            B_lo.append(B["coverage"]["p05"]); B_hi.append(B["coverage"]["p95"])
        models[name] = {
            "regime_A_val_to_test": {"coverage": A_cov, "mean_radius_m": A_rad},
            "regime_B_scene_folds": {"coverage": B_cov, "coverage_p05": B_lo,
                                     "coverage_p95": B_hi, "mean_radius_m": B_rad},
        }
    return models


def plot(payload: dict, out_png: Path) -> None:
    targets = payload["targets"]
    models = payload["models"]

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    lo, hi = min(targets) - 0.03, max(targets) + 0.03
    panels = [("regime_B_scene_folds",
               "Same distribution — calibrate and deploy on test scenes (regime B)",
               "mean over random 50/50 scene splits; scene clustering leaves it a few points short"),
              ("regime_A_val_to_test",
               "Calibrate on val, deploy on test (regime A)",
               "one draw; lower, but within the spread of regime-B scene splits")]

    for ax, (regime, title, sub) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.plot([lo, hi], [lo, hi], color=REF, lw=1.4, ls=(0, (5, 3)), zorder=2)
        # Regime-B scene-split spread (5-95%, averaged over models) on both panels:
        # the yardstick regime A has to fall outside of to count as a real shift.
        present = [k for k, _, _ in MODELS if k in models]
        b_lo = [sum(models[k]["regime_B_scene_folds"]["coverage_p05"][i] for k in present) / len(present)
                for i in range(len(targets))]
        b_hi = [sum(models[k]["regime_B_scene_folds"]["coverage_p95"][i] for k in present) / len(present)
                for i in range(len(targets))]
        ax.fill_between(targets, b_lo, b_hi, color=GRID, alpha=0.7, lw=0, zorder=1,
                        label="regime-B 5–95% over scene splits")
        for key, disp, col in MODELS:
            if key not in models:
                continue
            cov = models[key][regime]["coverage"]
            ax.plot(targets, cov, "-o", color=col, ms=7, lw=2, mec="white",
                    mew=1.4, zorder=4, label=disp)

        ax.set_xticks(targets)
        ax.set_xticklabels([f"{t:.0%}" for t in targets])
        ax.set_xlim(lo, hi)
        ax.set_ylim(min(0.57, min(b_lo) - 0.02), 1.005)
        ax.set_yticks([0.60, 0.70, 0.80, 0.90])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlabel("coverage target  (1 − α)", fontsize=10.5, color=INK2)
        ax.tick_params(axis="both", length=0, labelcolor=INK2)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.set_title(title, fontsize=10.5, color=INK, fontweight="bold", pad=20, loc="left")
        ax.annotate(sub, xy=(0, 1.015), xycoords="axes fraction", fontsize=9,
                    color=MUTED, ha="left", va="bottom")

    axes[0].set_ylabel("empirical coverage", fontsize=10.5, color=INK2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 8), frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.965), handletextpad=0.4,
               columnspacing=1.6)
    fig.suptitle("Conformal coverage across targets α  ·  "
                 "IDD-3D, deployable models, scene-level folds, targets 80/90/95%",
                 fontsize=12, color=INK, fontweight="bold", x=0.008, ha="left", y=1.03)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Alpha-sensitivity of the conformal findings")
    ap.add_argument("--windows", default=str(WINDOWS))
    ap.add_argument("--splits-file", default="splits/scene_splits.json")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.1, 0.05],
                    help="miscoverage levels; 0.2/0.1/0.05 -> 80/90/95% targets")
    ap.add_argument("--min-cal", type=int, default=30)
    ap.add_argument("--fold-seed", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=100)
    ap.add_argument("--preds-dir", default=str(OUT_DIR))
    ap.add_argument("--save-json", default=str(OUT_DIR / "conformal_sensitivity.json"))
    ap.add_argument("--save-png", default=str(OUT_DIR / "conformal_alpha_sensitivity.png"))
    args = ap.parse_args()

    alphas = sorted(args.alphas, reverse=True)   # loose -> tight target
    models = compute(alphas, args.min_cal, args.fold_seed, args.n_folds, args.windows,
                     args.splits_file, args.preds_dir)
    payload = {
        "alphas": alphas,
        "targets": [round(1.0 - a, 4) for a in alphas],
        "min_cal": args.min_cal, "fold_seed": args.fold_seed, "n_folds": args.n_folds,
        "models": models,
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": "Marginal per-step split-conformal coverage + mean radius per model at "
                "each target. Regime A calibrates on val and evaluates on test; regime B "
                "is the mean (and 5-95% spread) over n_folds random 50/50 splits of the "
                "test scenes. Scores are alpha-free and reused across targets. Headline "
                "conformal_metrics.json (alpha=0.1) is unchanged.",
    }
    out_json = Path(args.save_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved metrics: {out_json}")

    for key, disp, _ in MODELS:
        if key not in models:
            continue
        b = models[key]["regime_B_scene_folds"]["coverage"]
        a = models[key]["regime_A_val_to_test"]["coverage"]
        cells = "  ".join(f"{t:.0%}:B={bc:.3f}/A={ac:.3f}"
                          for t, bc, ac in zip(payload["targets"], b, a))
        print(f"{disp:13s} {cells}")

    plot(payload, Path(args.save_png))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
