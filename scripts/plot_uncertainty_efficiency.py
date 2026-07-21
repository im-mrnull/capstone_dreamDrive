"""Plot the accuracy->uncertainty link across the five deployable predictors.

Two figures, both regenerable from committed artifacts (like the tables):

  uncertainty_efficiency.png  -- deployable test ADE (x) vs the split-conformal
      band radius at the 90% coverage target (y). At a fixed coverage target the
      radius is pure uncertainty efficiency, and the expected trend holds: a
      lower-ADE model buys a tighter band.
  robustness_gap.png          -- coverage lost per model when calibration stops
      matching deployment (regime B minus regime A). Weaker predictors lose far
      more, so the robustness gap shrinks as the base forecaster improves.

ADE comes from outputs/benchmark_v2/summary.json (deployable test_ade_mean);
radius + both-regime coverage come from outputs/benchmark_v2/conformal_metrics.json
(regime B test-fold marginal and regime A val->test marginal).

Usage:
    python scripts/plot_uncertainty_efficiency.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SUMMARY = REPO / "outputs" / "benchmark_v2" / "summary.json"
METRICS = REPO / "outputs" / "benchmark_v2" / "conformal_metrics.json"
OUT_DIR = REPO / "outputs" / "benchmark_v2"

# Validated light-surface palette (dataviz skill; CVD-checked), matching the
# other benchmark figures so the three read as one system.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"

# conformal_metrics.json key -> (summary.json label needles, blocked substrings,
# short display name). The summary labels use em dashes and variant wording, so
# match on distinctive substrings rather than the exact string.
MODELS = [
    ("CV (vel_k=5)", ["CV"], "CV"),
    ("MLP", ["MLP"], "MLP"),
    ("K-mode decoder (top-1-sel)", ["K-mode", "top-1-selected"], "K-mode"),
    ("ST-GAT-inspired (MHA)", ["ST-GAT"], "ST-GAT"),
    ("Trajectron++ (most-likely)", ["Trajectron++", "most-likely"], "Trajectron++"),
]
_BLOCKED = ("hetero", "oracle")


def _find_ade(summ_models: list, needles: list) -> float:
    cand = [m for m in summ_models
            if m.get("deployable")
            and all(nd in m["label"] for nd in needles)
            and not any(b in m["label"] for b in _BLOCKED)]
    assert len(cand) == 1, f"{needles}: {[c['label'] for c in cand]}"
    return float(cand[0]["test_ade_mean"])


def load_rows():
    summ = json.loads(SUMMARY.read_text(encoding="utf-8"))["models"]
    conf = json.loads(METRICS.read_text(encoding="utf-8"))
    target = conf["target_coverage"]
    rows = []
    for ckey, needles, disp in MODELS:
        b = conf["models"][ckey]["regime_B_test_fold"]["marginal"]
        a = conf["models"][ckey]["regime_A_val_to_test"]["marginal"]
        rows.append({
            "name": disp,
            "ade": _find_ade(summ, needles),
            "radius": float(b["mean_radius_m"]),
            "gap": 100.0 * (float(b["coverage"]) - float(a["coverage"])),
        })
    return rows, target


def _fig(w=8.2, h=5.6):
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def plot_efficiency(rows, target, out_png: Path) -> None:
    xs = [r["ade"] for r in rows]
    ys = [r["radius"] for r in rows]

    # least-squares fit + Pearson r for the trend guide and the readout.
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    slope, intercept = sxy / sxx, my - (sxy / sxx) * mx
    r = sxy / (sxx * syy) ** 0.5

    fig, ax = _fig()
    x0, x1 = 2.3, 4.6
    ax.plot([x0, x1], [slope * x0 + intercept, slope * x1 + intercept],
            color=MUTED, lw=1.4, ls=(0, (5, 3)), zorder=2)
    for rw in rows:
        ax.plot(rw["ade"], rw["radius"], "o", ms=11, mfc=BLUE, mec="white",
                mew=2, zorder=5)

    # direct labels (hand-placed to dodge the MLP/K-mode near-overlap at ~13.8 m).
    off = {"Trajectron++": (0, 13, "center", "bottom"),
           "ST-GAT": (12, 0, "left", "center"),
           "MLP": (-11, -9, "right", "center"),
           "K-mode": (12, 6, "left", "center"),
           "CV": (0, 13, "center", "bottom")}
    for rw in rows:
        dx, dy, ha, va = off[rw["name"]]
        ax.annotate(f"{rw['name']}\n{rw['radius']:.1f} m",
                    xy=(rw["ade"], rw["radius"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=9, color=INK, ha=ha,
                    va=va, linespacing=1.25, fontweight="bold")

    ax.annotate(f"lower ADE → smaller band   (r = {r:.2f})", xy=(0.97, 0.06),
                xycoords="axes fraction", fontsize=9.5, color=INK2, ha="right",
                va="bottom", fontstyle="italic")

    ax.set_xlim(x0, x1)
    ax.set_ylim(9.2, 16.9)
    ax.set_xlabel("deployable test ADE (m)  —  lower is a better forecaster",
                  fontsize=10.5, color=INK2)
    ax.set_ylabel(f"conformal band radius (m) at the {target:.0%} target",
                  fontsize=10.5, color=INK2)
    ax.tick_params(axis="both", length=0, labelcolor=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    ax.set_title("Uncertainty efficiency: a more accurate model needs a smaller "
                 "band for the same coverage", fontsize=11.5, color=INK,
                 fontweight="bold", pad=16, loc="left")
    ax.annotate("IDD-3D test split · split-conformal band radius at the 90% target "
                "(regime B, valid) vs deployable test ADE · one point per model",
                xy=(0, 1.02), xycoords="axes fraction", fontsize=9, color=MUTED, ha="left")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_robustness(rows, out_png: Path) -> None:
    order = sorted(rows, key=lambda rw: rw["ade"])   # best forecaster first
    y = list(range(len(order)))[::-1]                # -> best at the top

    fig, ax = _fig(8.2, 5.2)
    ax.barh(y, [rw["gap"] for rw in order], height=0.62, color=BLUE, zorder=3)
    for yi, rw in zip(y, order):
        ax.text(rw["gap"] + 0.3, yi, f"{rw['gap']:.1f} pts", va="center",
                ha="left", fontsize=9.5, color=INK, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{rw['name']}   ·   ADE {rw['ade']:.2f}" for rw in order],
                       fontsize=9.5, color=INK)
    ax.set_xlim(0, max(rw["gap"] for rw in order) * 1.18)
    ax.set_xlabel("coverage lost under the val→test shift  (percentage points, "
                  "regime B − regime A)", fontsize=10.5, color=INK2)
    ax.tick_params(axis="both", length=0, labelcolor=INK2)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    ax.set_title("Robustness gap: weaker predictors lose far more coverage under "
                 "the val→test shift", fontsize=11.5, color=INK, fontweight="bold",
                 pad=16, loc="left")
    ax.annotate("IDD-3D · drop from regime B (calibration matches deployment, ≈90%) "
                "to regime A (calibrate on val, deploy on harder test) · shorter is "
                "more robust", xy=(0, 1.02), xycoords="axes fraction", fontsize=9,
                color=MUTED, ha="left")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> int:
    rows, target = load_rows()
    plot_efficiency(rows, target, OUT_DIR / "uncertainty_efficiency.png")
    plot_robustness(rows, OUT_DIR / "robustness_gap.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
