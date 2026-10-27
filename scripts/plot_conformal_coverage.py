"""Plot per-class conformal coverage: marginal vs class-conditional (regime B).

Reads outputs/benchmark_v2/conformal_metrics.json and renders one dumbbell plot
PER MODEL of per-IDD-3D-class empirical coverage under one global radius
(marginal) versus per-class radii (class-conditional), against the target. Shows
that one radius is per-class unfair and class-conditional pulls the under-covered
classes back toward nominal. Regenerable from the artifact.

Usage:
    python scripts/plot_conformal_coverage.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
METRICS = REPO / "outputs" / "benchmark_v2" / "conformal_metrics.json"
OUT_DIR = REPO / "outputs" / "benchmark_v2"
MIN_N = 250

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"      # marginal (one global radius)
ORANGE = "#eb6834"    # class-conditional (per-class radii)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plot_model(name: str, block: dict, target: float, out_png: Path) -> bool:
    marg = block["marginal"]["per_class_coverage"]
    cc = block["class_conditional"]["per_class_coverage"]
    n = block["marginal"]["per_class_n"]

    classes = [c for c in marg if n.get(c, 0) >= MIN_N]
    if not classes:
        return False
    classes.sort(key=lambda c: marg[c])           # worst-covered at the bottom
    y = range(len(classes))

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

    ax.axvline(target, color=INK2, lw=1.5, ls=(0, (5, 3)), zorder=2)
    ax.annotate(f"{target:.0%} target", xy=(target, -0.5),
                xytext=(target - 0.006, -0.5), fontsize=9, color=INK2,
                va="center", ha="right")

    for i, c in enumerate(classes):
        a, b = marg[c], cc[c]
        ax.plot([a, b], [i, i], color=MUTED, lw=2, zorder=3, solid_capstyle="round")
        ax.plot(a, i, "o", ms=9, mfc=BLUE, mec="white", mew=1.6, zorder=4)
        ax.plot(b, i, "o", ms=9, mfc=ORANGE, mec="white", mew=1.6, zorder=5)

    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{c}  (n={n[c]:,})" for c in classes], fontsize=9.5, color=INK)
    ax.set_ylim(-1.0, len(classes) - 0.2)
    ax.set_xlim(0.72, 1.02)
    ax.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("empirical coverage  (fraction of true positions inside the band)",
                  fontsize=10.5, color=INK2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(axis="both", length=0, labelcolor=INK2)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    # legend via proxy handles (two dots), in the empty upper-left.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none", mfc=BLUE, mec="white", ms=9,
               label="marginal (one global radius)"),
        Line2D([0], [0], marker="o", color="none", mfc=ORANGE, mec="white", ms=9,
               label="class-conditional (per-class radii)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9,
              handletextpad=0.3, borderaxespad=0.6)

    ax.set_title("Class-conditional conformal fixes the coverage one global radius leaves unfair",
                 fontsize=11.5, color=INK, fontweight="bold", pad=16, loc="left")
    ax.annotate(f"{name} on IDD-3D · one 50/50 test-scene fold · {target:.0%} target · "
                "per-class radii pull each class toward nominal",
                xy=(0, 1.02), xycoords="axes fraction", fontsize=9, color=MUTED,
                ha="left", va="bottom")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    return True


def main() -> int:
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    target = m["target_coverage"]
    # CV is served by the stable conformal_coverage.png below (the README's
    # primary figure), so skip its redundant per-model file; every other model
    # gets one conformal_coverage_<slug>.png.
    for name, r in m["models"].items():
        if name == "CV (vel_k=5)":
            continue
        plot_model(name, r["regime_B_test_fold"], target,
                   OUT_DIR / f"conformal_coverage_{_slug(name)}.png")
    # keep a stable primary filename for the README (CV baseline).
    if "CV (vel_k=5)" in m["models"]:
        plot_model("CV (vel_k=5)", m["models"]["CV (vel_k=5)"]["regime_B_test_fold"],
                   target, OUT_DIR / "conformal_coverage.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
