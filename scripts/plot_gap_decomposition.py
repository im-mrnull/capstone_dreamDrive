"""Plot the Trajectron++ oracle-vs-deployable gap decomposition.

Reads outputs/benchmark_v2/summary.json (trajectronpp_gap_decomposition, written
by run_benchmark_v2.aggregate) and renders the best-of-K curve with the gap split
into a calibration slice (random draw -> most-likely) and a coverage slice
(most-likely -> oracle). Figure is regenerable from the artifact, like the tables.

Usage:
    python scripts/plot_gap_decomposition.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SUMMARY = REPO / "outputs" / "benchmark_v2" / "summary.json"
OUT_PNG = REPO / "outputs" / "benchmark_v2" / "gap_decomposition.png"

# Validated light-surface palette (dataviz skill; CVD-checked, see commit note).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"      # the best-of-K curve (single series)
AQUA = "#1baf7a"      # calibration band
ORANGE = "#eb6834"    # coverage band


def main() -> int:
    gap = json.loads(SUMMARY.read_text(encoding="utf-8")).get("trajectronpp_gap_decomposition")
    if not gap:
        raise SystemExit("no trajectronpp_gap_decomposition in summary.json; run the benchmark")

    K = gap["k_list"]
    ade = gap["best_of_k_ade_mean"]
    std = gap.get("best_of_k_ade_std", [0.0] * len(K))
    ml = gap["most_likely_ade"]
    rnd = gap["mean_sample_ade"]
    oracle = gap["oracle_best_of_k_ade"]
    calib = gap["calibration_value_m"]
    cover = gap["coverage_value_m"]

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # --- decomposition bands: thickness == the slice value (linear y) ---
    ax.axhspan(oracle, ml, color=ORANGE, alpha=0.14, zorder=0)
    ax.axhspan(ml, rnd, color=AQUA, alpha=0.17, zorder=0)
    # most-likely boundary line between the two slices.
    ax.axhline(ml, color=INK2, lw=1.5, ls=(0, (5, 3)), zorder=2)

    # --- best-of-K curve (single series) ---
    lo = [a - s for a, s in zip(ade, std)]
    hi = [a + s for a, s in zip(ade, std)]
    ax.fill_between(K, lo, hi, color=BLUE, alpha=0.18, lw=0, zorder=3)
    ax.plot(K, ade, color=BLUE, lw=2, zorder=4)
    ax.plot(K, ade, "o", ms=8, mfc=BLUE, mec="white", mew=2, zorder=5)

    # --- direct labels: endpoints (reference readouts) ---
    ax.annotate(f"random single sample\n{rnd:.2f} m",
                xy=(K[0], ade[0]), xytext=(K[0] * 1.08, ade[0] + 0.04),
                fontsize=9.5, color=INK, va="bottom", ha="left", fontweight="bold")
    ax.annotate(f"oracle best-of-{K[-1]}\n{oracle:.2f} m",
                xy=(K[-1], ade[-1]), xytext=(K[-1] * 0.93, ade[-1] - 0.13),
                fontsize=9.5, color=INK, va="top", ha="right", fontweight="bold")
    # intermediate curve values (muted, small).
    for k, a in list(zip(K, ade))[1:-1]:
        ax.annotate(f"{a:.2f}", xy=(k, a), xytext=(k, a + 0.10),
                    fontsize=8.5, color=INK2, ha="center", va="bottom")

    # most-likely reference label: just BELOW the dashed line at the right,
    # into the empty top-right of the coverage band (clear of curve + calib text).
    ax.annotate(f"most-likely (deployable)  {ml:.2f} m",
                xy=(K[-1], ml), xytext=(K[-1] * 0.99, ml - 0.04),
                fontsize=9.5, color=INK2, va="top", ha="right", fontweight="bold")

    # --- band labels (satisfy the aqua contrast relief: labels present) ---
    # calibration in the wide clear centre of its band; coverage in the empty
    # upper-right of its band (the curve occupies the lower-left diagonal).
    ax.text(6.3, (ml + rnd) / 2 + 0.03,
            f"calibration = {calib:.2f} m\nrandom → most-likely\n(mode selection is informative)",
            fontsize=9.5, color=INK, ha="center", va="center", linespacing=1.3)
    ax.text(11, (ml + oracle) / 2 + 0.25 * (ml - oracle),
            f"coverage = {cover:.2f} m\nmost-likely → oracle\n(irreducible multimodality)",
            fontsize=9.5, color=INK, ha="center", va="center", linespacing=1.3)

    # --- axes / chrome ---
    ax.set_xscale("log")
    ax.set_xticks(K)
    ax.set_xticklabels([str(k) for k in K])
    ax.tick_params(axis="both", which="both", length=0, labelcolor=INK2)
    ax.minorticks_off()
    ax.set_xlim(K[0] * 0.92, K[-1] * 1.12)
    ax.set_ylim(oracle - 0.5, rnd + 0.40)
    ax.set_xlabel("K  (samples kept — best-of-K)", fontsize=10.5, color=INK2)
    ax.set_ylabel("ADE (m)  —  lower is better", fontsize=10.5, color=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    # title + subtitle kept within the axes width (wider text squeezes the plot).
    ax.set_title(f"Trajectron++: where the {cover:.2f} m oracle-vs-deployable gap comes from",
                 fontsize=12, color=INK, fontweight="bold", pad=36, loc="left")
    ax.annotate("IDD-3D test split, canonical windows, 3-seed mean. Best-of-K ADE from 20 CVAE\n"
                f"samples (same draws); the curve still falls at K={K[-1]} — genuine "
                "multimodality, not sampling noise.",
                xy=(0, 1.02), xycoords="axes fraction", fontsize=9, color=MUTED, ha="left",
                va="bottom", linespacing=1.3)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
