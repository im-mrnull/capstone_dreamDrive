"""README figures for the benchmark, the closed-loop test and the planner headroom test, all
regenerated from committed artifacts (like the tables):

  benchmark_ade_ci.png    deployable test ADE of each seed-0 model with its 95% scene-bootstrap
                          CI (outputs/benchmark_v2/bootstrap_ci.json).
  closed_loop_cv.png      closed-loop test v2, primary scenes: episode score with its 95% CI and
                          at-fault collision episodes per package (closed_loop_cv.json).
  planner_headroom.png    headroom test, public planners. Top: the matched-planner G2 cells
                          (headroom_matched.json); the planner moves the score more than the
                          predictor does. Bottom: the primary-scene confirmation
                          (headroom_confirm.json), ideal set minus each single-future cell with
                          the paired scene-bootstrap CI of counterfactual_sim.paired_boot (2,000
                          replicates, seed 0), the numbers in docs/wiki/headroom-test.md.

Usage:
    python scripts/plot_readme_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import counterfactual_sim as cf

REPO = Path(__file__).resolve().parent.parent
B2 = REPO / "outputs" / "benchmark_v2"

# Same light-surface palette as the other README figures (dataviz skill; CVD-checked:
# orange / blue / aqua pass, aqua is below 3:1 on the surface, so its marks carry labels).
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SINGLE, SET, PROPOSAL = "#eb6834", "#2a78d6", "#1baf7a"
BOOT_NAMES = {"ST-GAT-inspired (MHA)": "ST-GAT (MHA)", "map-free transformer": "Map-free transformer",
              "LED (most-likely)": "LED (mean-denoised)", "AgentFormer (most-likely)": "AgentFormer (2 Hz)",
              "Trajectron++ (most-likely, canonical windows)": "Trajectron++ (most-likely)", "MLP": "MLP",
              "K-mode decoder (top-1-sel)": "K-mode (top-1)", "CV (vel_k=5)": "CV (vel_k=5)"}


def signed(x: float, fmt: str = ".3f") -> str:
    t = f"{x:+{fmt}}"
    return "±" + t[1:] if float(t) == 0 else t.replace("-", "−")


def style(ax, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="x", colors=INK2, labelsize=9)
    ax.tick_params(axis="y", length=0, labelsize=9.5, labelcolor=INK)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)


def figure(w: float, h: float, **kw):
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(figsize=(w, h), **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def title(fig, text: str, sub: str) -> None:
    fig.text(0.012, 0.975, text, fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.925, sub, fontsize=9.5, color=INK2, va="top")


def ci_rows(ax, ys, mid, lo, hi, colors, labels=None, ms=8):
    for y, m, a, b, c in zip(ys, mid, lo, hi, colors):
        ax.plot([a, b], [y, y], color=c, lw=2, solid_capstyle="round", zorder=3)
        ax.plot(m, y, "o", ms=ms, mfc=c, mec=SURFACE, mew=2, zorder=4)
    if labels:
        x1 = ax.get_xlim()[1]
        for y, t in zip(ys, labels):
            ax.text(x1, y, t, fontsize=8.5, color=INK2, va="center", ha="left")


def benchmark_ade_ci(out: Path) -> None:
    b = json.loads((B2 / "bootstrap_ci.json").read_text(encoding="utf-8"))
    rows = sorted(b["models"].items(), key=lambda kv: kv[1]["ade"])
    fig, ax = figure(8.6, 4.4)
    ys = list(range(len(rows)))[::-1]
    ax.set_xlim(1.8, 7.2)
    ci_rows(ax, ys, [m["ade"] for _, m in rows], [m["ade_ci95"][0] for _, m in rows],
            [m["ade_ci95"][1] for _, m in rows], [SET] * len(rows),
            [f"  {m['ade']:.2f} [{m['ade_ci95'][0]:.2f}, {m['ade_ci95'][1]:.2f}]" for _, m in rows])
    ax.set_yticks(ys, [BOOT_NAMES.get(k, k) for k, _ in rows])
    style(ax, "Test ADE (m), seed-0 model; line = 95% CI over test-scene resamples")
    title(fig, "Every model's interval overlaps the others: rankings need paired tests",
          f"{b['n_scenes']} test scenes, {b['n_boot']:,} scene resamples. Paired differences (README table) "
          "separate some pairs, e.g. ST-GAT vs CV.")
    fig.subplots_adjust(left=0.22, right=0.80, top=0.82, bottom=0.13)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def closed_loop_cv(out: Path) -> None:
    cl = json.loads((B2 / "closed_loop_cv.json").read_text(encoding="utf-8"))
    a = next(v for k, v in cl["analyses"].items() if k.startswith("primary"))
    key = f"band_x{cl['planner']['band']:g}"
    pk = a["packages"]
    refs = [n for n in pk if key not in pk[n]]
    point = sorted([n for n in pk if key in pk[n] and "set" not in n], key=lambda n: -pk[n][key]["score"])
    sets = [n for n in pk if key in pk[n] and "set" in n]
    order = point + sets + refs
    val = {n: pk[n].get(key, pk[n].get("band_x1")) for n in order}
    color = {n: SINGLE if n in point else SET if n in sets else MUTED for n in order}
    fig, (ax1, ax2) = figure(10.4, 5.0, ncols=2, sharey=True, gridspec_kw={"width_ratios": [1.6, 1]})
    ys = list(range(len(order)))[::-1]
    ax1.set_xlim(0.55, 1.0)
    lo = [val[n].get("score_ci95", [val[n]["score"]] * 2)[0] for n in order]
    hi = [val[n].get("score_ci95", [val[n]["score"]] * 2)[1] for n in order]
    ci_rows(ax1, ys, [val[n]["score"] for n in order], lo, hi, [color[n] for n in order])
    for y, n in zip(ys, order):
        ax1.text(hi[ys.index(y)] + 0.008, y, f"{val[n]['score']:.3f}", fontsize=8.5, color=INK2, va="center")
    ax1.set_yticks(ys, order)
    style(ax1, "Episode score (higher is better), 95% scene-bootstrap CI")
    ax2.set_xlim(0, 20)
    for y, n in zip(ys, order):
        r = 100 * val[n]["at_fault_rate"]
        ax2.plot([0, r], [y, y], color=color[n], lw=2, solid_capstyle="round", zorder=3)
        ax2.plot(r, y, "o", ms=8, mfc=color[n], mec=SURFACE, mew=2, zorder=4)
        ax2.text(r + 0.6, y, f"{r:.1f}%", fontsize=8.5, color=INK2, va="center")
    style(ax2, "At-fault collision episodes (%)")
    for ax in (ax1, ax2):
        ax.axhline(len(order) - len(point) - 0.5, color=GRID, lw=1)
        ax.axhline(len(refs) - 0.5, color=GRID, lw=1)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=c, mec=c, ms=7, label=t)
               for c, t in ((SINGLE, "single trajectory"), (SET, "10-future set (planned as obstacles)"),
                            (MUTED, "reference"))]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.006, 0.895), ncol=3, frameon=False, fontsize=9,
               labelcolor=INK2, handletextpad=0.2, columnspacing=1.6)
    title(fig, "Closed loop: the point predictors tie; planning against every sampled future hurts",
          f"Closed-loop test v2, {a['n_scenes']} primary scenes never evaluated before, planner v2 without the band.")
    fig.subplots_adjust(left=0.23, right=0.98, top=0.78, bottom=0.12, wspace=0.08)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def planner_headroom(out: Path) -> None:
    m = json.loads((B2 / "headroom_matched.json").read_text(encoding="utf-8"))["cells"]
    c = json.loads((B2 / "headroom_confirm.json").read_text(encoding="utf-8"))["cells"]
    preds = ["Branch oracle", "Ideal single (mean)", "Ideal single (most likely)", "Ideal set"]
    fig = plt.figure(figsize=(10.4, 7.4))
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.0], hspace=0.78, wspace=0.08)
    ax = fig.add_subplot(gs[0, :])
    ys = list(range(len(preds)))[::-1]
    for y, p in zip(ys, preds):
        v2, pr = m["v2"][p]["score"], m["proposal"][p]["score"]
        ax.plot([v2, pr], [y, y], color=GRID, lw=3, zorder=2)
        ax.plot(v2, y, "o", ms=9, mfc=MUTED, mec=SURFACE, mew=2, zorder=4)
        ax.plot(pr, y, "o", ms=9, mfc=PROPOSAL, mec=SURFACE, mew=2, zorder=4)
        ax.text(v2 - 0.002, y, f"{v2:.3f}", fontsize=8.5, color=INK2, va="center", ha="right")
        ax.text(pr + 0.002, y, f"{pr:.3f}", fontsize=8.5, color=INK2, va="center", ha="left")
    ax.set_xlim(0.80, 0.88)
    ax.set_ylim(-0.6, len(preds) - 0.4)
    ax.set_yticks(ys, preds)
    style(ax, "Episode score (higher is better)")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", mfc=c, mec=c, ms=7, label=t)
                       for c, t in ((MUTED, "planner v2"), (PROPOSAL, "proposal planner"))],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9, labelcolor=INK2)

    scenes = sorted(c["proposal | Ideal set"]["per_scene"])
    counts = cf.boot_counts(len(scenes), 0)
    comps = [("proposal | Ideal single (mean)", "Ideal single (mean)"),
             ("proposal | Ideal single (most likely)", "Ideal single (most likely)"),
             ("proposal | Branch oracle", "Branch oracle"),
             ("proposal+margin | Ideal single (mean)", "single (mean) + 0.5 m margin")]
    ys = list(range(len(comps)))[::-1]
    for j, (metric, lab, scale, fmt, lim) in enumerate((("score", "Score gain of the ideal set", 1, ".3f", (-0.03, 0.03)),
                                                        ("at_fault", "At-fault gain of the ideal set (points)", 100, ".1f",
                                                         (-1.2, 2.2)))):
        ax = fig.add_subplot(gs[1, j])
        r = [cf.paired_boot(c["proposal | Ideal set"], c[x], metric, scenes, counts) for x, _ in comps]
        cols = [SINGLE if "margin" in x else INK2 for x, _ in comps]
        ax.set_xlim(*lim)
        ax.axvline(0, color="#c3c2b7", lw=1.2, zorder=1)
        ci_rows(ax, ys, [scale * t["gain"] for t in r], [scale * t["ci95"][0] for t in r],
                [scale * t["ci95"][1] for t in r], cols, ms=7)
        for y, t in zip(ys, r):
            ax.text(scale * t["ci95"][1] + 0.02 * (lim[1] - lim[0]), y + 0.28, signed(scale * t["gain"], fmt),
                    fontsize=8.5, color=INK2, va="center")
        ax.set_yticks(ys, [f"vs {n}" for _, n in comps] if j == 0 else [])
        style(ax, lab + ";  > 0 = set better")
    fig.text(0.012, 0.865, "A  Matched planners, 32 dev scenes, fresh draws: the planner moves the score "
             "+0.04–0.05, the predictor at most +0.008", fontsize=10, color=INK, va="top", fontweight="bold")
    fig.text(0.012, 0.448, "B  73 primary scenes, proposal planner: ideal set minus each single-future cell",
             fontsize=10, color=INK, va="top", fontweight="bold")
    fig.text(0.035, 0.422, "Against plain singles its edge is within noise; one trajectory with a 0.5 m margin "
             "beats it on score and ties it on at-fault collisions", fontsize=9.5, color=INK2, va="top")
    title(fig, "Planner headroom: the planner matters more than multiple futures",
          "Hidden timing branches (strong dose), ideal predictors, public planners only. "
          "Lines are 95% paired scene-bootstrap CIs.")
    fig.subplots_adjust(left=0.22, right=0.97, top=0.78, bottom=0.08)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    for name, fn in (("benchmark_ade_ci.png", benchmark_ade_ci), ("closed_loop_cv.png", closed_loop_cv),
                     ("planner_headroom.png", planner_headroom)):
        fn(B2 / name)
        print("wrote", B2 / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
