"""Planner development grid for the closed-loop test (band / planner interaction).

The pre-registered v1 planner treated each package's 90% conformal band as a hard
constraint over the whole 3 s horizon and, when nothing was clear, took the
least-penetrating manoeuvre; it braked hard in 37-50% of episodes and every model
did better without its band. This grid searches how the band is used:

  hard_s     the band is a hard constraint only up to this horizon (0.5, 1, 3 s)
  soft_w     beyond it, a cost on band penetration (0 or 5)
  fallback   when nothing is clear: least penetration, or a straight emergency brake
  band       band scale lambda (0, 0.5, 1)

on DEVELOPMENT scenes only: the frozen split's val (10) and test (22) scenes,
which the confirmatory cross-validation test excludes from its primary analysis.
Every configuration runs the 8 single-trajectory packages (frozen-split models,
seed 0). Selection criterion, fixed here before any run: the highest mean
episode score across the 8 packages (no package is favoured); ties within
0.005 go to the configuration listed first.

Usage:
    python scripts/closed_loop_dev.py
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

import closed_loop_sim as cl
from dd_split import load_splits

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "benchmark_v2" / "closed_loop_dev_grid.json"
SINGLE = [n for n, (kind, _, _) in cl.PACKAGES.items() if kind == "single"]


def grid() -> list[dict]:
    out = []
    for band, fallback, hard_s in product((1.0, 0.5, 0.0), ("brake", "least_penetration"), (1.0, 0.5, 3.0)):
        for soft_w in ((0.0,) if hard_s >= 3.0 else (5.0, 0.0)):
            out.append({"band": band, "hard_s": hard_s, "soft_w": soft_w, "fallback": fallback})
    return out


def main() -> int:
    m = load_splits(REPO / "splits" / "scene_splits.json")
    dev = sorted(s for s, v in m.items() if v in ("val", "test"))
    configs = grid()
    tasks = [(name, 0, dev, c["band"], {k: c[k] for k in ("hard_s", "soft_w", "fallback")}, None)
             for c in configs for name in SINGLE]
    refs = [(name, 0, dev, 1.0, {k: c[k] for k in ("hard_s", "soft_w", "fallback")}, None)
            for c in configs if c["band"] == 1.0 for name in ("Oracle (true future, no band)",
                                                               "No prediction (ignore agents)")]
    refs.append(("Human driver (log replay)", 0, dev, 1.0, dict(cl.PLAN_CFG), None))
    with ProcessPoolExecutor(max_workers=24) as ex:
        res = list(ex.map(cl.run_package, tasks + refs))
    by_cfg = {}
    for (name, _, scenes, lam, cfg, _), (_, _, _, eps, excl) in zip(tasks + refs, res):
        key = json.dumps({"band": lam, **cfg}, sort_keys=True)
        by_cfg.setdefault(key, {})[name] = {**cl.summarise(eps), "excluded": excl}
    rows = []
    for c in configs:
        key = json.dumps({"band": c["band"], "hard_s": c["hard_s"], "soft_w": c["soft_w"],
                          "fallback": c["fallback"]}, sort_keys=True)
        pk = by_cfg[key]
        rows.append({**c, "mean_score": float(np.mean([pk[n]["score"] for n in SINGLE])),
                     "mean_at_fault": float(np.mean([pk[n]["at_fault_rate"] for n in SINGLE])),
                     "mean_harsh": float(np.mean([pk[n]["harsh_braking_rate"] for n in SINGLE])),
                     "packages": {n: {k: pk[n][k] for k in ("score", "at_fault_rate", "progress",
                                                             "harsh_braking_rate")} for n in SINGLE}})
    best = max(rows, key=lambda r: r["mean_score"])
    chosen = next(r for r in rows if r["mean_score"] >= best["mean_score"] - 0.005)
    refs_out = {json.dumps({"band": 1.0, **cfg}, sort_keys=True) + " | " + name: {**cl.summarise(eps), "excluded": excl}
                for (name, _, _, _, cfg, _), (_, _, _, eps, excl) in zip(refs, res[len(tasks):])}
    payload = {"dev_scenes": dev, "packages": SINGLE, "seed": 0,
               "criterion": "highest mean episode score across the 8 single-trajectory packages; "
                            "ties within 0.005 go to the configuration listed first",
               "grid": rows, "chosen": {k: chosen[k] for k in ("band", "hard_s", "soft_w", "fallback")},
               "references": refs_out, "instrument": {"ego_rect": cl.EGO_RECT, "ego_centre": cl.EGO_CENTRE,
                                                      "collision_tol": cl.COLLISION_TOL, "glitch_m": cl.GLITCH_M},
               "when": datetime.now().isoformat(timespec="seconds")}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for r in sorted(rows, key=lambda r: -r["mean_score"])[:12]:
        print(f"band {r['band']:.1f} hard {r['hard_s']:.1f}s soft {r['soft_w']:.0f} {r['fallback']:17s}: "
              f"score {r['mean_score']:.3f}  at-fault {r['mean_at_fault']:.3f}  harsh {r['mean_harsh']:.2f}")
    print("chosen:", payload["chosen"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
