"""EXPLORATORY follow-up to the pre-registered closed-loop test (not used for the pick).

Questions asked after seeing the pre-registered result:
  1. Does prediction help at all? Each package vs the planner that ignores agents.
  2. Does any learned predictor beat constant velocity?
  3. What does the 90% conformal band do? Each package with vs without its band.
All paired over the same episodes, seed-pooled scene bootstrap (same machinery as
the decision rule). Reads outputs/closed_loop/episodes_test.json (written by
closed_loop_sim.py --episodes-out) and adds an `exploratory` block to
outputs/benchmark_v2/closed_loop.json.

Usage:
    python scripts/closed_loop_explore.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from closed_loop_sim import CL_DIR, OUT_JSON, PACKAGES, paired, replicates
from dd_bootstrap import bootstrap_counts

EPISODES = CL_DIR / "episodes_test.json"
IGNORE = "No prediction (ignore agents)"
CV = "CV (vel_k=5)"


def main() -> int:
    runs = json.loads(EPISODES.read_text(encoding="utf-8"))
    by = {}
    for r in runs:
        by.setdefault((r["package"], r["band_scale"]), {})[r["seed"]] = r["episodes"]
    scenes = np.asarray(sorted({e["scene"] for r in runs for e in r["episodes"]}), dtype=object)
    counts = bootstrap_counts(len(scenes), 2000, 0)
    rng = np.random.default_rng(2)

    def reps(key, metric):
        eps = by[key]
        return replicates([eps[s] for s in sorted(eps)], metric, scenes, counts, rng)

    def mean(key, metric):
        return float(np.mean([np.mean([float(e[metric]) for e in eps]) for eps in by[key].values()]))

    out = {"note": "Exploratory, added after the pre-registered result; not used for the pick. "
                   "Paired over identical episodes; seed-pooled scene bootstrap (2,000).",
           "comparisons": []}
    for metric in ("score", "at_fault"):
        R, P = {}, {}
        keys = [(n, lam) for n in PACKAGES for lam in (1.0, 0.0)] + [(IGNORE, 1.0)]
        for k in keys:
            R[k], P[k] = reps(k, metric), mean(k, metric)
        for n in PACKAGES:
            for lam in (1.0, 0.0):
                for ref, label in (((IGNORE, 1.0), "vs ignoring agents"), ((CV, lam), "vs CV, same band setting")):
                    if (n, lam) == ref:
                        continue
                    c = paired(R, (n, lam), ref, P)
                    out["comparisons"].append({"metric": metric, "package": n, "band": lam == 1.0,
                                               "against": label, "diff": c["diff"], "ci95": c["ci95"],
                                               "separated": not c["tied"]})
            c = paired(R, (n, 0.0), (n, 1.0), P)
            out["comparisons"].append({"metric": metric, "package": n, "band": None,
                                       "against": "without band vs with band", "diff": c["diff"],
                                       "ci95": c["ci95"], "separated": not c["tied"]})
    payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    payload["exploratory"] = out
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for c in out["comparisons"]:
        if c["separated"]:
            print(f"{c['metric']:8s} {c['package']:30s} band={c['band']!s:5s} {c['against']:26s} "
                  f"{c['diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
    print(f"separated: {sum(c['separated'] for c in out['comparisons'])} of {len(out['comparisons'])} comparisons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
