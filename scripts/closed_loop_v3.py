"""Planner v3 development: failure diagnosis and the clearance / smoothness grid.

DEVELOPMENT scenes only (the frozen split's val + test, 32 scenes), frozen-split
models at seed 0, as for planner v2 (scripts/closed_loop_dev.py). The 73 primary
scenes of the cross-validated test are not touched.

  diagnose  planner v2 on every package; each failed episode (at-fault collision,
            near-miss, harsh braking) is attributed by counterfactual against the
            oracle (true future) on the same episode:
              prediction         the oracle avoids that failure
              planner, no option the oracle fails too and the package found no clear
                                 manoeuvre at some replan (fallback)
              planner, choice    the oracle fails too, no fallback
            Near-misses are also split by whether the ego was moving.
  grid      planner v3 options (closed_loop_sim.V3_DEFAULTS) on the 8 single-
            trajectory packages + the oracle. Criterion, fixed here before any run:
            the highest mean episode score across the 8 packages; ties within 0.005
            go to the configuration listed first (planner v2 is listed first).
            Gate: the chosen setting must raise the dev score of the oracle and of CV
            by >= 0.01 each, without raising the mean at-fault rate.
  latency   single-process planning time per replan for v2 and the chosen v3.

Usage:
    python scripts/closed_loop_v3.py diagnose | grid | latency
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

import closed_loop_sim as cl
from dd_split import load_splits

REPO = Path(__file__).resolve().parent.parent
OUT_DIAG = REPO / "outputs" / "benchmark_v2" / "closed_loop_v3_diagnosis_dev.json"
OUT_GRID = REPO / "outputs" / "benchmark_v2" / "closed_loop_v3_dev_grid.json"
SINGLE = [n for n, (kind, _, _) in cl.PACKAGES.items() if kind == "single"]
SETS = [n for n, (kind, _, _) in cl.PACKAGES.items() if kind == "set"]
ORACLE, HUMAN = "Oracle (true future, no band)", "Human driver (log replay)"
FINE_ACCELS = (-6.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
FAILURES = ("at_fault", "near_miss", "harsh_braking")


def dev_scenes() -> list:
    m = load_splits(REPO / "splits" / "scene_splits.json")
    return sorted(s for s, v in m.items() if v in ("val", "test"))


def v2_cfg() -> tuple[dict, float]:
    g = json.loads((REPO / "outputs" / "benchmark_v2" / "closed_loop_dev_grid.json").read_text(encoding="utf-8"))
    return {k: g["chosen"][k] for k in ("hard_s", "soft_w", "fallback")}, float(g["chosen"]["band"])


def run(names, cfgs, lam, workers) -> dict:
    """{(cfg index, package): episodes} on the dev scenes."""
    dev = dev_scenes()
    tasks = [(i, (n, 0, dev, lam, c, None)) for i, c in enumerate(cfgs) for n in names]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(cl.run_package, [t for _, t in tasks]))
    return {(i, t[0]): r[3] for (i, t), r in zip(tasks, res)}


def key(e) -> tuple:
    return e["scene"], e["t_start"], e["persona"]


def summary(eps) -> dict:
    out = {k: v for k, v in cl.summarise(eps).items() if k != "by_persona"}
    fb = [e["n_fallback"] for e in eps if e["n_fallback"] is not None]
    rp = sum(e["n_replans"] for e in eps)
    out.update({"near_miss_moving_rate": float(np.mean([e["near_miss_moving"] for e in eps])),
                "jerk_rms": float(np.mean([e["jerk_rms"] for e in eps])),
                "fallback_replan_rate": float(sum(fb) / rp) if fb and rp else None,
                "plan_ms_parallel": float(np.mean([e["plan_ms"] for e in eps]))})
    return out


def diagnose(workers: int) -> None:
    cfg, lam = v2_cfg()
    names = SINGLE + SETS + [ORACLE, HUMAN]
    eps = run(names, [cfg], lam, workers)
    oracle = {key(e): e for e in eps[(0, ORACLE)]}
    packages = {}
    for n in names:
        E = eps[(0, n)]
        attr = {}
        for f in FAILURES:
            fails = [e for e in E if e[f]]
            pred = [e for e in fails if not oracle[key(e)][f]]
            plan = [e for e in fails if oracle[key(e)][f]]
            attr[f] = {"n": len(fails),
                       "prediction": len(pred),
                       "planner_no_option": sum(bool(e["n_fallback"]) for e in plan),
                       "planner_choice": sum(not e["n_fallback"] for e in plan)}
        attr["near_miss"]["while_stopped"] = sum(e["near_miss"] and not e["near_miss_moving"] for e in E)
        packages[n] = {**summary(E), "attribution": attr}
    payload = {"dev_scenes": dev_scenes(), "seed": 0, "planner": {**cfg, "band": lam},
               "attribution_rule": "a failure is 'prediction' if the oracle (true future) avoids it on the "
                                   "same episode; otherwise 'planner', split by whether the package found "
                                   "no clear manoeuvre at some replan",
               "packages": packages, "when": datetime.now().isoformat(timespec="seconds")}
    OUT_DIAG.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{'package':32s} score  atF  near(mov) harsh  jerk  fallback | at-fault pred/noopt/choice | "
          f"near-miss pred/noopt/choice/stopped | harsh pred/noopt/choice")
    for n, v in packages.items():
        a = v["attribution"]
        fbr = "  -  " if v["fallback_replan_rate"] is None else f"{v['fallback_replan_rate']:.3f}"
        print(f"{n:32s} {v['score']:.3f} {v['at_fault_rate']:.3f} {v['near_miss_rate']:.2f}({v['near_miss_moving_rate']:.2f}) "
              f"{v['harsh_braking_rate']:.2f} {v['jerk_rms']:5.1f} {fbr} | "
              f"{a['at_fault']['prediction']}/{a['at_fault']['planner_no_option']}/{a['at_fault']['planner_choice']} | "
              f"{a['near_miss']['prediction']}/{a['near_miss']['planner_no_option']}/{a['near_miss']['planner_choice']}"
              f"/{a['near_miss']['while_stopped']} | "
              f"{a['harsh_braking']['prediction']}/{a['harsh_braking']['planner_no_option']}/{a['harsh_braking']['planner_choice']}")
    print("saved:", OUT_DIAG)


def grid() -> list[dict]:
    """Planner v2 first, then every combination (lat_w stays 0)."""
    out = [{"accels": "v2", "clear_w": 0.0, "jerk_w": 0.0}]
    for acc, cw, jw in product(("v2", "fine"), (0.0, 0.5, 2.0), (0.0, 0.05, 0.2)):
        c = {"accels": acc, "clear_w": cw, "jerk_w": jw}
        if c not in out:
            out.append(c)
    return out


def to_cfg(g: dict, base: dict) -> dict:
    return {**base, "accels": cl.ACCELS if g["accels"] == "v2" else FINE_ACCELS,
            "clear_w": g["clear_w"], "clear_m": 1.0, "jerk_w": g["jerk_w"], "lat_w": 0.0}


def run_grid(workers: int) -> None:
    base, lam = v2_cfg()
    configs = grid()
    eps = run(SINGLE + [ORACLE], [to_cfg(g, base) for g in configs], lam, workers)
    rows = []
    for i, g in enumerate(configs):
        pk = {n: summary(eps[(i, n)]) for n in SINGLE + [ORACLE]}
        rows.append({**g, "mean_score": float(np.mean([pk[n]["score"] for n in SINGLE])),
                     "mean_at_fault": float(np.mean([pk[n]["at_fault_rate"] for n in SINGLE])),
                     "mean_near_miss": float(np.mean([pk[n]["near_miss_rate"] for n in SINGLE])),
                     "mean_harsh": float(np.mean([pk[n]["harsh_braking_rate"] for n in SINGLE])),
                     "mean_progress": float(np.mean([pk[n]["progress"] for n in SINGLE])),
                     "mean_jerk": float(np.mean([pk[n]["jerk_rms"] for n in SINGLE])),
                     "packages": pk})
    best = max(r["mean_score"] for r in rows)
    chosen = next(r for r in rows if r["mean_score"] >= best - 0.005)
    v2 = rows[0]
    gate = {"oracle_gain": chosen["packages"][ORACLE]["score"] - v2["packages"][ORACLE]["score"],
            "cv_gain": chosen["packages"]["CV (vel_k=5)"]["score"] - v2["packages"]["CV (vel_k=5)"]["score"],
            "at_fault_change": chosen["mean_at_fault"] - v2["mean_at_fault"]}
    gate["passed"] = bool(gate["oracle_gain"] >= 0.01 and gate["cv_gain"] >= 0.01 and gate["at_fault_change"] <= 0)
    payload = {"dev_scenes": dev_scenes(), "packages": SINGLE + [ORACLE], "seed": 0,
               "base": {**base, "band": lam}, "fine_accels": FINE_ACCELS, "clear_m": 1.0, "lat_w": 0.0,
               "criterion": "highest mean episode score across the 8 single-trajectory packages; ties within "
                            "0.005 go to the configuration listed first (planner v2 first)",
               "gate": "oracle and CV dev score each +0.01 or more, mean at-fault not higher than v2",
               "grid": rows, "chosen": {k: chosen[k] for k in ("accels", "clear_w", "jerk_w")},
               "gate_result": gate, "when": datetime.now().isoformat(timespec="seconds")}
    OUT_GRID.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for r in sorted(rows, key=lambda r: -r["mean_score"]):
        o = r["packages"][ORACLE]
        print(f"accels {r['accels']:4s} clear {r['clear_w']:.1f} jerk {r['jerk_w']:.2f}: score {r['mean_score']:.3f} "
              f"atF {r['mean_at_fault']:.3f} near {r['mean_near_miss']:.2f} harsh {r['mean_harsh']:.2f} "
              f"prog {r['mean_progress']:.3f} jerk {r['mean_jerk']:4.1f} | oracle {o['score']:.3f}")
    print("chosen:", payload["chosen"], "gate:", gate)


def latency() -> None:
    base, lam = v2_cfg()
    g = json.loads(OUT_GRID.read_text(encoding="utf-8"))["chosen"]
    scenes = dev_scenes()[:4]
    for label, cfg in (("v2", to_cfg({"accels": "v2", "clear_w": 0.0, "jerk_w": 0.0}, base)),
                       ("v3 chosen", to_cfg(g, base))):
        for n in ("CV (vel_k=5)", "K-mode set (10 modes)"):
            eps = cl.run_package((n, 0, scenes, lam, cfg, None))[3]
            ms = [e["plan_ms"] for e in eps]
            print(f"{label:10s} {n:24s} plan ms per replan: mean {np.mean(ms):.1f}  max episode mean {max(ms):.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("diagnose", "grid", "latency"))
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    if args.cmd == "diagnose":
        diagnose(args.workers)
    elif args.cmd == "grid":
        run_grid(args.workers)
    else:
        latency()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
