"""Confirmatory closed-loop test on scene-level cross-validation folds.

Pre-registered in docs/wiki/closed-loop-cv.md. Every one of the 105 IDD-3D scenes
is driven by models that never trained on it (splits/cv_folds/fold<k>.json), with
the planner configuration chosen on development scenes (closed_loop_dev.py) and
the instrument fixes (exact rectangles, measured ego footprint, pose-glitch
exclusion). The PRIMARY analysis uses only the 73 scenes of the frozen split's
train set: no closed-loop result was ever looked at on them. The 32 development
scenes (frozen val + test) are reported separately.

    python scripts/closed_loop_cv.py prepare --fold 0      # dumps, dense windows, predictions
    python scripts/closed_loop_cv.py verify                # dumps == fold metrics JSONs
    python scripts/closed_loop_cv.py run                   # closed loop + decision rule
    python scripts/closed_loop_cv.py sensitivity           # instrument sensitivity (amendment)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

import closed_loop_sim as cl
from dd_bootstrap import bootstrap_counts, ci
from dd_split import load_splits

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
FOLDS = 5
OUT_JSON = REPO / "outputs" / "benchmark_v2" / "closed_loop_cv.json"
DEV_GRID = REPO / "outputs" / "benchmark_v2" / "closed_loop_dev_grid.json"
RULE_DOC = "docs/wiki/closed-loop-cv.md"
PACKAGES_CV = ["CV (vel_k=5)", "MLP", "ST-GAT (MHA)", "Map-free transformer", "LED (mean-denoised)"]
LEARNED = ["ST-GAT (MHA)", "Map-free transformer", "LED (mean-denoised)"]
IGNORE = "No prediction (ignore agents)"


def fold_dir(k):
    return REPO / "outputs" / "cv_folds" / f"fold{k}"


def fold_split(k):
    return REPO / "splits" / "cv_folds" / f"fold{k}.json"


def run(cmd):
    print(">", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO)


def prepare(k: int) -> None:
    fd, sp = fold_dir(k), fold_split(k)
    dumps = fd / "dumps"
    run([PY, "scripts/dump_predictions.py", "--models", "stgat", "transformer", "--seed", "0",
         "--ckpt-dir", fd, "--splits-file", sp, "--out-dir", dumps])
    run([PY, "scripts/model7_led_train_eval.py", "--eval-only", "--seed", "0", "--device", "cuda",
         "--data", "outputs/canonical_world_obs20_pred60.npz", "--splits-file", sp,
         "--save-ckpt", fd / "led_ckpt_seed0.pt", "--save-metrics-json", dumps / "led_evalonly.json",
         "--save-pred-npz", dumps / "preds_led.npz"])
    dense = cl.CV_PRED_DIR / f"fold{k}" / "dense_windows_test.npz"
    run([PY, "scripts/closed_loop_windows.py", "--split", "test", "--splits-file", sp, "--out", dense])
    run([PY, "scripts/closed_loop_predict.py", "--models", "cv", "mlp", "stgat", "transformer", "led",
         "--seeds", "0", "--dense", dense, "--split", "test", "--splits-file", sp,
         "--ckpt-dir", fd, "--out-dir", cl.CV_PRED_DIR / f"fold{k}" / "preds"])


def verify() -> None:
    """Each fold dump reproduces its fold's metrics JSON (test ADE), MLP in numpy too."""
    import safety_scorecard as sc
    from dd_data import masked_ade_fde_np
    for k in range(FOLDS):
        pop = sc.load_population(fold_split(k))
        t = pop["split"] == "test"
        for key, jf, sub in (("stgat", "stgat_seed0.json", None), ("transformer", "transformer_seed0.json", None),
                             ("led", "led_seed0.json", "most_likely")):
            d = np.load(fold_dir(k) / "dumps" / f"preds_{key}.npz", allow_pickle=True)
            assert np.array_equal(d["split"].astype(str), pop["split"]), (k, key)
            ade = masked_ade_fde_np(d["pred"][t].astype(float), pop["gt"][t], pop["fm"][t])[0].mean()
            j = json.loads((fold_dir(k) / jf).read_text(encoding="utf-8"))["test"]
            ref = (j[sub] if sub else j)["ade"]
            assert abs(ade - ref) < 5e-4, f"fold {k} {key}: dump {ade:.4f} vs json {ref:.4f}"
            print(f"fold {k} {key:12s} test ADE {ade:.4f} == json {ref:.4f}")
        w = np.load(fold_dir(k) / "mlp_best_seed0.npz")
        ade = masked_ade_fde_np(sc.mlp_world(pop["obs"][t], {x: w[x] for x in w.files}), pop["gt"][t], pop["fm"][t])[0].mean()
        ref = json.loads((fold_dir(k) / "mlp_seed0.json").read_text(encoding="utf-8"))["test"]["ade"]
        assert abs(ade - ref) < 1e-3, f"fold {k} mlp: {ade:.4f} vs {ref:.4f}"
        print(f"fold {k} {'mlp':12s} test ADE {ade:.4f} == json {ref:.4f}")


def fold_of_scene() -> dict:
    out = {}
    for k in range(FOLDS):
        for s in json.loads(fold_split(k).read_text(encoding="utf-8"))["splits"]["test"]:
            out[s] = k
    return out


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a family of one-sided tests."""
    order = sorted(pvals, key=pvals.get)
    m, adj, run_max = len(order), {}, 0.0
    for i, h in enumerate(order):
        run_max = max(run_max, (m - i) * pvals[h])
        adj[h] = min(1.0, run_max)
    return adj


def analyse(eps_by, scene_set, counts, rng, latency) -> dict:
    scenes = np.asarray(sorted(scene_set), dtype=object)
    sel = {k: [e for e in v if e["scene"] in scene_set] for k, v in eps_by.items()}
    pk, reps, reps_v = {}, {}, {}
    for (name, lam), eps in sel.items():
        s = cl.summarise(eps)
        r = cl.replicates([eps], "score", scenes, counts, rng)
        pk.setdefault(name, {})[f"band_x{lam:g}"] = {**s, "score_ci95": ci(r)}
        reps[(name, lam)] = r
        reps_v[(name, lam)] = cl.replicates([eps], "at_fault_vru", scenes, counts, rng)
        reps[(name, lam, "af")] = cl.replicates([eps], "at_fault", scenes, counts, rng)
    lam = PLAN["band"]
    point = {n: pk[n][f"band_x{lam:g}"]["score"] for n in PACKAGES_CV}
    point_v = {n: pk[n][f"band_x{lam:g}"]["at_fault_vru_rate"] for n in PACKAGES_CV}
    decision = cl.decide(point, {n: reps[(n, lam)] for n in PACKAGES_CV}, point_v,
                         {n: reps_v[(n, lam)] for n in PACKAGES_CV}, latency)
    # pre-registered hypothesis family (one-sided, Holm-corrected)
    hyp, pvals = {}, {}

    def one_sided(a, b, key=None, lower_is_better=False):
        ra, rb = (reps[a], reps[b]) if key is None else (reps[(*a, key)], reps[(*b, key)])
        d = (rb - ra) if lower_is_better else (ra - rb)
        d = d[np.isfinite(d)]
        return float(np.mean(d <= 0)), ci(d)

    def add(name, p, c):
        hyp[name], pvals[name] = {"diff_ci95": c}, p

    for n in LEARNED:
        add(f"H1 {n} higher score than CV", *one_sided((n, lam), ("CV (vel_k=5)", lam)))
        add(f"H1b {n} fewer at-fault collisions than CV",
            *one_sided((n, lam), ("CV (vel_k=5)", lam), key="af", lower_is_better=True))
    for n in PACKAGES_CV:
        add(f"H2 {n} fewer at-fault collisions than ignoring agents",
            *one_sided((n, lam), (IGNORE, 1.0), key="af", lower_is_better=True))
    for n in PACKAGES_CV:
        other = 1.0 if lam == 0.0 else 0.0           # the planner's choice vs the alternative
        add(f"H3 {n} chosen band setting (x{lam:g}) scores higher than x{other:g}",
            *one_sided((n, lam), (n, other)))
    adj = holm(pvals)
    for h in hyp:
        hyp[h].update({"p_one_sided": pvals[h], "p_holm": adj[h], "supported": adj[h] < 0.05})
    return {"n_scenes": len(scene_set), "packages": pk, "decision": decision, "hypotheses": hyp}


PLAN = None
SENS_JSON = REPO / "outputs" / "benchmark_v2" / "closed_loop_cv_sensitivity.json"
# Pre-registered sensitivity amendment (docs/wiki/closed-loop-cv.md section 5): one setting
# varied at a time from the primary instrument.
SENSITIVITY = {
    "collision tolerance 0 m": {"collision_tol": 0.0},
    "collision tolerance 0.5 m": {"collision_tol": 0.5},
    "ego footprint 4.5 x 1.7 m nominal car (centre +0.5 m)": {"ego_rect": (4.5, 1.7), "ego_centre": 0.5},
    "glitch threshold 0.5 m": {"glitch_m": 0.5},
    "glitch threshold 2.0 m": {"glitch_m": 2.0},
    "agent sizes -20%": {"agent_scale": 0.8},
    "agent sizes +20%": {"agent_scale": 1.2},
}


def simulate(cfg, folds, scenes, workers):
    lams = sorted({PLAN["band"], 0.0, 1.0}, reverse=True)     # chosen setting + band / no band
    tasks = [(n, 0, scenes, lam, cfg, folds) for n in PACKAGES_CV for lam in lams]
    tasks += [(n, 0, scenes, 1.0, cfg, folds) for n in cl.REFERENCES]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(cl.run_package, tasks))
    eps_by, excluded = {}, {}
    for (name, _, _, lam, _, _), (_, _, _, eps, excl) in zip(tasks, res):
        eps_by[(name, lam)] = eps
        excluded[name] = excl
    return eps_by, excluded


def primary_scenes(scenes):
    m = load_splits(REPO / "splits" / "scene_splits.json")
    return {s for s in scenes if m[s] == "train"}


def main() -> int:
    global PLAN
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "verify", "run", "sensitivity"])
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    if args.step == "prepare":
        for k in ([args.fold] if args.fold is not None else range(FOLDS)):
            prepare(k)
        return 0
    if args.step == "verify":
        verify()
        return 0

    PLAN = json.loads(DEV_GRID.read_text(encoding="utf-8"))["chosen"]
    cfg = {k: PLAN[k] for k in ("hard_s", "soft_w", "fallback")}
    folds = fold_of_scene()
    scenes = sorted(folds)
    latency = json.loads(cl.LATENCY.read_text(encoding="utf-8"))["packages"] if cl.LATENCY.exists() else None
    if args.step == "sensitivity":
        prim = primary_scenes(scenes)
        counts = bootstrap_counts(len(prim), 2000, 0)
        out = {}
        for label, inst in SENSITIVITY.items():
            eps_by, excluded = simulate({**cfg, "instrument": inst}, folds, scenes, args.workers)
            a = analyse(eps_by, prim, counts, np.random.default_rng(1), latency)
            out[label] = {"instrument": inst, "excluded_episodes": excluded, "pick": a["decision"]["pick"],
                          "scores": {n: v[f"band_x{PLAN['band']:g}"]["score"] for n, v in a["packages"].items()},
                          "at_fault": {n: v[f"band_x{PLAN['band']:g}"]["at_fault_rate"] for n, v in a["packages"].items()},
                          "hypotheses": {h: {"p_holm": v["p_holm"], "supported": v["supported"]}
                                         for h, v in a["hypotheses"].items()}}
            sup = [h.split(" ")[0] + " " + h.split(" ")[1] for h, v in a["hypotheses"].items() if v["supported"]]
            print(f"{label:52s} pick {a['decision']['pick']}; supported: {', '.join(sup) or 'none'}", flush=True)
        SENS_JSON.write_text(json.dumps({"primary_scenes": len(prim), "variants": out,
                                         "when": datetime.now().isoformat(timespec="seconds")}, indent=2),
                             encoding="utf-8")
        print(f"saved: {SENS_JSON}")
        return 0
    eps_by, excluded = simulate(cfg, folds, scenes, args.workers)
    (cl.CV_PRED_DIR / "episodes_cv.json").write_text(json.dumps(
        [{"package": n, "band_scale": lam, "episodes": e} for (n, lam), e in eps_by.items()]), encoding="utf-8")

    m = load_splits(REPO / "splits" / "scene_splits.json")
    sets = {"primary (73 frozen-train scenes, never evaluated before)": {s for s in scenes if m[s] == "train"},
            "development (32 frozen val + test scenes)": {s for s in scenes if m[s] != "train"},
            "all 105 scenes": set(scenes)}
    out = {}
    for label, sset in sets.items():
        counts = bootstrap_counts(len(sset), 2000, 0)
        out[label] = analyse(eps_by, sset, counts, np.random.default_rng(1), latency)
    rc = subprocess.run(["git", "log", "-1", "--format=%H", "--", RULE_DOC], cwd=REPO,
                        capture_output=True, text=True).stdout.strip() or None
    payload = {"rule_doc": RULE_DOC, "rule_commit": rc, "planner": PLAN,
               "instrument": {"ego_rect": cl.EGO_RECT, "ego_centre": cl.EGO_CENTRE,
                              "collision_tol": cl.COLLISION_TOL, "glitch_m": cl.GLITCH_M},
               "excluded_episodes_per_package": excluded, "analyses": out,
               "when": datetime.now().isoformat(timespec="seconds")}
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    prim = out[next(iter(sets))]
    for n, v in prim["packages"].items():
        b = v.get(f"band_x{PLAN['band']:g}", v.get("band_x1"))
        print(f"{n:32s} score {b['score']:.3f} {[round(x, 3) for x in b['score_ci95']]}  at-fault {b['at_fault_rate']:.3f}")
    for s in prim["decision"]["steps"]:
        print(" ->", s)
    for h, v in prim["hypotheses"].items():
        print(f"{h:52s} p_holm {v['p_holm']:.3f} {'SUPPORTED' if v['supported'] else ''}")
    print(f"saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
