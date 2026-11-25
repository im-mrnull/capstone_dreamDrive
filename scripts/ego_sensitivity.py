"""Sensitivity of the reported IDD-3D results to the ego-motion defect (development of a data-quality check).

Plan, scene sets and reading rule: docs/wiki/ego-quality-sensitivity.md, fixed 2026-11-26 before any clean-subset
number was computed. Each part
  1. re-computes the reported numbers on the original scene set from the saved predictions, episodes or per-scene
     sums, and stops on a mismatch;
  2. re-scores the same quantities on the scenes whose ego-motion estimate passes the ego-only gate of
     scripts/ego_quality.py ("clean": the same rule on every split) and on the rest ("flagged").
Nothing is retrained. Part s4x re-simulates (CPU only) because the amendment-3 closed-loop episodes were never saved;
it must reproduce closed_loop_cv_extra.json before its episodes are used.

    python scripts/ego_sensitivity.py s1 s2 s3 s4 s5 s6 s7      # any of them; outputs/data_quality/sensitivity_<part>.json
    python scripts/ego_sensitivity.py s4x --workers 20          # amendment-3 episodes, then rerun s4
    python scripts/ego_sensitivity.py report                    # the reading rule applied to every listed conclusion
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from dd_bootstrap import bootstrap_counts, ci, per_scene_sums, ratio_replicates, scene_codes
from dd_split import load_splits

REPO = Path(__file__).resolve().parent.parent
DQ = REPO / "outputs" / "data_quality"
B2 = REPO / "outputs" / "benchmark_v2"
SPLITS = REPO / "splits" / "scene_splits.json"
GATE = {"dev": DQ / "ego_quality_dev.json", "test": DQ / "ego_quality_test_egoonly.json"}
ROAD_USER_CHECKS = ("pedestrian_speed", "common_mode")       # the dev-only part of the gate: not used here
SPLIT_OF = load_splits(SPLITS)
CV = "CV (vel_k=5)"
EXTRA_EPISODES = DQ / "episodes_cv_extra.json"


def clean_scenes() -> set:
    out = set()
    for f in GATE.values():
        for r in json.loads(f.read_text(encoding="utf-8"))["rows"]:
            if not [x for x in r["fails"] if x not in ROAD_USER_CHECKS]:
                out.add(r["scene"])
    return out


CLEAN = clean_scenes()
assert (sum(SPLIT_OF[s] != "test" for s in CLEAN), sum(SPLIT_OF[s] == "test" for s in CLEAN)) == (26, 6), \
    "the gate files differ from the ones fixed in the plan"


def subsets(scenes) -> dict:
    s = set(map(str, scenes))
    return {"all": s, "clean": s & CLEAN, "flagged": s - CLEAN}


def same(a, b, what: str, tol: float = 1e-9) -> None:
    """Reproduction check: stop if a re-computed number differs from the reported one."""
    if isinstance(b, (list, tuple)):
        assert len(a) == len(b), f"not reproduced: {what}"
        for x, y in zip(a, b):
            same(x, y, what, tol)
        return
    assert abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b))), f"not reproduced: {what}: {a} vs {b}"


def save(part: str, payload: dict) -> None:
    out = DQ / f"sensitivity_{part}.json"
    out.write_text(json.dumps({**payload, "plan": "docs/wiki/ego-quality-sensitivity.md",
                               "clean_scenes": sorted(CLEAN), "when": datetime.now().isoformat(timespec="seconds")},
                              indent=2), encoding="utf-8")
    print("saved:", out.relative_to(REPO), flush=True)


def load(part: str) -> dict:
    return json.loads((DQ / f"sensitivity_{part}.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------- S0 scene descriptors
VEHICLES = {"Car", "Bus", "Truck", "Van", "TourCar", "PoliceCar", "ConcreteTruck", "Crane"}


def s0() -> None:
    """How the clean and flagged scenes differ (they are not random samples). Counts and classes are free of the
    defect; world-frame speeds and the ego path carry the ego estimate's error and are labelled so."""
    from multifuture_scorecard import PED, TWO_WHEEL
    d = np.load(REPO / "outputs" / "agent_windows_canonical_obs20_pred60.npz", allow_pickle=True)
    scene, typ = d["scene_id"].astype(str), d["agent_type"].astype(str)
    obs = d["obs"].astype(float)
    sp = np.linalg.norm(obs[:, -1] - obs[:, -11], axis=-1)              # m/s over the last observed second
    grp = np.array(["ped" if t in PED else "two_wheeler" if t in TWO_WHEEL else "vehicle" if t in VEHICLES
                    else "other" for t in typ])
    gate = {r["scene"]: r for f in GATE.values() for r in json.loads(f.read_text(encoding="utf-8"))["rows"]}
    out = {}
    for split, test in (("dev", False), ("test", True)):
        scenes = sorted(s for s in set(scene) if (SPLIT_OF[s] == "test") == test)
        for sub, keep in subsets(scenes).items():
            m = np.isin(scene, list(keep))
            per = [int((scene == s).sum()) for s in keep]
            fails = [f for s in keep for f in gate[s]["fails"] if f not in ROAD_USER_CHECKS]
            out.setdefault(split, {})[sub] = {
                "n_scenes": len(keep), "road_users": int(m.sum()),
                "road_users_per_scene_median": float(np.median(per)),
                "road_users_per_scene_range": [int(min(per)), int(max(per))],
                "class_share": {g: float((grp[m] == g).mean()) for g in ("vehicle", "two_wheeler", "ped", "other")},
                "affected_by_defect": {
                    "moving_share_world_frame": float((sp[m] >= 0.5).mean()),
                    "median_speed_of_moving_world_frame": float(np.median(sp[m][sp[m] >= 0.5])),
                    "ego_path_m_estimated_median": float(np.median([gate[s]["path_m"] for s in keep]))},
                "gate_failures": {f: fails.count(f) for f in sorted(set(fails))}}
            o = out[split][sub]
            print(f"  {split:4s} {sub:8s} {o['n_scenes']:2d} scenes | road users/scene {o['road_users_per_scene_median']:.0f} "
                  f"{o['road_users_per_scene_range']} | veh {o['class_share']['vehicle']:.2f} 2w {o['class_share']['two_wheeler']:.2f} "
                  f"ped {o['class_share']['ped']:.2f} | moving {o['affected_by_defect']['moving_share_world_frame']:.2f} "
                  f"at {o['affected_by_defect']['median_speed_of_moving_world_frame']:.1f} m/s | ego path "
                  f"{o['affected_by_defect']['ego_path_m_estimated_median']:.0f} m | {o['gate_failures']}", flush=True)
    save("s0_descriptors", {"source": "agent_windows_canonical_obs20_pred60.npz and the gate files", "splits": out})


# --------------------------------------------------------------------------------------------------- S1 open loop
def seed_pooled_sub(preds_dir: Path, scenes: np.ndarray, counts: np.ndarray, rng_seed: int, seeds=(0, 1, 2)):
    """bootstrap_ci.seed_pooled restricted to the rows of `scenes` (identical when `scenes` is every test scene)."""
    from bootstrap_ci import SEED_DUMPS
    from conformal_wrapper import canonical_dumps
    from dd_data import masked_ade_fde_np
    S, B, k = len(scenes), counts.shape[0], len(seeds)
    rng = np.random.default_rng(rng_seed + 1)
    keep = set(map(str, scenes))
    per_model = {}
    for npz0 in canonical_dumps(preds_dir):
        files = [npz0] + [preds_dir / SEED_DUMPS / f"seed{s}" / npz0.name for s in seeds[1:]]
        if not all(f.exists() for f in files):
            continue
        num, den, name = [], None, None
        for f in files:
            p = np.load(f, allow_pickle=True)
            t = (p["split"].astype(object) == "test") & np.isin(p["scene_id"].astype(str), list(keep))
            ade = masked_ade_fde_np(p["pred"][t], p["gt"][t], p["fut_mask"][t])[0]
            codes, _ = scene_codes(p["scene_id"][t].astype(object), scenes)
            n = per_scene_sums(codes, S, np.ones_like(ade))
            assert den is None or np.array_equal(n, den), f"{f}: rows differ across seeds"
            den, name = n, str(p["model"])
            num.append(per_scene_sums(codes, S, ade))
        num = np.stack(num)
        w = rng.multinomial(k, np.full(k, 1.0 / k), size=B) / k
        per_model[name] = {"per_seed_ade": [float(x.sum() / den.sum()) for x in num],
                           "reps_scene_seed": (counts * (w @ num)).sum(1) / (counts @ den),
                           "reps_scene": counts @ num.mean(0) / (counts @ den)}
    for v in per_model.values():
        v["ade"] = float(np.mean(v["per_seed_ade"]))
    ranked = sorted(per_model, key=lambda m: per_model[m]["ade"])
    pairs = []
    for i, a in enumerate(ranked):
        for b in ranked[i + 1:]:
            ma, mb = per_model[a], per_model[b]
            d = ma["reps_scene_seed"] - mb["reps_scene_seed"]
            pairs.append({"a": a, "b": b, "ade_diff_a_minus_b": ma["ade"] - mb["ade"], "ci95": ci(d),
                          "p_a_not_better": float((d[np.isfinite(d)] >= 0).mean())})
    return {"ranking_by_ade": ranked,
            "models": {m: {"ade_seed_mean": per_model[m]["ade"], "per_seed_ade": per_model[m]["per_seed_ade"],
                           "ade_ci95_scene_seed": ci(per_model[m]["reps_scene_seed"])} for m in ranked},
            "pairs": pairs}


def s1() -> None:
    import bootstrap_ci as bc
    from conformal_wrapper import WINDOWS, mlp_predict
    from dd_data import masked_ade_fde_np
    ref = json.loads((B2 / "bootstrap_ci.json").read_text(encoding="utf-8"))
    rows = bc.test_rows(str(WINDOWS), str(SPLITS), B2)
    tests = sorted({str(s) for r in rows.values() for s in r[4]})
    # MLP seeds 1-2 (the README table's 3-seed mean; CV is deterministic)
    d = np.load(WINDOWS, allow_pickle=True)
    t = np.array([SPLIT_OF.get(str(s)) == "test" for s in d["scene_id"]])
    obs, fut, fm = d["obs"][t].astype(np.float32), d["fut"][t].astype(np.float32), d["fut_mask"][t].astype(bool)
    mlp_scene = d["scene_id"][t].astype(str)
    mlp_ade = [masked_ade_fde_np(mlp_predict(obs, dict(np.load(B2 / f"mlp_best_seed{s}.npz"))), fut, fm)[0]
               for s in (0, 1, 2)]
    out = {}
    for sub, keep in subsets(tests).items():
        scenes = np.asarray(sorted(keep), dtype=object)
        S = len(scenes)
        counts = bootstrap_counts(S, ref["n_boot"], ref["seed"])
        models, reps = {}, {}
        for name, (ade, fde, _, typ, scn) in rows.items():
            m = np.isin(scn.astype(str), scenes.astype(str))
            codes, _ = scene_codes(scn[m], scenes)
            n = per_scene_sums(codes, S, np.ones(m.sum()))
            reps[name] = ratio_replicates(counts, per_scene_sums(codes, S, ade[m]), n)
            r_fde = ratio_replicates(counts, per_scene_sums(codes, S, fde[m]), n)
            models[name] = {"n_rows": int(m.sum()), "n_scenes": int((n > 0).sum()), "ade": float(ade[m].mean()),
                            "ade_ci95": ci(reps[name]), "fde": float(fde[m].mean()), "fde_ci95": ci(r_fde)}
        ranked = sorted(models, key=lambda k: models[k]["ade"])

        def paired(a, b):
            dd = reps[a] - reps[b]
            dd = dd[np.isfinite(dd)]
            return {"a": a, "b": b, "ade_diff_a_minus_b": models[a]["ade"] - models[b]["ade"], "ci95": ci(dd),
                    "p_a_not_better": float((dd >= 0).mean())}
        mk = np.isin(mlp_scene, list(keep))
        out[sub] = {"n_scenes": S, "scenes": [str(s) for s in scenes], "ranking_by_ade": ranked, "models": models,
                    "adjacent_pairs": [paired(ranked[i], ranked[i + 1]) for i in range(len(ranked) - 1)],
                    "vs_best": [paired(ranked[0], m) for m in ranked[1:]],
                    "vs_cv": [paired(m, CV) for m in ranked if m != CV],
                    "mlp_3seed_ade": float(np.mean([a[mk].mean() for a in mlp_ade])),
                    "seed_pooled": seed_pooled_sub(B2, scenes, counts, ref["seed"])}
    a = out["all"]
    assert a["ranking_by_ade"] == ref["ranking_by_ade"], "ranking not reproduced"
    for k, v in ref["models"].items():
        for f in ("ade", "ade_ci95", "fde", "fde_ci95"):
            same(a["models"][k][f], v[f], f"s1 {k} {f}")
    for p, q in zip(a["adjacent_pairs"] + a["vs_best"], ref["adjacent_pairs"] + ref["vs_best"]):
        same(p["ci95"], q["ci95"], f"s1 pair {p['a']} vs {p['b']}")
    # bootstrap_ci.json (2026-10-27) pooled the 3 models that had seed-1/2 dumps then; 3 more were dumped later, so the
    # pooled block now has 6 models and its random seed draws are consumed in a different order: the per-seed ADEs and
    # the pair differences of the original 3 must match exactly, their CIs only approximately.
    rp, ap = ref["seed_pooled"], a["seed_pooled"]
    assert [m for m in ap["ranking_by_ade"] if m in rp["models"]] == rp["ranking_by_ade"]
    for m, v in rp["models"].items():
        same(ap["models"][m]["per_seed_ade"], v["per_seed_ade"], f"s1 pooled {m} per-seed ADE")
    for q in rp["pairs"]:
        p = next(x for x in ap["pairs"] if (x["a"], x["b"]) == (q["a"], q["b"]))
        same(p["ade_diff_a_minus_b"], q["ade_diff_a_minus_b"], f"s1 pooled {p['a']}/{p['b']}")
        same(p["ci95"], q["ci95"], f"s1 pooled CI {p['a']}/{p['b']}", tol=0.05)
    print("S1 reproduced bootstrap_ci.json (seed 0 exactly; seed-pooled per-seed ADEs and differences exactly)", flush=True)
    for sub, r in out.items():
        print(f"  {sub:8s} {r['n_scenes']:2d} scenes | seed-pooled ranking: " + " < ".join(
            f"{m.split(' (')[0]} {r['seed_pooled']['models'][m]['ade_seed_mean']:.2f}" for m in r["seed_pooled"]["ranking_by_ade"])
            + f" | CV {r['models'][CV]['ade']:.2f}, MLP(3 seeds) {r['mlp_3seed_ade']:.2f}")
    save("s1_open_loop", {"source": "bootstrap_ci.json (seed dumps)", "subsets": out})


# ------------------------------------------------------------------------------------------------- S2 conformal
def s2() -> None:
    import conformal_wrapper as cw
    ref = json.loads((B2 / "conformal_metrics.json").read_text(encoding="utf-8"))
    alpha, min_cal, seed, n_boot = ref["alpha"], ref["min_cal"], ref["fold_seed"], ref["n_boot"]
    data = cw.assemble_model_scores(str(cw.WINDOWS), str(SPLITS), B2)
    out = {}
    for name, (s, v, types, scenes, is_val, is_test) in data.items():
        tests = sorted(set(map(str, scenes[is_test])))
        for sub, keep in subsets(tests).items():
            ev = is_test & np.isin(scenes.astype(str), list(keep))
            A = cw.evaluate_split(s[is_val], v[is_val], types[is_val], s[ev], v[ev], types[ev], alpha, min_cal,
                                  cal_scenes=scenes[is_val], rng=np.random.default_rng(seed))
            marg = np.asarray(A["marginal_radius_by_step"])
            simt = np.asarray(A["simultaneous_radius_by_step"])
            A["marginal"].update(cw.bootstrap_report_ci(s[ev], v[ev], scenes[ev], cw.radii_matrix(types[ev], marg, None),
                                                        n_boot, seed))
            A["simultaneous"]["marginal"].update(cw.bootstrap_report_ci(
                s[ev], v[ev], scenes[ev], cw.radii_matrix(types[ev], simt, None), n_boot, seed))
            out.setdefault(name, {})[sub] = {
                "n_scenes": len(keep), "n_agents": int(ev.sum()),
                "step_coverage": A["marginal"]["coverage"], "step_coverage_ci95": A["marginal"]["coverage_ci95"],
                "trajectory_coverage_per_step_band": A["marginal"]["trajectory_coverage"],
                "class_conditional_step_coverage": A["class_conditional"]["coverage"],
                "simultaneous_trajectory_coverage": A["simultaneous"]["marginal"]["trajectory_coverage"],
                "simultaneous_trajectory_coverage_ci95": A["simultaneous"]["marginal"]["trajectory_coverage_ci95"],
                "per_class_step_coverage": A["marginal"]["per_class_coverage"]}
        r = ref["models"][name]["regime_A_val_to_test"]
        same(out[name]["all"]["step_coverage"], r["marginal"]["coverage"], f"s2 {name} coverage")
        same(out[name]["all"]["step_coverage_ci95"], r["marginal"]["coverage_ci95"], f"s2 {name} coverage CI")
        same(out[name]["all"]["simultaneous_trajectory_coverage"], r["simultaneous"]["marginal"]["trajectory_coverage"],
             f"s2 {name} simultaneous")
        print(f"  {name:45s} step coverage (target 0.90): " + "  ".join(
            f"{k} {x['step_coverage']:.3f} [{x['step_coverage_ci95'][0]:.2f}, {x['step_coverage_ci95'][1]:.2f}]"
            for k, x in out[name].items()), flush=True)
    print("S2 reproduced conformal_metrics.json regime A")
    save("s2_conformal", {"source": "conformal_metrics.json, regime A (calibrated on all of val, as reported)",
                          "target": 1 - alpha, "models": out})


# --------------------------------------------------------------------------------------------- S3 closed-loop test
def analyse_test(results: dict, keep: set) -> dict:
    """closed_loop_sim.main's analysis block on the episodes of `keep`."""
    import closed_loop_sim as cl
    scenes = np.asarray(sorted(keep), dtype=object)
    counts = bootstrap_counts(len(scenes), 2000, 0)
    rng = np.random.default_rng(1)
    packages, reps_p, reps_v, reps_af = {}, {}, {}, {}
    for (name, lam), by_seed0 in results.items():
        by_seed = {s: [e for e in eps if e["scene"] in keep] for s, eps in by_seed0.items()}
        seeds = sorted(by_seed)
        per = [cl.summarise(by_seed[s]) for s in seeds]
        entry = {k: float(np.mean([p[k] for p in per])) for k in per[0] if k not in ("by_persona", "n_episodes")}
        entry.update({"n_episodes": per[0]["n_episodes"], "seeds": seeds, "band_scale": lam})
        if lam == 1.0:
            rp = cl.replicates([by_seed[s] for s in seeds], "score", scenes, counts, rng)
            entry["score_ci95"] = ci(rp)
            if name in cl.PACKAGES:
                reps_p[name] = rp
                reps_v[name] = cl.replicates([by_seed[s] for s in seeds], "at_fault_vru", scenes, counts, rng)
            packages[name] = {**packages.get(name, {}), **entry}
        else:
            packages.setdefault(name, {})["no_band"] = {k: entry[k] for k in
                                                        ("score", "at_fault_rate", "at_fault_vru_rate", "progress")}
    # added here (not in the original block): at-fault vs ignoring agents, seed-pooled, separate rng
    rng2 = np.random.default_rng(2)
    ign = "No prediction (ignore agents)"
    for (name, lam), by_seed0 in results.items():
        if lam == 1.0:
            by_seed = {s: [e for e in eps if e["scene"] in keep] for s, eps in by_seed0.items()}
            reps_af[name] = cl.replicates([by_seed[s] for s in sorted(by_seed)], "at_fault", scenes, counts, rng2)
    vs_ignore = {n: {"at_fault_diff": packages[n]["at_fault_rate"] - packages[ign]["at_fault_rate"],
                     "ci95": ci(reps_af[n] - reps_af[ign])} for n in reps_p}
    latency = json.loads(cl.LATENCY.read_text(encoding="utf-8"))["packages"] if cl.LATENCY.exists() else None
    decision = cl.decide({n: packages[n]["score"] for n in reps_p}, reps_p,
                         {n: packages[n]["at_fault_vru_rate"] for n in reps_v}, reps_v, latency)
    return {"n_scenes": len(keep), "packages": packages, "decision": decision, "at_fault_vs_ignore": vs_ignore}


def s3() -> None:
    import closed_loop_sim as cl
    ref = json.loads((B2 / "closed_loop.json").read_text(encoding="utf-8"))
    results = {}
    for r in json.loads((cl.CL_DIR / "episodes_test.json").read_text(encoding="utf-8")):
        results.setdefault((r["package"], r["band_scale"]), {})[r["seed"]] = r["episodes"]
    tests = {e["scene"] for v in results.values() for eps in v.values() for e in eps}
    out = {sub: analyse_test(results, keep) for sub, keep in subsets(tests).items()}
    a = out["all"]
    for n, v in ref["packages"].items():
        for f in ("score", "at_fault_rate", "at_fault_vru_rate"):
            same(a["packages"][n][f], v[f], f"s3 {n} {f}")
        if "score_ci95" in v:
            same(a["packages"][n]["score_ci95"], v["score_ci95"], f"s3 {n} score CI")
    assert a["decision"]["pick"] == ref["decision"]["pick"] and a["decision"]["tied_on_score"] == ref["decision"]["tied_on_score"]
    print(f"S3 reproduced closed_loop.json (pick {ref['decision']['pick']})")
    for sub, r in out.items():
        pk = r["packages"]
        print(f"  {sub:8s} {r['n_scenes']:2d} scenes | pick {r['decision']['pick']} | leader {r['decision']['leader']} | "
              f"tied {len(r['decision']['tied_on_score'])} | ignore-agents at-fault "
              f"{pk['No prediction (ignore agents)']['at_fault_rate']:.3f} vs CV {pk[CV]['at_fault_rate']:.3f}", flush=True)
    save("s3_closed_loop_test", {"source": "closed_loop.json (episodes_test.json)", "subsets": out})


# ----------------------------------------------------------------------------------------------- S4 closed-loop CV
def extra_analyse(eps_by: dict, prim: set) -> dict:
    """closed_loop_cv.run_extra's analysis block on the scenes `prim`."""
    import closed_loop_cv as cvm
    import closed_loop_sim as cl
    scen = np.asarray(sorted(prim), dtype=object)
    counts = bootstrap_counts(len(prim), 2000, 0)
    rng = np.random.default_rng(1)
    lam = cvm.PLAN["band"]
    sel = {key: [e for e in v if e["scene"] in prim] for key, v in eps_by.items()}
    reps = {(n, l, m): cl.replicates([e], m, scen, counts, rng) for (n, l), e in sel.items() for m in ("score", "at_fault")}
    summary = {}
    for (n, l), e in sel.items():
        summary.setdefault(n, {})[f"band_x{l:g}"] = {**cl.summarise(e), "score_ci95": ci(reps[(n, l, "score")])}
    tests, pv = {}, {}

    def t(label, a, b, metric, lower):
        ra, rb = reps[(a, lam, metric)], reps[(b, lam, metric)]
        d = (rb - ra) if lower else (ra - rb)
        d = d[np.isfinite(d)]
        tests[label], pv[label] = {"diff_ci95": ci(d)}, float(np.mean(d <= 0))
    for st, single in cvm.EXTRA_SET_OF.items():
        t(f"S1 {st} higher score than {single}", st, single, "score", False)
        t(f"S2 {st} fewer at-fault collisions than {single}", st, single, "at_fault", True)
    t("T1 MF transformer (top-1) higher score than Map-free transformer",
      "MF transformer (top-1)", "Map-free transformer", "score", False)
    t("T2 MF transformer (top-1) fewer at-fault collisions than Map-free transformer",
      "MF transformer (top-1)", "Map-free transformer", "at_fault", True)
    adj = cvm.holm(pv)
    for h in tests:
        tests[h].update({"p_one_sided": pv[h], "p_holm": adj[h], "supported": adj[h] < 0.05})
    return {"n_scenes": len(prim), "packages": summary, "amendment3_hypotheses": tests}


def s4x(workers: int) -> None:
    """Re-simulate the amendment-3 packages (CPU) exactly as closed_loop_cv.run_extra did, keeping the episodes."""
    from concurrent.futures import ProcessPoolExecutor
    import closed_loop_cv as cvm
    import closed_loop_sim as cl
    cvm.PLAN = json.loads(cvm.DEV_GRID.read_text(encoding="utf-8"))["chosen"]
    cfg = {k: cvm.PLAN[k] for k in ("hard_s", "soft_w", "fallback")}
    folds = cvm.fold_of_scene()
    scenes = sorted(folds)
    lams = sorted({cvm.PLAN["band"], 0.0, 1.0}, reverse=True)
    tasks = [(n, 0, scenes, lam, cfg, folds) for n in cvm.EXTRA_CV for lam in lams]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(cl.run_package, tasks))
    EXTRA_EPISODES.write_text(json.dumps([{"package": n, "band_scale": lam, "episodes": eps}
                                          for (n, _, _, lam, _, _), (_, _, _, eps, _) in zip(tasks, res)]),
                              encoding="utf-8")
    print("saved:", EXTRA_EPISODES.relative_to(REPO), flush=True)


def s4() -> None:
    import closed_loop_cv as cvm
    import closed_loop_sim as cl
    cvm.PLAN = json.loads(cvm.DEV_GRID.read_text(encoding="utf-8"))["chosen"]
    ref = json.loads((B2 / "closed_loop_cv.json").read_text(encoding="utf-8"))
    latency = json.loads(cl.LATENCY.read_text(encoding="utf-8"))["packages"] if cl.LATENCY.exists() else None
    prior = json.loads((cl.CV_PRED_DIR / "episodes_cv.json").read_text(encoding="utf-8"))
    eps_by = {(r["package"], r["band_scale"]): r["episodes"] for r in prior}
    scenes = sorted(cvm.fold_of_scene())                 # as closed_loop_cv.main: every scene, with episodes or not
    assert len(scenes) == 105
    sets = {"primary (73 frozen-train scenes, never evaluated before)": {s for s in scenes if SPLIT_OF[s] == "train"},
            "development (32 frozen val + test scenes)": {s for s in scenes if SPLIT_OF[s] != "train"},
            "all 105 scenes": set(scenes)}

    def run(sset):
        return cvm.analyse(eps_by, sset, bootstrap_counts(len(sset), 2000, 0), np.random.default_rng(1), latency)
    out = {}
    for label, sset in sets.items():
        r = run(sset)
        rr = ref["analyses"][label]
        for h, v in rr["hypotheses"].items():
            same([r["hypotheses"][h]["p_holm"], *r["hypotheses"][h]["diff_ci95"]], [v["p_holm"], *v["diff_ci95"]], f"s4 {label} {h}")
        for h, v in rr["secondary_multi_future"].items():
            same(r["secondary_multi_future"][h]["p_holm"], v["p_holm"], f"s4 {label} {h}")
        for n, v in rr["packages"].items():
            for b, x in v.items():
                same([r["packages"][n][b]["score"], r["packages"][n][b]["at_fault_rate"]], [x["score"], x["at_fault_rate"]],
                     f"s4 {label} {n} {b}")
        assert r["decision"]["pick"] == rr["decision"]["pick"]
        name = label.split(" (")[0]
        out[name] = {}
        for sub, keep in subsets(sset).items():
            out[name][sub] = r if sub == "all" else run(keep)
    print("S4 reproduced closed_loop_cv.json (primary, development, all)", flush=True)
    extra = None
    if EXTRA_EPISODES.exists():
        ref_x = json.loads((B2 / "closed_loop_cv_extra.json").read_text(encoding="utf-8"))
        ex_by = {(r["package"], r["band_scale"]): r["episodes"] for r in
                 json.loads(EXTRA_EPISODES.read_text(encoding="utf-8"))}
        ex_by.update(eps_by)                                         # run_extra's order: new packages, then the primary run
        prim = sets["primary (73 frozen-train scenes, never evaluated before)"]
        r = extra_analyse(ex_by, prim)
        for n, v in ref_x["packages"].items():
            for b, x in v.items():
                same([r["packages"][n][b]["score"], r["packages"][n][b]["at_fault_rate"], *r["packages"][n][b]["score_ci95"]],
                     [x["score"], x["at_fault_rate"], *x["score_ci95"]], f"s4x {n} {b}")
        for h, v in ref_x["amendment3_hypotheses"].items():
            same([r["amendment3_hypotheses"][h]["p_holm"], *r["amendment3_hypotheses"][h]["diff_ci95"]],
                 [v["p_holm"], *v["diff_ci95"]], f"s4x {h}")
        print("S4 amendment 3: re-simulated episodes reproduce closed_loop_cv_extra.json", flush=True)
        extra = {sub: (r if sub == "all" else extra_analyse(ex_by, keep)) for sub, keep in subsets(prim).items()}
    for name, by in out.items():
        for sub, r in by.items():
            sup = [h.split(" ")[0] + ":" + h.split(" ", 1)[1][:28] for h, v in r["hypotheses"].items() if v["supported"]]
            print(f"  {name:12s} {sub:8s} {r['n_scenes']:3d} scenes | pick {r['decision']['pick']} | supported: "
                  f"{len(sup)} of {len(r['hypotheses'])}", flush=True)
    save("s4_closed_loop_cv", {"source": "closed_loop_cv.json (episodes_cv.json); amendment 3 from re-simulated "
                                         "episodes (episodes_cv_extra.json) when present", "sets": out,
                               "amendment3_primary": extra})


# -------------------------------------------------------------------------------------------------- S5 headroom
def s5() -> None:
    from counterfactual_sim import G2M, matched_gate, paired_boot
    hm = json.loads((B2 / "headroom_matched.json").read_text(encoding="utf-8"))
    hc = json.loads((B2 / "headroom_confirm.json").read_text(encoding="utf-8"))
    PRED = ("Branch oracle", "Ideal single (mean)", "Ideal single (most likely)", "Ideal set")

    def restrict(cell: dict, keep: set) -> dict:
        ps = {s: v for s, v in cell["per_scene"].items() if s in keep}
        n = sum(v[0] for v in ps.values())
        return {"score": sum(v[1] for v in ps.values()) / n, "at_fault_rate": sum(v[2] for v in ps.values()) / n,
                "n_episodes": n, "per_scene": ps}

    def boot(keep: set, seed: int = 0):
        sc = sorted(keep)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(sc), (G2M["replicates"], len(sc)))
        return sc, np.stack([np.bincount(r, minlength=len(sc)) for r in idx]).astype(float)

    # matched (dev = val + test scenes), planners v2 and proposal
    out_m = {}
    dev_scenes = sorted(hm["cells"]["v2"]["Ideal set"]["per_scene"])
    for sub, keep in subsets(dev_scenes).items():
        cells = {p: {n: restrict(hm["cells"][p][n], keep) for n in PRED} for p in hm["cells"]}
        sc, counts = boot(keep)
        planner = {n: paired_boot(cells["proposal"][n], cells["v2"][n], "score", sc, counts) for n in PRED}
        planner_af = {n: paired_boot(cells["proposal"][n], cells["v2"][n], "at_fault", sc, counts) for n in PRED}
        set_vs = {}
        for p in cells:
            best = max(("Ideal single (mean)", "Ideal single (most likely)"), key=lambda n: cells[p][n]["score"])
            set_vs[p] = {"vs": best, "score": paired_boot(cells[p]["Ideal set"], cells[p][best], "score", sc, counts),
                         "at_fault": paired_boot(cells[p]["Ideal set"], cells[p][best], "at_fault", sc, counts)}
        out_m[sub] = {"n_scenes": len(keep), "n_episodes_per_cell": cells["v2"]["Ideal set"]["n_episodes"],
                      "cells": {p: {n: {k: v[k] for k in ("score", "at_fault_rate")} for n, v in row.items()}
                                for p, row in cells.items()},
                      "planner_effect_score": planner, "planner_effect_at_fault": planner_af,
                      "set_vs_best_single": set_vs, "g2_matched_public": matched_gate(cells, ["v2", "proposal"], seed=0)}
    for p, row in hm["cells"].items():
        for n, v in row.items():
            same([out_m["all"]["cells"][p][n]["score"], out_m["all"]["cells"][p][n]["at_fault_rate"]],
                 [v["score"], v["at_fault_rate"]], f"s5 {p} {n}", tol=1e-9)
    # confirmation (73 primary scenes), public cells only
    public = [c for c in hc["cells"] if c.split(" | ")[0] in ("proposal", "proposal+margin")]
    out_c = {}
    prim = sorted(hc["cells"][public[0]]["per_scene"])
    assert all(SPLIT_OF[s] == "train" for s in prim) and len(prim) == 62       # 62 of 73 have eligible episodes
    for sub, keep in subsets(prim).items():
        cells = {c: restrict(hc["cells"][c], keep) for c in public}
        sc, counts = boot(keep)
        cmp_ = {c: {"score": paired_boot(cells["proposal | Ideal set"], cells[c], "score", sc, counts),
                    "at_fault": paired_boot(cells["proposal | Ideal set"], cells[c], "at_fault", sc, counts)}
                for c in public if c != "proposal | Ideal set"}
        out_c[sub] = {"n_scenes": len(keep), "n_episodes_per_cell": cells[public[0]]["n_episodes"],
                      "cells": {c: {k: v[k] for k in ("score", "at_fault_rate")} for c, v in cells.items()},
                      "ideal_set_vs": cmp_}
    for c in public:
        same([out_c["all"]["cells"][c]["score"], out_c["all"]["cells"][c]["at_fault_rate"]],
             [hc["cells"][c]["score"], hc["cells"][c]["at_fault_rate"]], f"s5 confirm {c}")
    print("S5 reproduced headroom_matched.json and headroom_confirm.json cells (public)")
    for sub, r in out_m.items():
        pe = [v["gain"] for v in r["planner_effect_score"].values()]
        sv = r["set_vs_best_single"]["proposal"]["score"]
        print(f"  matched {sub:8s} {r['n_scenes']:2d} scenes | planner effect {min(pe):+.3f}..{max(pe):+.3f} | set vs best "
              f"single (proposal) {sv['gain']:+.3f} [{sv['ci95'][0]:+.3f}, {sv['ci95'][1]:+.3f}] | G2 public passes "
              f"{r['g2_matched_public']['passes']}")
    for sub, r in out_c.items():
        h4 = r["ideal_set_vs"]["proposal+margin | Ideal single (mean)"]["score"]
        print(f"  confirm {sub:8s} {r['n_scenes']:2d} scenes | set vs margin single (H4) {h4['gain']:+.3f} "
              f"[{h4['ci95'][0]:+.3f}, {h4['ci95'][1]:+.3f}]")
    save("s5_headroom", {"source": "headroom_matched.json, headroom_confirm.json (per-scene sums; public cells only)",
                         "matched_dev": out_m, "confirm_primary": out_c})


# ------------------------------------------------------------------------------------------- S6 multi-future card
def s6() -> None:
    import multifuture_scorecard as mf
    ref = json.loads(mf.OUT.read_text(encoding="utf-8"))
    orig = mf.per_agent
    keep_now: set = set()

    def per_agent_sub(z):
        r = orig(z)
        m = np.isin(r["scene_id"].astype(str), list(keep_now))
        return {k: v[m] for k, v in r.items()}
    mf.per_agent = per_agent_sub
    z0 = np.load(mf.DUMPS / "seed0" / mf.MODELS["AgentFormer"][0], allow_pickle=True)
    tests = sorted(set(map(str, z0["scene_id"][z0["split"].astype(object) == "test"])))
    out = {}
    for sub, keep in subsets(tests).items():
        keep_now.clear()
        keep_now.update(keep)
        scenes = np.asarray(sorted(keep), dtype=object)
        counts = bootstrap_counts(len(scenes), mf.N_BOOT, 0)
        rng = np.random.default_rng(1)
        out[sub] = {"n_scenes": len(scenes),
                    "models": {label: mf.score_model(fname, scenes, counts, rng) for label, (fname, _) in mf.MODELS.items()}}
    for label, v in ref["models"].items():
        for metric, x in v.items():
            if isinstance(x, dict) and "value" in x:
                same([out["all"]["models"][label][metric]["value"], *out["all"]["models"][label][metric]["ci95"]],
                     [x["value"], *x["ci95"]], f"s6 {label} {metric}")
    print("S6 reproduced multifuture_scorecard.json")
    for sub, r in out.items():
        print(f"  {sub:8s} {r['n_scenes']:2d} scenes | " + "  ".join(
            f"{m.split(' ')[0]} minADE {v['min_ade']['value']:.2f} ES {v['energy_score']['value']:.2f}"
            for m, v in r["models"].items()), flush=True)
    save("s6_multifuture", {"source": "multifuture_scorecard.json (seed dumps, K = 10)", "subsets": out})


# --------------------------------------------------------------------------------------- S7 chaos gap (dev, IDD val)
def s7() -> None:
    import cg_ft_diagnose as fd
    rep = json.loads((fd.DEV / "ft_budget_diagnosis.json").read_text(encoding="utf-8"))
    FT, ks = fd.DEV / "ft_budget", fd.KS
    c, order = fd.vb.canonical_order(fd.vb.IDD_CANON, fd.SPLIT)
    half = fd.vb.load_splits(fd.SPLIT)
    keys = [k[:3] for k in order]
    hb = np.array([half[k[0]] == "test" for k in order])
    scene = np.array([k[0] for k in order])
    pos = {k: i for i, k in enumerate(keys)}

    def flat(path: Path, split: str | None = None) -> np.ndarray:
        p = np.load(path, allow_pickle=True)
        a = fd.agent_ade(p["pred"].astype(float), p["gt"].astype(float), p["fut_mask"].astype(bool))
        dorder = fd.dump_order(split)
        assert len(dorder) == len(a)
        out = np.full(len(order), np.nan)
        for (s, t0, aid, _), v in zip(dorder, a):
            i = pos.get((s, t0, aid))
            if i is not None:
                out[i] = v
        assert not np.isnan(out).any(), path.name
        return out
    cv = flat(fd.DEV / "cv_matched_preds.npz")
    rows = {}
    for m in rep["models"]:
        zs = np.mean([flat(fd.DEV / f) for f in fd.ZS[m]], axis=0)
        refk = []
        for r in range(3):
            t = f"{m}_k{ks[-1]}_r{r}_scr"
            meta = json.loads((FT / f"{t}.json").read_text(encoding="utf-8"))
            dump = FT / (f"{t}_eval_preds.npz" if m in ("stgat", "transformer") else f"{t}_preds.npz")
            refk.append(flat(dump, str(REPO / meta["splits_file"])))
        ref_ade = np.mean(refk, axis=0)
        for hname, sel in (("half A", ~hb), ("half B", hb)):                     # reproduction of D_curve G0
            same(fd.half_stat(scene[sel])(zs[sel] - ref_ade[sel])[0], rep["D_curve"][f"{hname} {m}"]["G0"]["estimate"],
                 f"s7 {hname} {m} G0")
        row = {}
        for sub, keep in subsets(set(scene)).items():
            sel = np.isin(scene, list(keep))
            row[sub] = {"n_scenes": len(keep), "n_agents": int(sel.sum()), "cv_ade": float(cv[sel].mean()),
                        "zero_shot_ade": float(zs[sel].mean()), "reference_ade": float(ref_ade[sel].mean()),
                        "G0": float((zs - ref_ade)[sel].mean())}
        row["per_clean_scene"] = {s: {"half": "B" if half[s] == "test" else "A", "n_agents": int((scene == s).sum()),
                                      "G0": float((zs - ref_ade)[scene == s].mean()),
                                      "zero_shot_ade": float(zs[scene == s].mean()),
                                      "reference_ade": float(ref_ade[scene == s].mean())}
                                  for s in sorted(set(scene) & CLEAN)}
        rows[m] = row
        print(f"  {m:12s} G0 all {row['all']['G0']:+.2f} | clean (3 scenes) {row['clean']['G0']:+.2f} | flagged (7) "
              f"{row['flagged']['G0']:+.2f} | per clean scene " + " ".join(
                  f"{v['G0']:+.2f}" for v in row["per_clean_scene"].values()), flush=True)
    print("S7 reproduced ft_budget_diagnosis.json G0 (both halves)")
    save("s7_chaos_gap_dev", {"source": "ft_budget_diagnosis.json (option 1, primary): zero-shot (AV2-matched, 3 seeds) "
                                        "minus the from-scratch k = 72 reference (3 draws), pooled over both IDD val "
                                        "halves; descriptive only (3 clean scenes)", "models": rows})


# ------------------------------------------------------------------------------------------------------ report
def classify(orig_est: float, orig_sig: bool, est: float, lo: float, hi: float, sig: bool) -> str:
    """The reading rule fixed in the plan."""
    if np.sign(est) != np.sign(orig_est) or not (lo <= orig_est <= hi):
        return "changes"
    return "holds" if sig == orig_sig else "not determinable"


def point_of(h: str, pk: dict, lam: float) -> float:
    """Point estimate of a closed-loop CV hypothesis (> 0 = in the hypothesis' direction), from the package summaries."""
    key = f"band_x{lam:g}"
    names = {"CV": CV, "ignoring agents": "No prediction (ignore agents)"}
    g = lambda n: pk[n].get(key, pk[n].get("band_x1"))                     # noqa: E731  references run at x1 only
    if h.startswith("H3"):
        n = h.split(" ", 1)[1].split(" chosen band")[0]
        return g(n)["score"] - pk[n]["band_x1" if lam == 0 else "band_x0"]["score"]
    body = h.split(" ", 1)[1]
    if " higher score than " in body:
        a, b = body.split(" higher score than ")
        return g(a)["score"] - g(names.get(b, b))["score"]
    a, b = body.split(" fewer at-fault collisions than ")
    return g(names.get(b, b))["at_fault_rate"] - g(a)["at_fault_rate"]


def report() -> None:
    lines = []

    def add(part, claim, sub, verdict, detail):
        lines.append({"part": part, "claim": claim, "subset": sub, "verdict": verdict, "detail": detail})

    s = load("s1_open_loop")["subsets"]
    for p0 in s["all"]["seed_pooled"]["pairs"]:
        for sub in ("clean", "flagged"):
            p = next(x for x in s[sub]["seed_pooled"]["pairs"] if {x["a"], x["b"]} == {p0["a"], p0["b"]})
            sign = 1 if p["a"] == p0["a"] else -1
            est, lo, hi = sign * p["ade_diff_a_minus_b"], *sorted(sign * np.array(p["ci95"]))
            add("S1", f"ADE {p0['a']} vs {p0['b']} (3-seed means)", sub,
                classify(p0["ade_diff_a_minus_b"], not (p0["ci95"][0] <= 0 <= p0["ci95"][1]), est, lo, hi,
                         not (lo <= 0 <= hi)), f"{p0['ade_diff_a_minus_b']:+.3f} -> {est:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for p0 in s["all"]["vs_cv"]:
        for sub in ("clean", "flagged"):
            p = next(x for x in s[sub]["vs_cv"] if x["a"] == p0["a"])
            add("S1", f"ADE {p0['a']} vs CV (seed 0)", sub,
                classify(p0["ade_diff_a_minus_b"], not (p0["ci95"][0] <= 0 <= p0["ci95"][1]), p["ade_diff_a_minus_b"],
                         *p["ci95"], not (p["ci95"][0] <= 0 <= p["ci95"][1])),
                f"{p0['ade_diff_a_minus_b']:+.3f} -> {p['ade_diff_a_minus_b']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]")
    s = load("s2_conformal")
    for m, by in s["models"].items():
        for sub in ("clean", "flagged"):
            a, x = by["all"], by[sub]
            add("S2", f"{m}: regime-A step coverage below the 90% target (shift)", sub,
                classify(a["step_coverage"] - 0.9, a["step_coverage_ci95"][1] < 0.9, x["step_coverage"] - 0.9,
                         x["step_coverage_ci95"][0] - 0.9, x["step_coverage_ci95"][1] - 0.9, x["step_coverage_ci95"][1] < 0.9),
                f"{a['step_coverage']:.3f} -> {x['step_coverage']:.3f} [{x['step_coverage_ci95'][0]:.3f}, "
                f"{x['step_coverage_ci95'][1]:.3f}]")
    s = load("s3_closed_loop_test")["subsets"]
    for sub in ("clean", "flagged"):
        add("S3", "closed-loop test: the pre-registered pick", sub,
            "holds" if s[sub]["decision"]["pick"] == s["all"]["decision"]["pick"] else "changes",
            f"{s['all']['decision']['pick']} -> {s[sub]['decision']['pick']} (tied with leader: "
            f"{len(s[sub]['decision']['tied_on_score'])} of {len(s[sub]['decision']['vs_leader']) + 1})")
        for n, v0 in s["all"]["at_fault_vs_ignore"].items():
            v = s[sub]["at_fault_vs_ignore"][n]
            add("S3", f"closed-loop test: {n} at-fault vs ignoring agents (exploratory)", sub,
                classify(v0["at_fault_diff"], v0["ci95"][1] < 0, v["at_fault_diff"], *v["ci95"], v["ci95"][1] < 0),
                f"{v0['at_fault_diff']:+.3f} -> {v['at_fault_diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    s4_ = load("s4_closed_loop_cv")
    lam = json.loads((B2 / "closed_loop_cv.json").read_text(encoding="utf-8"))["planner"]["band"]
    for name, by in s4_["sets"].items():
        for sub in ("clean", "flagged"):
            if by[sub]["n_scenes"] == 0:
                continue
            add("S4", f"CV {name}: pick", sub, "holds" if by[sub]["decision"]["pick"] == by["all"]["decision"]["pick"]
                else "changes", f"{by['all']['decision']['pick']} -> {by[sub]['decision']['pick']}")
            for fam in ("hypotheses", "secondary_multi_future"):
                for h, v0 in by["all"][fam].items():
                    v = by[sub][fam][h]                       # diff_ci95: > 0 is the hypothesis' direction
                    e0, e = point_of(h, by["all"]["packages"], lam), point_of(h, by[sub]["packages"], lam)
                    add("S4", f"CV {name}: {h}", sub, classify(e0, v0["supported"], e, *v["diff_ci95"], v["supported"]),
                        f"{e0:+.3f} -> {e:+.3f} [{v['diff_ci95'][0]:+.3f}, {v['diff_ci95'][1]:+.3f}]; "
                        f"p_holm {v0['p_holm']:.3f} -> {v['p_holm']:.3f}")
    if s4_.get("amendment3_primary"):
        by = s4_["amendment3_primary"]
        for sub in ("clean", "flagged"):
            for h, v0 in by["all"]["amendment3_hypotheses"].items():
                v = by[sub]["amendment3_hypotheses"][h]
                e0, e = point_of(h, by["all"]["packages"], lam), point_of(h, by[sub]["packages"], lam)
                add("S4", f"CV primary amendment 3: {h}", sub, classify(e0, v0["supported"], e, *v["diff_ci95"], v["supported"]),
                    f"{e0:+.3f} -> {e:+.3f} [{v['diff_ci95'][0]:+.3f}, {v['diff_ci95'][1]:+.3f}]; "
                    f"p_holm {v0['p_holm']:.3f} -> {v['p_holm']:.3f}")
    s = load("s5_headroom")
    m = s["matched_dev"]
    for sub in ("clean", "flagged"):
        for n, v0 in m["all"]["planner_effect_score"].items():
            v = m[sub]["planner_effect_score"][n]
            add("S5", f"headroom: proposal planner beats v2 on score, {n}", sub,
                classify(v0["gain"], v0["ci95"][0] > 0, v["gain"], *v["ci95"], v["ci95"][0] > 0),
                f"{v0['gain']:+.3f} -> {v['gain']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
        for p, v0 in m["all"]["set_vs_best_single"].items():
            v = m[sub]["set_vs_best_single"][p]["score"]
            v0 = v0["score"]
            add("S5", f"headroom: Ideal set vs best ideal single under {p} (score)", sub,
                classify(v0["gain"], not (v0["ci95"][0] <= 0 <= v0["ci95"][1]), v["gain"], *v["ci95"],
                         not (v["ci95"][0] <= 0 <= v["ci95"][1])),
                f"{v0['gain']:+.3f} -> {v['gain']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
        add("S5", "headroom: G2-matched (public planners) does not pass", sub,
            "holds" if m[sub]["g2_matched_public"]["passes"] == m["all"]["g2_matched_public"]["passes"] else "changes",
            f"passes {m['all']['g2_matched_public']['passes']} -> {m[sub]['g2_matched_public']['passes']}")
        c = s["confirm_primary"]
        for cell, v0 in c["all"]["ideal_set_vs"].items():
            v0, v = v0["score"], c[sub]["ideal_set_vs"][cell]["score"]
            add("S5", f"headroom confirm: Ideal set vs {cell} (score)", sub,
                classify(v0["gain"], not (v0["ci95"][0] <= 0 <= v0["ci95"][1]), v["gain"], *v["ci95"],
                         not (v["ci95"][0] <= 0 <= v["ci95"][1])),
                f"{v0['gain']:+.3f} -> {v['gain']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    s = load("s6_multifuture")["subsets"]
    for metric in ("min_ade", "miss_rate", "energy_score"):
        best0 = min(s["all"]["models"], key=lambda k: s["all"]["models"][k][metric]["value"])
        for sub in ("clean", "flagged"):
            best = min(s[sub]["models"], key=lambda k: s[sub]["models"][k][metric]["value"])
            add("S6", f"multi-future: best model on {metric}", sub, "holds" if best == best0 else "changes",
                f"{best0} -> {best}")
    out = DQ / "sensitivity_report.json"
    out.write_text(json.dumps({"rule": "holds / not determinable / changes, as fixed in the plan", "lines": lines,
                               "when": datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    from collections import Counter
    for part in sorted({x["part"] for x in lines}):
        for sub in ("clean", "flagged"):
            cnt = Counter(x["verdict"] for x in lines if x["part"] == part and x["subset"] == sub)
            print(f"{part} {sub:8s} " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())))
    for x in lines:
        if x["verdict"] == "changes":
            print(f"  CHANGES {x['part']} {x['subset']:8s} {x['claim']}: {x['detail']}")
    print("saved:", out.relative_to(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", choices=["s0", "s1", "s2", "s3", "s4", "s4x", "s5", "s6", "s7", "report"])
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    for p in args.parts:
        print(f"=== {p}", flush=True)
        s4x(args.workers) if p == "s4x" else globals()[p]()
    print("0 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
