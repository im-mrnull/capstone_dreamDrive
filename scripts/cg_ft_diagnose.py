"""Chaos-gap fine-tuning diagnosis and gap recovery (development, CPU only, from saved results).

From one fine-tuning grid's saved metrics JSONs and prediction dumps only (no training; the frozen IDD test
is never read). Grids: ft (best epoch on IDD val half A), ft_fixed (fixed AV2-matched budget), ft_budget
(Phase 4b option 1, the size-matched AV2 budget: the primary result) and ft_stop (option 2, early stopping
on held-out IDD train scenes: a diagnostic; k up to 64; its runs report on all of IDD val).
  A. epoch 0 as a candidate (grids whose val is half A): for each pretrained run, the AV2 checkpoint's own
     half-A ADE (zero-shot) against the selected fine-tuned half-A ADE. Where zero-shot is lower on half A,
     half-A selection over {epoch 0} + epochs would have kept the AV2 weights; the half-B result of that
     rule is reported. ft_stop instead: how often the selected epoch is 0 (the starting weights);
  B. per-scene half-B ADE (and each scene's median speed): CV, zero-shot (mean of AV2 seeds 0-2) and
     fine-tuned pretrained / from-scratch at the smallest and largest k (mean over draws);
  C. half B split into moving / stationary agents (moving: >= 0.5 m/s over the last observed second, the
     balanced-halves definition);
  D. gap recovery on each half separately (no combined number), from per-agent ADE averaged over the
     three draws (zero-shot: over AV2-matched seeds 0-2):
       G0 = ADE(zero-shot) - ADE(reference), the reference = from scratch at the grid's largest k;
       Gk = ADE(zero-shot) - ADE(k, arm);  Recovery(k) = 100 * Gk / G0.
     Recovery is computed only where G0 > 0 and is never capped (negative, or above 100%, as it is); where
     G0's CI includes 0 the half is flagged "no demonstrable gap". Also pretrained - scratch ADE per k.
     95% CIs: scene bootstrap (scripts/dd_bootstrap.py) with one count matrix per half shared by every
     statistic, so differences and ratios are paired. The 50% / 90% recovery points (the smallest k whose
     recovery reaches them) are given only where G0's CI excludes 0 and the curve is monotone within its
     CIs (no step from one k to the next whose CI lies wholly on the worse side); otherwise "not
     identifiable on 5 + 5 dev scenes". The chosen epoch and optimizer steps of every run are recorded.
     Half A is a fair reporting set only for a grid that selects nothing on it (all but ft).
  E. the multi-modal models (MFP, AgentFormer, AutoBot, LED): section D again on best-of-6 ADE (the first 6
     samples of a sampler, the 6 highest-probability modes of a scored-mode model, as in the Phase 1 table;
     an oracle that uses the future), reported alongside the most-likely result, never instead of it.
Models: every model whose grid is complete (all k, draws and arms), or --models. Recovery is on the
deployable most-likely output (MFP: top-1). ST-GAT and transformer runs are read from their contract-format
dumps (dump_predictions.py, *_eval_preds.npz), the others from their own --save-pred-npz (*_preds.npz).
--compare reads several grids' <grid>_diagnosis.json and prints their D sections side by side, aligned by
k position (ft_stop's largest k is 64, the others' 72); writes outputs/chaos_gap/dev/ft_compare.json.

Usage:
    python scripts/cg_ft_diagnose.py [--ft-dir outputs/chaos_gap/dev/ft_budget]
    python scripts/cg_ft_diagnose.py --compare ft_budget ft_stop ft_fixed
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

import cg_budget_rule as br
import dd_bootstrap as bs

REPO = Path(__file__).resolve().parent.parent
DEV = REPO / "outputs" / "chaos_gap" / "dev"
spec = importlib.util.spec_from_file_location("vb", REPO / "scripts" / "viz_scene_bundle.py")
vb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vb)

SPLIT = REPO / "splits" / "chaos_gap" / "ft" / "idd_k72_r0_dev.json"     # halves A (val) / B (test): same in every file
KS = ("05", "10", "20", "40", "72")
KS_STOP = ("05", "10", "20", "40", "64")
MODELS = ("mlp", "stgat", "transformer", "mfp", "agentformer", "autobot", "led")
MULTIMODAL = ("mfp", "agentformer", "autobot", "led")
ARMS = ("pre", "scr")
N_BOOT, SEED = 10000, 0
ZS = {"mlp": ["mlp_matched_eval_preds.npz", "mlp_matched_s1_preds.npz", "mlp_matched_s2_preds.npz"],
      "stgat": ["stgat_matched_eval_preds.npz", "stgat_matched_s1_eval_preds.npz", "stgat_matched_s2_eval_preds.npz"],
      "transformer": ["transformer_matched_eval_preds.npz", "transformer_matched_s1_eval_preds.npz",
                      "transformer_matched_s2_eval_preds.npz"],
      "mfp": ["mfp_matched_preds.npz", "mfp_matched_s1_preds.npz", "mfp_matched_s2_preds.npz"],
      "agentformer": ["agentformer_matched_eval_preds.npz", "agentformer_matched_s1_preds.npz",
                      "agentformer_matched_s2_preds.npz"],
      "autobot": ["autobot_matched_ent8_x4_eval_preds.npz", "autobot_matched_s1_preds.npz", "autobot_matched_s2_preds.npz"],
      "led": ["led_matched_eval_preds.npz", "led_matched_s1_preds.npz", "led_matched_s2_preds.npz"]}
NOT_IDENTIFIABLE = "not identifiable on 5 + 5 dev scenes"


def agent_ade(pred, gt, fm) -> np.ndarray:
    err = np.linalg.norm(pred - gt, axis=-1)
    return np.array([e[f].mean() for e, f in zip(err, fm)])


def best_of_6(p, gt, fm) -> np.ndarray | None:
    """Per-agent best-of-6 ADE as in the Phase 1 table: the first 6 samples of a sampler, the 6
    highest-probability modes of a scored-mode model (MFP mode_logits, AutoBot probs); None for a point model."""
    sets = p["samples"] if "samples" in p.files else p["modes"] if "modes" in p.files else None
    if sets is None:
        return None
    probs = p["probs"] if "probs" in p.files else p["mode_logits"] if "mode_logits" in p.files else None
    six = (np.argsort(-probs, axis=1)[:, :6] if probs is not None                  # softmax keeps the logits' order
           else np.tile(np.arange(min(6, sets.shape[1])), (len(fm), 1)))
    err = np.linalg.norm(sets.astype(float) - gt[:, None], axis=-1)                # (N, K, T)
    ade_k = (err * fm[:, None]).sum(-1) / fm.sum(-1, keepdims=True).clip(1)
    return np.take_along_axis(ade_k, six, 1).min(1)


def val_ade(d: dict) -> float:
    """The selection-set ADE a run reports: point models, MFP top-1, most-likely for the others."""
    v = d["val"]
    return v["ade"] if "ade" in v else v["top1"]["ade"] if "top1" in v else v["most_likely"]["ade"]


def complete(ft_dir: Path, m: str, ks) -> bool:
    return all((ft_dir / f"{m}_k{k}_r{r}_{arm}.json").exists() for k in ks for r in range(3) for arm in ARMS)


@lru_cache(maxsize=None)
def dump_order(split: str | None) -> list:
    """(scene, t0, agent, row) of a dump's rows: val then test of an IDD split file, or (None) the
    AV2-matched file's AV2 val then IDD val."""
    if split is None:
        return vb.canonical_order(vb.CG / "av2_matched_dev.npz", vb.CG_SPLITS)[1]
    return vb.canonical_order(vb.IDD_CANON, Path(split))[1]


def half_stat(scene_ids: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """mean(v) over one half's agents, with its scene-bootstrap replicates; every call shares one count
    matrix, so differences and ratios of two calls are paired."""
    codes, scenes = bs.scene_codes(scene_ids)
    counts = bs.bootstrap_counts(len(scenes), n_boot, seed)
    ones = bs.per_scene_sums(codes, len(scenes), np.ones(len(scene_ids)))

    def stat(v: np.ndarray) -> tuple[float, np.ndarray]:
        num = bs.per_scene_sums(codes, len(scenes), v)
        return float(num.sum() / ones.sum()), bs.ratio_replicates(counts, num, ones)
    return stat


def summary(est: float, reps: np.ndarray) -> dict:
    return {"estimate": est, "ci95": bs.ci(reps)}


def recovery_curve(stat, cv: np.ndarray, z: np.ndarray, per_k: dict, ks) -> dict:
    """Section D for one half and model: per_k[(k, arm)] = per-agent ADE (mean of draws), z = zero-shot."""
    kmax = ks[-1]
    g0, g0r = stat(z - per_k[(kmax, "scr")])
    d = {"cv": summary(*stat(cv)), "zero_shot": summary(*stat(z)), "reference_k": kmax,
         "reference": summary(*stat(per_k[(kmax, "scr")])), "G0": summary(g0, g0r),
         "G0_nonpositive_replicates": float(np.mean(g0r[np.isfinite(g0r)] <= 0))}
    d["gap"] = ("no gap (G0 <= 0): recovery not computed" if g0 <= 0
                else "no demonstrable gap" if d["G0"]["ci95"][0] <= 0 else "demonstrable")
    for arm in ARMS:
        for k in ks:
            gk, gkr = stat(z - per_k[(k, arm)])
            row = {"ade": summary(*stat(per_k[(k, arm)])), "Gk": summary(gk, gkr)}
            if g0 > 0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    row["recovery_pct"] = summary(100 * gk / g0, 100 * gkr / g0r)
            d[f"k{k}_{arm}"] = row
        worse = []                                                       # a step whose CI is wholly on the worse side
        for a, b in zip(ks, ks[1:]):
            gain = summary(*stat(per_k[(a, arm)] - per_k[(b, arm)]))     # > 0: ADE falls from k=a to k=b
            if gain["ci95"][1] < 0:
                worse.append(f"k{a}->k{b}")
        d[f"monotone_{arm}"] = {"monotone": not worse, "worse_steps": worse}
        d[f"points_{arm}"] = recovery_points(d, ks, arm)
    for k in ks:
        d[f"k{k}_pre_minus_scr"] = summary(*stat(per_k[(k, "pre")] - per_k[(k, "scr")]))
    return d


def recovery_points(d: dict, ks, arm: str) -> dict:
    """The smallest k whose recovery reaches 50% / 90%, only on a demonstrable gap and a monotone curve."""
    if d["gap"] != "demonstrable" or not d[f"monotone_{arm}"]["monotone"]:
        return {"50": NOT_IDENTIFIABLE, "90": NOT_IDENTIFIABLE}
    out = {}
    for target in (50, 90):
        hit = next((k for k in ks if d[f"k{k}_{arm}"]["recovery_pct"]["estimate"] >= target), None)
        out[str(target)] = f"k={int(hit)}" if hit else f"not reached by k={int(ks[-1])}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-dir", default=str(DEV / "ft"))
    ap.add_argument("--compare", nargs="+", default=None, metavar="GRID",
                    help="grids (dev/ subdirectory names) whose *_diagnosis.json to put side by side")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=None,
                    help="default: every model whose grid is complete")
    args = ap.parse_args()
    if args.compare:
        return compare(args.compare)
    FT = Path(args.ft_dir)
    stop = any(FT.glob("*_k64_r0_pre.json"))
    ks = KS_STOP if stop else KS
    models = tuple(args.models or [m for m in MODELS if complete(FT, m, ks)])
    missing = [m for m in models if not complete(FT, m, ks)]
    if missing or not models:
        print(f"incomplete grid for {missing or 'every model'} in {FT}")
        return 1
    print(f"models: {', '.join(models)}" + (f" (not complete yet: {', '.join(m for m in MODELS if m not in models)})"
                                            if len(models) < len(MODELS) else ""))
    c, order = vb.canonical_order(vb.IDD_CANON, SPLIT)                   # half A rows, then half B rows
    half = vb.load_splits(SPLIT)
    keys = [k[:3] for k in order]
    hb = np.array([half[k[0]] == "test" for k in order])
    scene = np.array([k[0] for k in order])
    obs = c["obs_traj"][[k[3] for k in order]]                           # (N, 2, 20)
    speed = np.linalg.norm(obs[:, :, -1] - obs[:, :, -11], axis=1)       # m/s over the last observed second
    moving = speed >= 0.5
    pos = {k: i for i, k in enumerate(keys)}
    fut = c["pred_traj"][[k[3] for k in order]].transpose(0, 2, 1).astype(float)   # (N, 60, 2)
    fm_c = c["fut_mask"][[k[3] for k in order]].astype(bool)
    ref_steps = np.where(fm_c[:, 1:] & fm_c[:, :-1], np.linalg.norm(np.diff(fut, axis=1), axis=-1), 0.0)

    def flat(path: Path, split: str | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Per-agent most-likely ADE (and best-of-6, None for a point model) in `order` from a flat dump, keyed
        by (scene, t0, agent); rows outside IDD val (AV2 val, the stop set) are dropped, and every kept row's
        frame-invariant step lengths are checked."""
        p = np.load(path, allow_pickle=True)
        gt, fm = p["gt"].astype(float), p["fut_mask"].astype(bool)
        a, b6 = agent_ade(p["pred"].astype(float), gt, fm), best_of_6(p, gt, fm)
        steps = np.where(fm[:, 1:] & fm[:, :-1], np.linalg.norm(np.diff(gt, axis=1), axis=-1), 0.0)
        dorder = dump_order(split)
        assert len(dorder) == len(a), f"{path.name}: {len(a)} rows vs {len(dorder)} in its split"
        out, out6 = np.full(len(order), np.nan), None if b6 is None else np.full(len(order), np.nan)
        for j, ((s, t0, aid, _), v, st) in enumerate(zip(dorder, a, steps)):
            i = pos.get((s, t0, aid))
            if i is not None:
                assert np.allclose(st, ref_steps[i], atol=1e-3), f"{path.name}: row {i} differs"
                out[i] = v
                if out6 is not None:
                    out6[i] = b6[j]
        return out, out6

    cv = flat(DEV / "cv_matched_preds.npz")[0]
    zs, zs6 = {}, {}
    for m in models:
        both = [flat(DEV / f) for f in ZS[m]]
        zs[m], zs6[m] = [a for a, _ in both], [b for _, b in both]
    ft, ft6 = defaultdict(list), defaultdict(list)                       # (model, k, arm) -> [per-agent ADE per draw]
    meta = {}
    for m in models:
        for k in ks:
            for r in range(3):
                for arm in ARMS:
                    t = f"{m}_k{k}_r{r}_{arm}"
                    meta[t] = json.loads((FT / f"{t}.json").read_text(encoding="utf-8"))
                    dump = FT / (f"{t}_eval_preds.npz" if m in ("stgat", "transformer") else f"{t}_preds.npz")
                    a, b6 = flat(dump, str(REPO / meta[t]["splits_file"]))
                    ft[(m, k, arm)].append(a)
                    ft6[(m, k, arm)].append(b6)
    for v in [cv, *[a for z in zs.values() for a in z]]:
        assert not np.isnan(v).any()
    for v in ft.values():
        assert all(not np.isnan(a).any() for a in v)
    multi = [m for m in models if m in MULTIMODAL]
    for m in multi:                                                      # best-of-6 needs every dump's samples / modes
        assert all(b is not None and not np.isnan(b).any() for b in zs6[m]), f"{m}: zero-shot dump without its set"
        assert all(b is not None and not np.isnan(b).any() for k in ks for arm in ARMS for b in ft6[(m, k, arm)]), \
            f"{m}: a run dump without its samples / modes (--save-samples)"

    mean = lambda a, sel: float(np.nanmean(a[sel]))                       # noqa: E731
    kmin, kmax = ks[0], ks[-1]
    report = {"grid": FT.name, "ks": list(ks), "n_half_a": int((~hb).sum()), "n_half_b": int(hb.sum()),
              "chosen": {t: chosen(d) for t, d in meta.items()}}

    # A. epoch 0 as a selection candidate (pretrained arm), or how often it was chosen (ft_stop)
    rows = []
    if stop:
        report["A_epoch0_chosen"] = {f"{m}_k{k}_{arm}": sum(meta[f"{m}_k{k}_r{r}_{arm}"]["best_epoch"] == 0
                                                            for r in range(3))
                                     for m in models for k in ks for arm in ARMS}
        if "led" in models:                                              # LED's stage 1 (core) has its own epoch 0
            report["A_epoch0_chosen_led_core"] = {f"led_k{k}_{arm}": sum(meta[f"led_k{k}_r{r}_{arm}"]["core_best_epoch"]
                                                                         == 0 for r in range(3))
                                                  for k in ks for arm in ARMS}
    else:
        for m in models:
            for k in ks:
                for r in range(3):
                    t = f"{m}_k{k}_r{r}_pre"
                    zs_a, zs_b = mean(zs[m][r], ~hb), mean(zs[m][r], hb)
                    ft_a, ft_b = val_ade(meta[t]), mean(ft[(m, k, "pre")][r], hb)
                    rows.append({"run": t, "zero_shot_A": zs_a, "finetuned_A": ft_a, "zero_shot_B": zs_b,
                                 "finetuned_B": ft_b, "keeps_epoch0": zs_a < ft_a,
                                 "rule_B": zs_b if zs_a < ft_a else ft_b, "best_epoch": meta[t]["best_epoch"]})
        report["A_epoch0"] = rows

    # B. per-scene half B
    scenes = sorted(set(scene[hb]))
    per = []
    for s in scenes:
        sel = hb & (scene == s)
        row = {"scene": s, "n": int(sel.sum()), "median_speed": float(np.median(speed[sel])),
               "moving_share": float(moving[sel].mean()), "cv": mean(cv, sel)}
        for m in models:
            row[f"{m}_zero_shot"] = float(np.mean([mean(a, sel) for a in zs[m]]))
            for k in (kmin, kmax):
                for arm in ARMS:
                    row[f"{m}_k{k}_{arm}"] = float(np.mean([mean(a, sel) for a in ft[(m, k, arm)]]))
        per.append(row)
    report["B_per_scene"] = per

    # C. moving / stationary on half B, every k
    strat = {}
    for name, sel in (("moving", hb & moving), ("stationary", hb & ~moving), ("all", hb)):
        d = {"n": int(sel.sum()), "cv": mean(cv, sel)}
        for m in models:
            d[f"{m}_zero_shot"] = float(np.mean([mean(a, sel) for a in zs[m]]))
            for k in ks:
                for arm in ARMS:
                    d[f"{m}_k{k}_{arm}"] = float(np.mean([mean(a, sel) for a in ft[(m, k, arm)]]))
        strat[name] = d
    report["C_moving"] = strat

    # D. gap recovery per half, paired scene-bootstrap CIs
    curve, curve6 = {}, {}
    for hname, sel in (("half A", ~hb), ("half B", hb)):
        stat = half_stat(scene[sel])
        for m in models:
            per_k = {(k, arm): np.mean(ft[(m, k, arm)], axis=0)[sel] for k in ks for arm in ARMS}
            curve[f"{hname} {m}"] = recovery_curve(stat, cv[sel], np.mean(zs[m], axis=0)[sel], per_k, ks)
        for m in multi:                                                  # E. best-of-6, alongside
            per_k = {(k, arm): np.mean(ft6[(m, k, arm)], axis=0)[sel] for k in ks for arm in ARMS}
            curve6[f"{hname} {m}"] = recovery_curve(stat, cv[sel], np.mean(zs6[m], axis=0)[sel], per_k, ks)
    report["models"] = list(models)
    report["D_curve"] = curve
    if curve6:
        report["E_best_of_6"] = curve6

    out = DEV / f"{FT.name}_diagnosis.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    if stop:
        print("A. how often the selected epoch is 0 (the starting weights), of 3 draws (pretrained / scratch)")
        e0 = report["A_epoch0_chosen"]
        for m in models:
            print(f"  {m:12s} " + "  ".join(f"k{k} {e0[f'{m}_k{k}_pre']}/{e0[f'{m}_k{k}_scr']}" for k in ks))
        if "led" in models:
            e0 = report["A_epoch0_chosen_led_core"]
            print("  led (core)   " + "  ".join(f"k{k} {e0[f'led_k{k}_pre']}/{e0[f'led_k{k}_scr']}" for k in ks))
    else:
        print(f"A. epoch 0 as a candidate (pretrained arm; half A n={report['n_half_a']}, half B n={report['n_half_b']})")
        for m in models:
            rs = [x for x in rows if x["run"].startswith(m + "_")]
            print(f"  {m:12s} keeps AV2 weights in {sum(x['keeps_epoch0'] for x in rs):2d}/15 runs | half A: zero-shot "
                  f"{np.mean([x['zero_shot_A'] for x in rs]):.2f} vs fine-tuned {np.mean([x['finetuned_A'] for x in rs]):.2f} | "
                  f"half B: zero-shot {np.mean([x['zero_shot_B'] for x in rs]):.2f}, fine-tuned {np.mean([x['finetuned_B'] for x in rs]):.2f}, "
                  f"with epoch 0 {np.mean([x['rule_B'] for x in rs]):.2f}")
    print(f"\nB. half B per scene (mean ADE; zero-shot = AV2 seeds 0-2; k{kmin} / k{kmax} = pretrained/scratch, mean of draws)")
    for r in per:
        print(f"  {r['scene']:24s} n={r['n']:3d} speed={r['median_speed']:5.1f} moving={r['moving_share']:.2f} CV={r['cv']:6.2f}")
        for m in models:
            print(f"      {m:12s} zero-shot {r[m + '_zero_shot']:6.2f} | k{kmin} {r[f'{m}_k{kmin}_pre']:6.2f}/{r[f'{m}_k{kmin}_scr']:6.2f}"
                  f" | k{kmax} {r[f'{m}_k{kmax}_pre']:6.2f}/{r[f'{m}_k{kmax}_scr']:6.2f}")
    print("\nC. half B by motion (pretrained/scratch per k)")
    for name, d in strat.items():
        print(f"  {name:10s} n={d['n']:3d} CV={d['cv']:.2f}")
        for m in models:
            print(f"      {m:12s} zero-shot {d[m + '_zero_shot']:.2f} | " + "  ".join(
                f"k{k} {d[f'{m}_k{k}_pre']:.2f}/{d[f'{m}_k{k}_scr']:.2f}" for k in ks))
    print(f"\nD. gap recovery (95% paired scene-bootstrap CIs; reference = k{kmax} from scratch)")
    for name, d in curve.items():
        print_curve(name, d, ks)
    if curve6:
        print("\nE. best-of-6 ADE (oracle; alongside the most-likely result, never instead of it)")
        for name, d in curve6.items():
            print_curve(name, d, ks)
    print(f"\nwrote {out}")
    return 0


def chosen(d: dict) -> dict:
    """The kept checkpoint's epoch and optimizer steps (from the history, or for runs before it, the steps
    formula of cg_budget_rule)."""
    if d.get("history"):
        out = {"epoch": d["best_epoch"], "steps": next(e["steps"] for e in d["history"] if e["epoch"] == d["best_epoch"])}
        if d.get("core_history"):                                        # LED: stage 1 too
            out.update(core_epoch=d["core_best_epoch"],
                       core_steps=next(e["steps"] for e in d["core_history"] if e["epoch"] == d["core_best_epoch"]))
        return out
    data = REPO / (d.get("input") or d["data"])
    n = br.train_count(_scene_ids(str(data)), REPO / d["splits_file"])
    spe = br.steps_per_epoch(n, d["hyper"].get("train_repeat", 1), d["hyper"]["batch_size"])
    return {"epoch": d["best_epoch"], "steps": d["best_epoch"] * spe}


@lru_cache(maxsize=None)
def _scene_ids(data: str) -> np.ndarray:
    return np.load(data, allow_pickle=True)["scene_id"]


def _fmt(s: dict, pct: bool = False) -> str:
    f = "{:.0f}%" if pct else "{:.2f}"
    return f"{f.format(s['estimate'])} [{f.format(s['ci95'][0])}, {f.format(s['ci95'][1])}]"


def print_curve(name: str, d: dict, ks) -> None:
    print(f"  {name}: CV {_fmt(d['cv'])} | zero-shot {_fmt(d['zero_shot'])} | reference {_fmt(d['reference'])}"
          f" | G0 {_fmt(d['G0'])}: {d['gap']}")
    for arm in ARMS:
        cells = []
        for k in ks:
            row = d[f"k{k}_{arm}"]
            rec = f" ({_fmt(row['recovery_pct'], pct=True)})" if "recovery_pct" in row else ""
            cells.append(f"k{k} {row['ade']['estimate']:.2f}{rec}")
        mono = d[f"monotone_{arm}"]
        print(f"      {arm}: " + " | ".join(cells))
        print(f"           monotone: {mono['monotone']}{' ' + str(mono['worse_steps']) if mono['worse_steps'] else ''};"
              f" 50%: {d[f'points_{arm}']['50']}; 90%: {d[f'points_{arm}']['90']}")
    print("      pre - scratch: " + " | ".join(f"k{k} {_fmt(d[f'k{k}_pre_minus_scr'])}" for k in ks))


def compare(grids: list[str]) -> int:
    """Side-by-side D sections, aligned by k position, with the chosen epochs and steps (median of draws); each
    model in the grids that have it (at least two)."""
    reps = {g: json.loads((DEV / f"{g}_diagnosis.json").read_text(encoding="utf-8")) for g in grids}
    out = {}
    has = {m: [g for g, rep in reps.items() if f"half A {m}" in rep["D_curve"]] for m in MODELS}
    models = [m for m in MODELS if len(has[m]) >= 2]
    for hname in ("half A", "half B"):
        for m in models:
            print(f"\n{hname} {m}  (ADE pre/scratch; recovery of the pretrained arm; epochs / steps, median of draws)")
            for g in has[m]:
                rep = reps[g]
                d, ks = rep["D_curve"][f"{hname} {m}"], rep["ks"]
                ch = rep["chosen"]
                cells = {}
                for k in ks:
                    pre, scr = d[f"k{k}_pre"], d[f"k{k}_scr"]
                    ep = {arm: [ch[f"{m}_k{k}_r{r}_{arm}"] for r in range(3)] for arm in ARMS}
                    cells[f"k{k}"] = {
                        "pre": pre["ade"]["estimate"], "scr": scr["ade"]["estimate"],
                        "recovery_pre": pre.get("recovery_pct"),
                        "pre_minus_scr": d[f"k{k}_pre_minus_scr"],
                        "epochs": {arm: int(np.median([c["epoch"] for c in ep[arm]])) for arm in ARMS},
                        "steps": {arm: int(np.median([c["steps"] for c in ep[arm]])) for arm in ARMS}}
                out[f"{hname} {m} {g}"] = {"zero_shot": d["zero_shot"], "G0": d["G0"], "gap": d["gap"],
                                           "points_pre": d["points_pre"], "cells": cells}
                print(f"  {g:10s} zero-shot {d['zero_shot']['estimate']:.2f} | G0 {_fmt(d['G0'])} {d['gap']}")
                print("             " + " | ".join(
                    f"k{k[1:]} {c['pre']:.2f}/{c['scr']:.2f}"
                    + (f" {c['recovery_pre']['estimate']:.0f}%" if c["recovery_pre"] else "")
                    + f" ({c['epochs']['pre']} ep/{c['steps']['pre']} st)" for k, c in cells.items()))
    path = DEV / "ft_compare.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
