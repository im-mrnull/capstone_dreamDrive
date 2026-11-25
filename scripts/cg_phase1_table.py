"""Chaos-gap Phase 1 comparison table (development, seed 0), from the prediction dumps only.

For every AV2-trained model (AV2-matched; AV2-large where run) on AV2 val (in-domain) and IDD val
(zero-shot), and every IDD-trained benchmark model (seeds 0-2) on IDD val, the evaluation contract's
figures:
  - most-likely ADE / FDE (masked; FDE at the last valid step);
  - best-of-K at the model's native K, and best-of-6: the first 6 samples of a sampler, the 6
    highest-probability modes of a scored-mode model (oracle: uses the future);
  - native-2 Hz ADE: at future steps 5, 10, ..., 60 only (AgentFormer predicts at 2 Hz), over the
    windows with at least one valid keyframe;
  - common support on IDD val: the same figures without the 24 windows where Trajectron++ has no node.
The IDD-trained AutoBot reference is left out (it fails on one half-B scene; halves report).

Usage:
    python scripts/cg_phase1_table.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEV = REPO / "outputs" / "chaos_gap" / "dev"
SEEDS = REPO / "outputs" / "benchmark_v2" / "seed_dumps"
spec = importlib.util.spec_from_file_location("vb", REPO / "scripts" / "viz_scene_bundle.py")
vb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vb)

AV2 = [  # (label, size, dump, kind)
    ("CV (vel_k = 5)", "-", "cv_matched_preds.npz", "point"), ("MLP", "matched", "mlp_matched_eval_preds.npz", "point"),
    ("MLP", "large", "mlp_large_eval_preds.npz", "point"), ("ST-GAT", "matched", "stgat_matched_eval_preds.npz", "point"),
    ("ST-GAT", "large", "stgat_large_eval_preds.npz", "point"), ("Transformer", "matched", "transformer_matched_eval_preds.npz", "point"),
    ("Transformer", "large", "transformer_large_eval_preds.npz", "point"), ("MFP", "matched", "mfp_matched_preds.npz", "modes"),
    ("LED", "matched", "led_matched_eval_preds.npz", "samples"), ("LED", "large", "led_large_init10_resume_eval_preds.npz", "samples"),
    ("AgentFormer", "matched", "agentformer_matched_eval_preds.npz", "samples"), ("AgentFormer", "large", "agentformer_large_preds.npz", "samples"),
    ("AutoBot", "matched", "autobot_matched_ent8_x4_eval_preds.npz", "modes"),
]
IDD = [("ST-GAT", "preds_stgat.npz", "point"), ("Transformer", "preds_transformer.npz", "point"), ("MFP", "preds_mfp.npz", "modes"),
       ("LED", "preds_led.npz", "samples"), ("AgentFormer", "preds_agentformer.npz", "samples"),
       ("Trajectron++", "preds_trajectronpp_canonical.npz", "samples")]
KEY = np.arange(4, 60, 5)


def metrics(pred, gt, fm, sets=None, probs=None, kind="point") -> dict:
    err = np.linalg.norm(pred - gt, axis=-1)
    last = np.array([np.flatnonzero(f)[-1] for f in fm])
    n = np.arange(len(fm))
    out = {"n": len(fm), "ade": float(np.mean([e[f].mean() for e, f in zip(err, fm)])), "fde": float(err[n, last].mean())}
    kf = fm[:, KEY]
    ok = kf.any(1)
    out["ade_2hz"] = float(np.mean([e[KEY][f].mean() for e, f in zip(err[ok], kf[ok])]))
    out["n_2hz"] = int(ok.sum())
    if sets is not None:
        es = np.linalg.norm(sets - gt[:, None], axis=-1)                # (N, K, T)
        ade_k = np.array([[e[f].mean() for e in row] for row, f in zip(es, fm)])
        out["K"] = sets.shape[1]
        out["best_of_K"] = float(ade_k.min(1).mean())
        six = np.argsort(-probs, axis=1)[:, :6] if (kind == "modes" and probs is not None) else np.tile(np.arange(min(6, sets.shape[1])), (len(fm), 1))
        out["best_of_6"] = float(np.take_along_axis(ade_k, six, 1).min(1).mean())
    return out


def load(path: Path, data: Path, splits: Path, windows: bool):
    c, order = vb.canonical_order(data, splits)
    p = np.load(path, allow_pickle=True)
    assert len(p["pred"]) == len(order)
    split = np.array(["val"] * sum(1 for _ in order))
    m = vb.load_splits(splits)
    split = np.array([m[k[0]] for k in order])
    keys = [k[:3] for k in order]
    sets = p["samples"] if "samples" in p.files else p["modes"] if "modes" in p.files else None
    probs = p["probs"] if "probs" in p.files else None
    if probs is None and "mode_logits" in p.files:
        z = p["mode_logits"]; probs = np.exp(z - z.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
    return keys, split, p["pred"].astype(float), p["gt"].astype(float), p["fut_mask"].astype(bool), \
        None if sets is None else sets.astype(float), probs


def main() -> int:
    val_scenes = set(json.loads((REPO / "splits" / "scene_splits.json").read_text(encoding="utf-8"))["splits"]["val"])
    tpp_spec = next(s for k, *_, s in vb.MODELS if k == "tpp_idd")
    fb, _ = vb.read_dump(tpp_spec, "samples", "Trajectron++", set(vb.idd_rows(val_scenes)))
    no_node = {k for k, v in fb.items() if v["fallback"]}
    assert len(no_node) == 24
    rows = []

    def add(label, size, domain, eval_set, keys, sel, pred, gt, fm, sets, probs, kind):
        r = metrics(pred[sel], gt[sel], fm[sel], None if sets is None else sets[sel], None if probs is None else probs[sel], kind)
        if eval_set == "IDD val":
            cs = sel & np.array([k not in no_node for k in keys])
            c = metrics(pred[cs], gt[cs], fm[cs], None if sets is None else sets[cs], None if probs is None else probs[cs], kind)
            r["common_support"] = {"n": c["n"], "ade": c["ade"], "fde": c["fde"]}
        rows.append({"model": label, "size": size, "trained_on": domain, "evaluated_on": eval_set, **r})

    for label, size, f, kind in AV2:
        win = f.startswith(("cv_", "mlp_"))
        keys, split, pred, gt, fm, sets, probs = load(DEV / f, vb.CG / ("av2_matched_dev.npz"), vb.CG_SPLITS, win)
        for s, name in (("val", "AV2 val"), ("test", "IDD val")):
            add(label, size, "AV2" if label != "CV (vel_k = 5)" else "-", name, keys, split == s, pred, gt, fm, sets, probs, kind)
    for seed in (0, 1, 2):
        for label, f, kind in IDD:
            keys, split, pred, gt, fm, sets, probs = load(SEEDS / f"seed{seed}" / f, vb.IDD_CANON, vb.BENCH_SPLITS, False)
            add(label, f"seed {seed}", "IDD", "IDD val", keys, split == "val", pred, gt, fm, sets, probs, kind)
    out = DEV / "phase1_table.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    f3 = lambda v: "   -  " if v is None else f"{v:6.3f}"                 # noqa: E731
    print(f"{'model':13s} {'size':8s} {'train':5s} {'eval':8s} {'n':>5s} {'ADE':>6s} {'FDE':>6s} {'bo6':>6s} {'boK':>6s} K  {'2Hz':>6s} {'n2Hz':>5s} | common support n, ADE")
    for r in rows:
        cs = r.get("common_support")
        print(f"{r['model']:13s} {r['size']:8s} {r['trained_on']:5s} {r['evaluated_on']:8s} {r['n']:5d} {f3(r['ade'])} {f3(r['fde'])} "
              f"{f3(r.get('best_of_6'))} {f3(r.get('best_of_K'))} {r.get('K', 1):2d} {f3(r['ade_2hz'])} {r['n_2hz']:5d} | "
              + (f"{cs['n']}, {cs['ade']:.3f}" if cs else ""))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
