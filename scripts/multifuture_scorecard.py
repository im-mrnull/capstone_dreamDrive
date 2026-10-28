"""Four-criterion scorecard for the multi-future predictors (open loop).

Criteria 1-3 on the frozen split's test windows, 3 seeds, K = 10 futures per agent
(the first 10 of 20 samples for LED / AgentFormer / Trajectron++, as in the closed loop):

  1. prediction quality   most-likely ADE / FDE; minADE@10 / minFDE@10
  2. diversity / coverage miss rate@10 (minFDE > 2 m); endpoint spread (mean pairwise
                          distance of the 10 endpoints); spread / most-likely FDE;
                          implausible-sample rate (class speed / acceleration limits)
  3. probability quality  energy score (a proper scoring rule for sample sets, weighted
                          by the model's mode probabilities where it has them, equal
                          weights otherwise); for models with probabilities also the NLL
                          of the best mode, top-mode accuracy and the expected calibration
                          error (ECE) of the top-mode probability
  4. driving usefulness   closed loop: see docs/wiki/reactive-test.md (not computed here)

Also the real-time filter: median GPU latency per scene snapshot (safety_latency.json;
the multi-future transformer is timed here the same way). Reported, not a decision rule.
CIs: seed-pooled scene bootstrap (scene counts shared across models, seeds resampled per
model), as in bootstrap_ci.py.

Usage:
    python scripts/multifuture_scorecard.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from dd_bootstrap import bootstrap_counts, ci, per_scene_sums, scene_codes

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "outputs" / "benchmark_v2"
DUMPS = BENCH / "seed_dumps"
OUT = BENCH / "multifuture_scorecard.json"
SEEDS = (0, 1, 2)
K = 10
N_BOOT = 2000
MISS_M = 2.0
DT = 0.1
MODELS = {   # label -> (dump file, latency key in safety_latency.json)
    "K-mode decoder": ("preds_mfp.npz", "K-mode set (10 modes)"),
    "Multi-future transformer": ("preds_mftf.npz", None),
    "AgentFormer": ("preds_agentformer.npz", "AgentFormer set (10 samples)"),
    "LED": ("preds_led.npz", "LED set (10 samples)"),
    "Trajectron++": ("preds_trajectronpp_canonical.npz", "Trajectron++ set (10 samples)"),
}
# Physical plausibility of a sample (speeds / accelerations over 0.5 s windows)
PED = {"Pedestrian", "Child"}
TWO_WHEEL = {"MotorcyleRider", "ScooterRider", "BicycleRider", "Motorcycle", "Scooter", "Bicycle",
             "BicycleGroup"}
LIMITS = {"ped": (4.0, 3.0), "two_wheeler": (25.0, 8.0), "vehicle": (35.0, 8.0)}   # (m/s, m/s^2)
ECE_BINS = np.linspace(0.0, 1.0, 11)


def class_group(c: str) -> str:
    return "ped" if c in PED else "two_wheeler" if c in TWO_WHEEL else "vehicle"


def last_valid(fm: np.ndarray) -> np.ndarray:
    return fm.shape[1] - 1 - np.argmax(fm[:, ::-1], axis=1)


def implausible(samples: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """(N,) fraction of each agent's samples breaking its class speed / acceleration limit."""
    w = 5                                                              # 0.5 s windows
    v = np.linalg.norm(samples[:, :, w::w] - samples[:, :, :-w:w], axis=-1) / (w * DT)   # (N,K,11)
    a = np.abs(np.diff(v, axis=-1)) / (w * DT)
    vmax = np.array([LIMITS[class_group(str(c))][0] for c in cls])[:, None]
    amax = np.array([LIMITS[class_group(str(c))][1] for c in cls])[:, None]
    return ((v.max(-1) > vmax) | (a.max(-1) > amax)).mean(axis=1)


def energy_score(samples: np.ndarray, w: np.ndarray, gt: np.ndarray, fm: np.ndarray) -> np.ndarray:
    """(N,) weighted energy score on the valid steps, in metres RMS per step:
    E_w||X - y|| - 1/2 E_w,w'||X - X'||, trajectories as vectors over valid steps / sqrt(T)."""
    m = fm[:, None, :, None]
    t = np.maximum(fm.sum(1), 1)[:, None]
    d_y = np.sqrt((((samples - gt[:, None]) ** 2) * m).sum(axis=(2, 3)) / t)          # (N,K)
    diff = samples[:, :, None] - samples[:, None]                                       # (N,K,K,T,2)
    d_xx = np.sqrt(((diff ** 2) * fm[:, None, None, :, None]).sum(axis=(3, 4)) / t[:, :, None])
    return (w * d_y).sum(1) - 0.5 * np.einsum("nk,nl,nkl->n", w, w, d_xx)


def per_agent(z) -> dict:
    """Row-level values on the test split for one dump."""
    t = z["split"].astype(object) == "test"
    gt, fm, cls = z["gt"][t].astype(np.float64), z["fut_mask"][t].astype(bool), z["agent_type"][t]
    pred, samp = z["pred"][t].astype(np.float64), z["samples"][t][:, :K].astype(np.float64)
    probs = z["probs"][t].astype(np.float64) if "probs" in z.files else None
    n_valid = fm.sum(1)
    lv = last_valid(fm)
    rows = np.arange(len(gt))
    err_ml = np.linalg.norm(pred - gt, axis=-1)
    err_k = np.linalg.norm(samp - gt[:, None], axis=-1)                                 # (N,K,T)
    ade_k = (err_k * fm[:, None]).sum(-1) / n_valid[:, None]
    fde_k = err_k[rows, :, lv]
    end = samp[rows, :, lv]                                                             # (N,K,2)
    spread = np.linalg.norm(end[:, :, None] - end[:, None], axis=-1).sum((1, 2)) / (K * (K - 1))
    w = probs if probs is not None else np.full((len(gt), K), 1.0 / K)
    out = {"scene_id": z["scene_id"][t].astype(object), "n": np.ones(len(gt)),
           "ade_ml": (err_ml * fm).sum(1) / n_valid, "fde_ml": err_ml[rows, lv],
           "min_ade": ade_k.min(1), "min_fde": fde_k.min(1), "miss": (fde_k.min(1) > MISS_M).astype(float),
           "spread": spread, "implausible": implausible(samp, cls),
           "energy": energy_score(samp, w, gt, fm)}
    if probs is not None:
        best = ade_k.argmin(1)
        top = probs.argmax(1)
        conf = probs.max(1)
        out.update({"nll_best": -np.log(np.clip(probs[rows, best], 1e-12, None)),
                    "top_acc": (top == best).astype(float), "conf": conf,
                    "bin": np.clip(np.digitize(conf, ECE_BINS) - 1, 0, len(ECE_BINS) - 2)})
    return out


RATIOS = {   # metric -> (numerator field, denominator field)
    "ade_ml": ("ade_ml", "n"), "fde_ml": ("fde_ml", "n"), "min_ade": ("min_ade", "n"),
    "min_fde": ("min_fde", "n"), "miss_rate": ("miss", "n"), "spread": ("spread", "n"),
    "spread_over_fde": ("spread", "fde_ml"), "implausible_rate": ("implausible", "n"),
    "energy_score": ("energy", "n"), "nll_best": ("nll_best", "n"), "top_mode_accuracy": ("top_acc", "n"),
}


def score_model(fname: str, scenes, counts, rng) -> dict:
    S, B = len(scenes), counts.shape[0]
    per_seed, sums = [], {}
    for s in SEEDS:
        z = np.load(DUMPS / f"seed{s}" / fname, allow_pickle=True)
        r = per_agent(z)
        codes, _ = scene_codes(r["scene_id"], scenes)
        sums_s = {k: per_scene_sums(codes, S, v) for k, v in r.items() if k not in ("scene_id", "conf", "bin")}
        if "conf" in r:                                        # ECE: per-bin scene sums
            nb = len(ECE_BINS) - 1
            for b in range(nb):
                sel = r["bin"] == b
                sums_s[f"ece_n{b}"] = per_scene_sums(codes[sel], S, np.ones(sel.sum()))
                sums_s[f"ece_conf{b}"] = per_scene_sums(codes[sel], S, r["conf"][sel])
                sums_s[f"ece_acc{b}"] = per_scene_sums(codes[sel], S, r["top_acc"][sel])
        per_seed.append(sums_s)
    for k in per_seed[0]:
        sums[k] = np.stack([p[k] for p in per_seed])           # (seeds, S)
    w = rng.multinomial(len(SEEDS), np.full(len(SEEDS), 1.0 / len(SEEDS)), size=B) / len(SEEDS)
    rep = lambda k: counts * (w @ sums[k])                     # noqa: E731  (B, S) resampled sums
    out = {}
    for metric, (num, den) in RATIOS.items():
        if num not in sums:
            continue
        point = float(np.mean([sums[num][i].sum() / sums[den][i].sum() for i in range(len(SEEDS))]))
        out[metric] = {"value": point, "ci95": ci(rep(num).sum(1) / rep(den).sum(1)),
                       "per_seed": [float(sums[num][i].sum() / sums[den][i].sum()) for i in range(len(SEEDS))]}
    if "ece_n0" in sums:
        point = float(np.mean([ece(lambda k, i=i: sums[k][i]) for i in range(len(SEEDS))]))
        out["ece_top_mode"] = {"value": point, "ci95": ci(ece(rep))}
    return out


def ece(get) -> np.ndarray:
    """Expected calibration error from per-bin sums; get(field) -> (..., S) scene sums of
    ece_n{b} (agents), ece_conf{b} (top-mode probability), ece_acc{b} (top mode correct)."""
    nb = len(ECE_BINS) - 1
    n = sum(get(f"ece_n{b}").sum(-1) for b in range(nb))
    return sum(np.abs(get(f"ece_acc{b}").sum(-1) - get(f"ece_conf{b}").sum(-1)) for b in range(nb)) / n


def mftf_latency() -> dict:
    """Median GPU ms per test scene snapshot for the multi-future transformer's 10 modes,
    timed exactly as scripts/safety_latency.py times the other packages."""
    import torch
    from dd_data import CanonicalSeqDataset, collate_pad
    lat = __import__("safety_latency")
    dp = lat._load_script("dump_predictions.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CanonicalSeqDataset(lat.DATA, "test", lat.SPLITS, rot_aug=False, seed=0)
    fn = dp.load_mftf(device, 0, BENCH)
    items = [ds[i] for i in range(len(ds))]
    r = lat.timed(lambda it: (fn.samples_fn(collate_pad([it]).to(device)), fn.probs_fn(collate_pad([it]).to(device))),
                  items, device)
    return r | {"device": str(device)}


def main() -> int:
    z0 = np.load(DUMPS / "seed0" / MODELS["AgentFormer"][0], allow_pickle=True)
    t = z0["split"].astype(object) == "test"
    scenes = np.asarray(sorted(set(map(str, z0["scene_id"][t]))), dtype=object)
    counts = bootstrap_counts(len(scenes), N_BOOT, 0)
    rng = np.random.default_rng(1)
    latency = json.loads((BENCH / "safety_latency.json").read_text(encoding="utf-8"))["packages"]
    out = {}
    for label, (fname, lat_key) in MODELS.items():
        out[label] = score_model(fname, scenes, counts, rng)
        out[label]["latency_ms"] = (latency[lat_key] | {"source": "safety_latency.json"}) if lat_key \
            else mftf_latency() | {"source": "timed here"}
        out[label]["real_time_200ms"] = bool(out[label]["latency_ms"]["median_ms"] <= 200.0)
        print(f"{label:26s} " + "  ".join(f"{m} {v['value']:.3f}" for m, v in out[label].items()
                                          if isinstance(v, dict) and "value" in v)
              + f"  latency {out[label]['latency_ms']['median_ms']:.0f} ms", flush=True)
    payload = {"split": "test", "seeds": list(SEEDS), "k": K, "n_scenes": len(scenes), "n_boot": N_BOOT,
               "miss_m": MISS_M, "plausibility_limits": LIMITS, "models": out,
               "notes": "energy score in metres (RMS per valid step), lower is better; spread in metres; "
                        "ECE over 10 equal-width bins of the top-mode probability",
               "when": datetime.now().isoformat(timespec="seconds")}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("saved:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
