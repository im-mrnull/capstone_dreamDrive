"""Constant Velocity (CV) baseline on the canonical per-agent windows (v2).

Input: outputs/agent_windows_canonical_obs20_pred60.npz — the per-agent view of
the canonical multi-agent dataset, so CV is evaluated on exactly the same
population as every other model. Windows are centered on the last observed
point (and heading-normalized, which does not affect CV or the metrics).

CV model: velocity = mean of the last k observed deltas; future positions are
linear extrapolation. Deterministic — one run, no seeds. Metrics are masked by
fut_mask (ADE over valid steps; FDE at last valid step) and reported for both
the val and test splits of splits/scene_splits.json, with per-class breakdown.

Usage:
    python scripts/model1_baseline_cv_eval.py --vel-k 5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from dd_split import load_splits
from dd_data import masked_ade_fde_np, summarize_np


def compute_cv_predictions(obs: np.ndarray, pred_len: int, vel_k: int) -> np.ndarray:
    """obs (N, obs_len, 2) centered -> pred (N, pred_len, 2)."""
    if vel_k < 1:
        raise ValueError("vel_k must be >= 1")
    obs_len = obs.shape[1]
    k = min(vel_k, obs_len - 1)
    deltas = obs[:, -k:, :] - obs[:, -k - 1 : -1, :]
    v = deltas.mean(axis=1)
    steps = np.arange(1, pred_len + 1, dtype=obs.dtype).reshape(1, pred_len, 1)
    return steps * v[:, None, :]


def main() -> int:
    ap = argparse.ArgumentParser(description="CV baseline (benchmark v2)")
    ap.add_argument("--input-npz", default="outputs/agent_windows_canonical_obs20_pred60.npz")
    ap.add_argument("--splits-file", default="splits/scene_splits.json")
    ap.add_argument("--vel-k", type=int, default=5)
    ap.add_argument("--save-metrics-json", default="outputs/benchmark_v2/cv_metrics.json")
    args = ap.parse_args()

    d = np.load(args.input_npz, allow_pickle=True)
    obs = d["obs"].astype(np.float32)
    fut = d["fut"].astype(np.float32)
    fut_mask = d["fut_mask"].astype(bool)
    scene_id = d["scene_id"].astype(object)
    agent_type = d["agent_type"].astype(object)

    mapping = load_splits(args.splits_file)
    split_of = np.array([mapping.get(str(s), "?") for s in scene_id])

    pred_len = fut.shape[1]
    pred = compute_cv_predictions(obs, pred_len=pred_len, vel_k=int(args.vel_k))

    results = {}
    for split in ("val", "test"):
        sel = split_of == split
        ade, fde, path = masked_ade_fde_np(pred[sel], fut[sel], fut_mask[sel])
        results[split] = summarize_np(ade, fde, path, agent_type[sel])
        print(f"{split}: ADE={results[split]['ade']:.4f} FDE={results[split]['fde']:.4f} "
              f"(n={results[split]['n_agents']})")

    metrics = {
        "model": "CV",
        "input": str(args.input_npz),
        "splits_file": str(args.splits_file),
        "obs_len": int(obs.shape[1]),
        "pred_len": int(pred_len),
        "vel_k": int(args.vel_k),
        "val": results["val"],
        "test": results["test"],
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": "Benchmark v2: canonical per-agent windows, frozen splits, masked metrics.",
    }
    out = Path(args.save_metrics_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved metrics: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
