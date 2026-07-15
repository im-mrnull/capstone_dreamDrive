"""Evaluate a Constant Velocity (CV) baseline on windowed trajectories.

Input NPZ (created earlier):
- obs: (N, obs_len, 2) centered so obs[:, -1] == [0,0]
- fut: (N, pred_len, 2) centered with same origin
- scene_id: (N,)

CV model:
- estimate velocity v from last two points (or average over last K deltas)
- predict future: p_t = (t+1) * v  for t=0..pred_len-1

Metrics:
- ADE: mean L2 error over all future timesteps
- FDE: L2 error at final timestep

Example:
python scripts/baseline_cv_eval.py --input-npz outputs/windows_obs20_pred60.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def _compute_cv_predictions(obs, pred_len: int, vel_k: int):
    """obs: (N, obs_len, 2) -> pred: (N, pred_len, 2)"""
    import numpy as np

    if vel_k < 1:
        raise ValueError("vel_k must be >= 1")

    # Use last K velocity increments from obs.
    # If vel_k=1: v = obs[-1] - obs[-2]
    # If vel_k>1: average last K deltas.
    obs_len = obs.shape[1]
    k = min(vel_k, obs_len - 1)
    deltas = obs[:, -k:, :] - obs[:, -k - 1 : -1, :]
    v = deltas.mean(axis=1)  # (N,2)

    steps = (np.arange(1, pred_len + 1, dtype=obs.dtype)).reshape(1, pred_len, 1)
    pred = steps * v[:, None, :]
    return pred


def _ade_fde(pred, fut) -> Tuple[float, float]:
    import numpy as np

    err = np.linalg.norm(pred - fut, axis=-1)  # (N, pred_len)
    ade = float(err.mean())
    fde = float(err[:, -1].mean())
    return ade, fde


def _metrics_by_scene(pred, fut, scene_id) -> Dict[str, Dict[str, float]]:
    import numpy as np

    metrics: Dict[str, Dict[str, float]] = {}
    scenes = np.asarray(scene_id)
    for s in np.unique(scenes):
        idx = np.where(scenes == s)[0]
        ade, fde = _ade_fde(pred[idx], fut[idx])
        metrics[str(s)] = {"count": int(idx.shape[0]), "ade": ade, "fde": fde}
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="CV baseline evaluation")
    parser.add_argument(
        "--input-npz",
        type=str,
        default=str(Path("outputs") / "windows_obs20_pred60.npz"),
        help="Path to windows NPZ",
    )
    parser.add_argument(
        "--vel-k",
        type=int,
        default=1,
        help="Average velocity over last K observed deltas (1 = last-step velocity)",
    )
    parser.add_argument(
        "--save-pred-npz",
        type=str,
        default=str(Path("outputs") / "baselines" / "cv_pred_obs20_pred60.npz"),
        help="Optional path to save predictions NPZ",
    )
    parser.add_argument(
        "--save-metrics-json",
        type=str,
        default=str(Path("outputs") / "baselines" / "cv_metrics_obs20_pred60.json"),
        help="Path to save metrics JSON",
    )
    parser.add_argument(
        "--no-save-pred",
        action="store_true",
        help="Do not save predictions NPZ",
    )
    parser.add_argument(
        "--per-scene",
        action="store_true",
        help="Also compute per-scene ADE/FDE in metrics JSON",
    )
    parser.add_argument(
        "--split-by-scene",
        action="store_true",
        help="Evaluate on a held-out scene-level val split (same protocol as the learned models)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Split seed (used with --split-by-scene)")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Val fraction (used with --split-by-scene)")
    args = parser.parse_args()

    import numpy as np

    input_npz = Path(args.input_npz)
    if not input_npz.exists():
        raise SystemExit(f"Input NPZ not found: {input_npz}")

    data = np.load(input_npz, allow_pickle=False)
    obs = data["obs"].astype(np.float32, copy=False)
    fut = data["fut"].astype(np.float32, copy=False)
    scene_id = data["scene_id"]

    split_kind = "all"
    if args.split_by_scene:
        # Mirrors make_scene_split in model2/model3/model4 so the val scenes match.
        rng = np.random.default_rng(int(args.seed))
        unique = np.unique(scene_id)
        rng.shuffle(unique)
        n_val = max(1, int(round(len(unique) * float(args.val_frac))))
        val_scenes = set(unique[:n_val].tolist())
        is_val = np.array([sid in val_scenes for sid in scene_id], dtype=bool)
        obs, fut, scene_id = obs[is_val], fut[is_val], scene_id[is_val]
        split_kind = "scene"

    pred_len = fut.shape[1]

    pred = _compute_cv_predictions(obs, pred_len=pred_len, vel_k=args.vel_k)
    ade, fde = _ade_fde(pred, fut)

    metrics = {
        "model": "CV",
        "input": str(input_npz.as_posix()),
        "obs_len": int(obs.shape[1]),
        "pred_len": int(pred_len),
        "vel_k": int(args.vel_k),
        "split": split_kind,
        "seed": int(args.seed) if args.split_by_scene else None,
        "val_frac": float(args.val_frac) if args.split_by_scene else None,
        "count": int(obs.shape[0]),
        "ade": ade,
        "fde": fde,
    }

    if args.per_scene:
        metrics["per_scene"] = _metrics_by_scene(pred, fut, scene_id)

    out_metrics = Path(args.save_metrics_json)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"CV baseline results (N={metrics['count']}): ADE={ade:.4f} | FDE={fde:.4f}")
    print(f"Saved metrics: {out_metrics}")

    if not args.no_save_pred:
        out_pred = Path(args.save_pred_npz)
        out_pred.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_pred, pred=pred.astype(np.float32, copy=False), scene_id=scene_id)
        print(f"Saved predictions: {out_pred}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
