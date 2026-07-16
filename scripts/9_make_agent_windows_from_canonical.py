"""Per-agent window view of the canonical multi-agent dataset (for CV / MLP).

Benchmark v2 requires every model to be evaluated on the SAME population.
This adapter re-expresses each agent of outputs/canonical_world_obs20_pred60.npz
as a single-agent window, so CV and the MLP see exactly the agents the
interaction models see.

Output arrays (windows-style layout):
- obs        (N, obs_len, 2)  centered on the last observed point
- fut        (N, pred_len, 2) same frame; zero-filled where unobserved
- fut_mask   (N, pred_len)    bool
- scene_id   (N,)             per-agent scene id
- agent_type (N,)             IDD-3D class per agent

Heading normalization (default on): each window is rotated so the agent's
mean velocity over the last 5 observed steps points along +x. ADE/FDE are
rotation-invariant, so results remain comparable with unrotated evaluation;
the normalization only makes the regression problem easier for single-agent
models. Agents slower than 0.1 m/s keep the identity rotation.

Usage:
    python scripts/9_make_agent_windows_from_canonical.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HEADING_K = 5          # deltas averaged for the heading estimate
STATIC_SPEED = 0.01    # m per frame (0.1 m/s at 10 fps): below this, no rotation


def rotation_matrices(headings_xy: np.ndarray) -> np.ndarray:
    """(N,2) heading vectors -> (N,2,2) rotations mapping heading to +x."""
    norms = np.linalg.norm(headings_xy, axis=1)
    cos = np.ones_like(norms)
    sin = np.zeros_like(norms)
    moving = norms > STATIC_SPEED * HEADING_K  # norm of summed deltas
    cos[moving] = headings_xy[moving, 0] / norms[moving]
    sin[moving] = headings_xy[moving, 1] / norms[moving]
    # Rotate by -theta: [[cos, sin], [-sin, cos]]
    R = np.stack(
        [np.stack([cos, sin], axis=1), np.stack([-sin, cos], axis=1)],
        axis=1,
    )
    return R.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/canonical_world_obs20_pred60.npz")
    ap.add_argument("--output", default="outputs/agent_windows_canonical_obs20_pred60.npz")
    ap.add_argument("--stats", default="outputs/agent_windows_canonical_obs20_pred60_stats.json")
    ap.add_argument("--no-heading-norm", dest="heading_norm", action="store_false", default=True)
    args = ap.parse_args()

    d = np.load(args.input, allow_pickle=True)
    obs_traj = d["obs_traj"].astype(np.float32)        # (N, 2, obs)
    pred_traj = d["pred_traj"].astype(np.float32)      # (N, 2, pred)
    fut_mask = d["fut_mask"].astype(bool)              # (N, pred)
    agent_type = d["agent_type"].astype(object)        # (N,)
    seq_scene = d["scene_id"].astype(object)           # (S,)
    sse = d["seq_start_end"].astype(np.int64)          # (S, 2)

    # Per-agent scene ids from the per-sequence ids.
    n_agents = obs_traj.shape[0]
    scene_id = np.empty(n_agents, dtype=object)
    for s, (a, b) in enumerate(sse):
        scene_id[a:b] = seq_scene[s]

    obs = np.transpose(obs_traj, (0, 2, 1))   # (N, obs, 2)
    fut = np.transpose(pred_traj, (0, 2, 1))  # (N, pred, 2)

    # Center on last observed point (zero-filled future steps must stay
    # masked-out, so shift only the valid ones; invalid stay at 0).
    center = obs[:, -1:, :].copy()            # (N, 1, 2)
    obs = obs - center
    fut = np.where(fut_mask[:, :, None], fut - center, 0.0).astype(np.float32)

    if args.heading_norm:
        deltas = obs[:, -HEADING_K:, :] - obs[:, -HEADING_K - 1 : -1, :]
        heading = deltas.sum(axis=1)          # (N, 2)
        R = rotation_matrices(heading)        # (N, 2, 2)
        obs = np.einsum("nij,ntj->nti", R, obs).astype(np.float32)
        fut = np.einsum("nij,ntj->nti", R, fut).astype(np.float32)
        fut = np.where(fut_mask[:, :, None], fut, 0.0).astype(np.float32)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs=obs.astype(np.float32),
        fut=fut.astype(np.float32),
        fut_mask=fut_mask,
        scene_id=scene_id,
        agent_type=agent_type,
    )

    stats = {
        "source": str(args.input),
        "heading_norm": bool(args.heading_norm),
        "num_agents": int(n_agents),
        "mean_future_coverage": float(fut_mask.mean()),
        "classes": {str(c): int((agent_type == c).sum()) for c in sorted(set(agent_type.tolist()))},
    }
    Path(args.stats).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {out} (N={n_agents}, heading_norm={args.heading_norm})")
    print(f"wrote {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
