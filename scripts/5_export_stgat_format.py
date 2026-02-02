"""Export multi-agent sequences in a ST-GAT/SocialGAN-style format.

Many ST-GAT implementations expect the ETH/UCY-style arrays:
- obs_traj:      (N_agents_total, 2, obs_len)
- pred_traj:     (N_agents_total, 2, pred_len)
- obs_traj_rel:  (N_agents_total, 2, obs_len)
- pred_traj_rel: (N_agents_total, 2, pred_len)
- seq_start_end: (N_seqs, 2) indices into the first dimension

This script converts our per-scene tensors from `outputs/multiagent_trajpp_v1/scenes/*.npz`
into a single NPZ containing the above arrays.

Important:
- We only keep agents that are present for *all* timesteps in the (obs+pred) window.
  This matches the assumptions of many ST-GAT repos.
- Time here is in frame-index order (not real seconds). dt is one frame step.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class SeqMeta:
    scene_id: str
    t0: int
    num_agents: int


def _traj_rel_inplace(rel: np.ndarray, traj_abs: np.ndarray) -> None:
    """Fill rel with frame-to-frame diffs within the same trajectory.

    traj_abs: (M, 2, T)
    rel:      (M, 2, T)
    rel[:, :, 0] is set to 0.
    """
    rel.fill(0.0)
    if traj_abs.shape[-1] >= 2:
        rel[:, :, 1:] = traj_abs[:, :, 1:] - traj_abs[:, :, :-1]


def _pred_rel(pred_abs: np.ndarray, obs_abs: np.ndarray) -> np.ndarray:
    """Compute pred relative displacements.

    - pred_rel[..., 0] = pred_abs[..., 0] - obs_abs[..., -1]
    - pred_rel[..., t] = pred_abs[..., t] - pred_abs[..., t-1]
    """
    rel = np.zeros_like(pred_abs, dtype=np.float32)
    if pred_abs.shape[-1] == 0:
        return rel
    rel[:, :, 0] = pred_abs[:, :, 0] - obs_abs[:, :, -1]
    if pred_abs.shape[-1] >= 2:
        rel[:, :, 1:] = pred_abs[:, :, 1:] - pred_abs[:, :, :-1]
    return rel


def export_dataset(
    scenes_dir: Path,
    obs_len: int,
    pred_len: int,
    stride: int,
    min_agents: int,
) -> Tuple[dict, List[SeqMeta]]:
    seq_len = obs_len + pred_len

    obs_all: List[np.ndarray] = []
    pred_all: List[np.ndarray] = []
    obs_rel_all: List[np.ndarray] = []
    pred_rel_all: List[np.ndarray] = []
    agent_id_all: List[np.ndarray] = []

    seq_start_end: List[Tuple[int, int]] = []
    seq_meta: List[SeqMeta] = []

    cursor = 0
    scene_paths = sorted(scenes_dir.glob("*.npz"))
    if not scene_paths:
        raise SystemExit(f"No scene NPZs found in: {scenes_dir}")

    for p in scene_paths:
        scene_id = p.stem
        data = np.load(p, allow_pickle=True)
        pos = np.asarray(data["pos"], dtype=np.float32)  # (T, A, 2)
        mask = np.asarray(data["mask"], dtype=bool)      # (T, A)
        agent_id = np.asarray(data["agent_id"], dtype=np.int32)  # (A,)

        T, A, _ = pos.shape
        if T < seq_len:
            continue

        for t0 in range(0, T - seq_len + 1, stride):
            win_mask = mask[t0 : t0 + seq_len]  # (seq_len, A)
            keep = win_mask.all(axis=0)         # (A,)
            idx = np.where(keep)[0]
            if idx.size < min_agents:
                continue

            win_pos = pos[t0 : t0 + seq_len, idx, :]  # (seq_len, M, 2)
            # to (M, 2, seq_len)
            traj = np.transpose(win_pos, (1, 2, 0)).astype(np.float32)

            obs = traj[:, :, :obs_len]
            pred = traj[:, :, obs_len:]

            obs_rel = np.zeros_like(obs, dtype=np.float32)
            _traj_rel_inplace(obs_rel, obs)

            # Pred rel needs to be relative to the last obs point at t=0
            pred_rel = _pred_rel(pred, obs)

            obs_all.append(obs)
            pred_all.append(pred)
            obs_rel_all.append(obs_rel)
            pred_rel_all.append(pred_rel)
            agent_id_all.append(agent_id[idx])

            start = cursor
            end = cursor + idx.size
            seq_start_end.append((start, end))
            seq_meta.append(SeqMeta(scene_id=scene_id, t0=t0, num_agents=int(idx.size)))
            cursor = end

    if cursor == 0:
        raise SystemExit(
            "No sequences were exported. Try lowering --min-agents, or check that scenes have enough continuous tracks."
        )

    # Concatenate across sequences
    obs_cat = np.concatenate(obs_all, axis=0)
    pred_cat = np.concatenate(pred_all, axis=0)
    obs_rel_cat = np.concatenate(obs_rel_all, axis=0)
    pred_rel_cat = np.concatenate(pred_rel_all, axis=0)
    agent_id_cat = np.concatenate(agent_id_all, axis=0)

    seq_start_end_arr = np.asarray(seq_start_end, dtype=np.int64)

    dataset = {
        "obs_traj": obs_cat,
        "pred_traj": pred_cat,
        "obs_traj_rel": obs_rel_cat,
        "pred_traj_rel": pred_rel_cat,
        "seq_start_end": seq_start_end_arr,
        "agent_id": agent_id_cat,
        "scene_id": np.asarray([m.scene_id for m in seq_meta], dtype=object),
        "t0": np.asarray([m.t0 for m in seq_meta], dtype=np.int32),
    }

    return dataset, seq_meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-dir", default="outputs/multiagent_trajpp_v1/scenes")
    ap.add_argument("--out", default="outputs/stgat_dataset_obs20_pred60.npz")
    ap.add_argument("--obs-len", type=int, default=20)
    ap.add_argument("--pred-len", type=int, default=60)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--min-agents", type=int, default=2)
    ap.add_argument("--write-stats", default="outputs/stgat_dataset_obs20_pred60_stats.json")
    args = ap.parse_args()

    scenes_dir = Path(args.scenes_dir)

    dataset, seq_meta = export_dataset(
        scenes_dir=scenes_dir,
        obs_len=int(args.obs_len),
        pred_len=int(args.pred_len),
        stride=int(args.stride),
        min_agents=int(args.min_agents),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **dataset)

    stats = {
        "format": "stgat_socialgan_v1",
        "scenes_dir": str(scenes_dir).replace("\\", "/"),
        "obs_len": int(args.obs_len),
        "pred_len": int(args.pred_len),
        "stride": int(args.stride),
        "min_agents": int(args.min_agents),
        "num_seqs": int(len(seq_meta)),
        "num_agents_total": int(dataset["obs_traj"].shape[0]),
        "out": str(out_path).replace("\\", "/"),
    }

    with open(args.write_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(
        f"exported ST-GAT dataset: seqs={stats['num_seqs']} agents_total={stats['num_agents_total']} -> {stats['out']}"
    )
    print(f"wrote stats: {args.write_stats}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
