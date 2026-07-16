"""Shared benchmark-v2 data + masked-metric utilities for the multi-agent models.

Used by model3_stgat_train_eval.py and model4_mfp_train_eval.py. Centralizes:
- CanonicalSeqDataset: reads outputs/canonical_world_obs20_pred60.npz, applies the
  frozen scene split from splits/scene_splits.json, optional random scene rotation.
- collate_pad: pads variable agent counts; carries fut_mask and agent_type.
- Masked losses/metrics: every future timestep is weighted by fut_mask; FDE is
  taken at the last VALID step per agent; aggregation is sum-and-divide over the
  whole eval set (no per-batch mean-of-means).
- per_class_summary: per-IDD-3D-class ADE/FDE with displacement normalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from dd_split import load_splits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class Batch:
    obs_abs: torch.Tensor      # (B, M, 2, obs_len)
    pred_abs: torch.Tensor     # (B, M, 2, pred_len)
    obs_rel: torch.Tensor      # (B, M, 2, obs_len)
    pred_rel: torch.Tensor     # (B, M, 2, pred_len)
    agent_mask: torch.Tensor   # (B, M) bool — real agent vs padding
    fut_mask: torch.Tensor     # (B, M, pred_len) bool — observed future step
    agent_type: np.ndarray     # (B, M) object — '' for padding

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            obs_abs=self.obs_abs.to(device),
            pred_abs=self.pred_abs.to(device),
            obs_rel=self.obs_rel.to(device),
            pred_rel=self.pred_rel.to(device),
            agent_mask=self.agent_mask.to(device),
            fut_mask=self.fut_mask.to(device),
            agent_type=self.agent_type,
        )


def _rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


class CanonicalSeqDataset(torch.utils.data.Dataset):
    """Multi-agent sequences from the canonical NPZ, split by the frozen file."""

    def __init__(
        self,
        npz_path: str | Path,
        split: str,
        splits_file: str | Path,
        rot_aug: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        d = np.load(npz_path, allow_pickle=True)
        self.obs_traj = d["obs_traj"].astype(np.float32)
        self.pred_traj = d["pred_traj"].astype(np.float32)
        self.obs_traj_rel = d["obs_traj_rel"].astype(np.float32)
        self.pred_traj_rel = d["pred_traj_rel"].astype(np.float32)
        self.fut_mask = d["fut_mask"].astype(bool)
        self.agent_type = d["agent_type"].astype(object)
        self.seq_start_end = d["seq_start_end"].astype(np.int64)
        self.scene_id = d["scene_id"].astype(object)

        mapping = load_splits(splits_file)
        keep = [i for i in range(self.seq_start_end.shape[0])
                if mapping.get(str(self.scene_id[i])) == split]
        if not keep:
            raise ValueError(f"no sequences for split {split!r}")
        self.seq_indices = np.asarray(keep, dtype=np.int64)
        self.split = split
        self.rot_aug = bool(rot_aug)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.seq_indices.shape[0])

    def __getitem__(self, idx: int):
        s, e = self.seq_start_end[int(self.seq_indices[idx])]
        obs = self.obs_traj[s:e]
        pred = self.pred_traj[s:e]
        obs_rel = self.obs_traj_rel[s:e]
        pred_rel = self.pred_traj_rel[s:e]
        fut_mask = self.fut_mask[s:e]
        atype = self.agent_type[s:e]

        if self.rot_aug and self.split == "train":
            R = _rot2d(float(self._rng.uniform(0.0, 2.0 * math.pi)))
            # arrays are (M, 2, T): rotate the coordinate axis (dim 1)
            obs = np.einsum("ij,mjt->mit", R, obs)
            pred = np.einsum("ij,mjt->mit", R, pred)
            obs_rel = np.einsum("ij,mjt->mit", R, obs_rel)
            pred_rel = np.einsum("ij,mjt->mit", R, pred_rel)

        return obs, pred, obs_rel, pred_rel, fut_mask, atype


def collate_pad(items) -> Batch:
    B = len(items)
    obs_len = items[0][0].shape[-1]
    pred_len = items[0][1].shape[-1]
    max_m = max(it[0].shape[0] for it in items)

    obs_abs = torch.zeros((B, max_m, 2, obs_len), dtype=torch.float32)
    pred_abs = torch.zeros((B, max_m, 2, pred_len), dtype=torch.float32)
    obs_rel = torch.zeros((B, max_m, 2, obs_len), dtype=torch.float32)
    pred_rel = torch.zeros((B, max_m, 2, pred_len), dtype=torch.float32)
    agent_mask = torch.zeros((B, max_m), dtype=torch.bool)
    fut_mask = torch.zeros((B, max_m, pred_len), dtype=torch.bool)
    agent_type = np.full((B, max_m), "", dtype=object)

    for b, (o, p, orl, prl, fm, at) in enumerate(items):
        m = o.shape[0]
        obs_abs[b, :m] = torch.from_numpy(np.ascontiguousarray(o))
        pred_abs[b, :m] = torch.from_numpy(np.ascontiguousarray(p))
        obs_rel[b, :m] = torch.from_numpy(np.ascontiguousarray(orl))
        pred_rel[b, :m] = torch.from_numpy(np.ascontiguousarray(prl))
        agent_mask[b, :m] = True
        fut_mask[b, :m] = torch.from_numpy(fm)
        agent_type[b, :m] = at

    return Batch(obs_abs, pred_abs, obs_rel, pred_rel, agent_mask, fut_mask, agent_type)


# ---------------------------------------------------------------------------
# Masked losses / metrics (torch)
# ---------------------------------------------------------------------------

def rel_valid_mask(fut_mask: torch.Tensor) -> torch.Tensor:
    """(B,M,T) absolute-step mask -> mask of valid GT displacements.

    Displacement t is valid iff steps t and t-1 are both observed (t=0 pairs
    with the always-observed last obs point)."""
    rv = fut_mask.clone()
    rv[..., 1:] &= fut_mask[..., :-1]
    return rv


def masked_reg_loss(pred_rel_hat: torch.Tensor, gt_rel: torch.Tensor,
                    agent_mask: torch.Tensor, fut_mask: torch.Tensor,
                    kind: str = "mse") -> torch.Tensor:
    """Displacement regression masked per valid timestep. Shapes (B,M,2,T)."""
    m = (agent_mask.unsqueeze(-1) & rel_valid_mask(fut_mask)).unsqueeze(2).float()  # (B,M,1,T)
    diff = (pred_rel_hat - gt_rel) * m
    denom = (m.sum() * pred_rel_hat.shape[2]).clamp_min(1.0)
    if kind == "mse":
        return (diff * diff).sum() / denom
    if kind == "huber":
        return torch.nn.functional.smooth_l1_loss(
            diff, torch.zeros_like(diff), reduction="sum") / denom
    raise ValueError(f"unknown loss kind: {kind}")


def masked_variety_loss(pred_rel_k: torch.Tensor, gt_rel: torch.Tensor,
                        agent_mask: torch.Tensor, fut_mask: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Best-of-K displacement loss over valid steps.

    pred_rel_k (B,M,K,2,T); returns (loss, best_mode (B,M))."""
    rv = (agent_mask.unsqueeze(-1) & rel_valid_mask(fut_mask))  # (B,M,T)
    m = rv.unsqueeze(2).unsqueeze(3).float()                    # (B,M,1,1,T)
    diff = (pred_rel_k - gt_rel.unsqueeze(2)) * m
    per_mode = (diff * diff).sum(dim=(3, 4))                    # (B,M,K)
    per_mode = per_mode + (~agent_mask).unsqueeze(2).float() * 1e9
    best_mode = per_mode.argmin(dim=2)

    best = pred_rel_k.gather(
        2, best_mode.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        .expand(-1, -1, 1, 2, gt_rel.shape[-1])).squeeze(2)
    diff_best = (best - gt_rel) * rv.unsqueeze(2).float()
    denom = (rv.float().sum() * 2.0).clamp_min(1.0)
    return (diff_best * diff_best).sum() / denom, best_mode


def abs_from_rel(obs_abs: torch.Tensor, pred_rel: torch.Tensor) -> torch.Tensor:
    """Integrate displacements from the last observed position. (B,M,2,T)."""
    start = obs_abs[:, :, :, -1]
    return start.unsqueeze(-1) + torch.cumsum(pred_rel, dim=-1)


@torch.no_grad()
def per_agent_errors(pred_abs: torch.Tensor, gt_abs: torch.Tensor,
                     agent_mask: torch.Tensor, fut_mask: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked per-agent ADE / FDE / GT path length.

    ADE averages the L2 error over each agent's valid steps; FDE is the error
    at the agent's LAST valid step. Returns flat tensors over real agents."""
    err = torch.linalg.vector_norm(
        (pred_abs - gt_abs).permute(0, 1, 3, 2), dim=-1)           # (B,M,T)
    fm = fut_mask.float()
    n_valid = fm.sum(dim=-1).clamp_min(1.0)                        # (B,M)
    ade = (err * fm).sum(dim=-1) / n_valid

    T = err.shape[-1]
    steps = torch.arange(T, device=err.device).view(1, 1, T)
    last_idx = (steps * fut_mask.long()).max(dim=-1).values        # (B,M)
    fde = err.gather(-1, last_idx.unsqueeze(-1)).squeeze(-1)       # (B,M)

    # GT path length over valid consecutive displacements (for normalization).
    gt_step = torch.linalg.vector_norm(
        (gt_abs[..., 1:] - gt_abs[..., :-1]).permute(0, 1, 3, 2), dim=-1)  # (B,M,T-1)
    rv = (fut_mask[..., 1:] & fut_mask[..., :-1]).float()
    path = (gt_step * rv).sum(dim=-1)                              # (B,M)

    real = agent_mask
    return ade[real], fde[real], path[real]


@torch.no_grad()
def per_agent_errors_bestofk(pred_abs_k: torch.Tensor, gt_abs: torch.Tensor,
                             agent_mask: torch.Tensor, fut_mask: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle min-over-modes per-agent ADE/FDE. pred_abs_k (B,M,K,2,T)."""
    err = torch.linalg.vector_norm(
        (pred_abs_k - gt_abs.unsqueeze(2)).permute(0, 1, 2, 4, 3), dim=-1)  # (B,M,K,T)
    fm = fut_mask.unsqueeze(2).float()
    n_valid = fm.sum(dim=-1).clamp_min(1.0)
    ade_k = (err * fm).sum(dim=-1) / n_valid                        # (B,M,K)

    T = err.shape[-1]
    steps = torch.arange(T, device=err.device).view(1, 1, T)
    last_idx = (steps * fut_mask.long()).max(dim=-1).values         # (B,M)
    fde_k = err.gather(
        -1, last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, err.shape[2], 1)
    ).squeeze(-1)                                                   # (B,M,K)

    ade = ade_k.min(dim=2).values
    fde = fde_k.min(dim=2).values

    gt_step = torch.linalg.vector_norm(
        (gt_abs[..., 1:] - gt_abs[..., :-1]).permute(0, 1, 3, 2), dim=-1)
    rv = (fut_mask[..., 1:] & fut_mask[..., :-1]).float()
    path = (gt_step * rv).sum(dim=-1)

    real = agent_mask
    return ade[real], fde[real], path[real]


# ---------------------------------------------------------------------------
# Aggregation (numpy)
# ---------------------------------------------------------------------------

class ErrorAccumulator:
    """Collects per-agent errors across batches; summarizes once at the end."""

    def __init__(self) -> None:
        self.ade: List[np.ndarray] = []
        self.fde: List[np.ndarray] = []
        self.path: List[np.ndarray] = []
        self.types: List[np.ndarray] = []

    def add(self, ade: torch.Tensor, fde: torch.Tensor, path: torch.Tensor,
            agent_type_batch: np.ndarray, agent_mask: torch.Tensor) -> None:
        self.ade.append(ade.detach().cpu().numpy())
        self.fde.append(fde.detach().cpu().numpy())
        self.path.append(path.detach().cpu().numpy())
        self.types.append(agent_type_batch[agent_mask.detach().cpu().numpy()])

    def summary(self, min_class_n: int = 50) -> dict:
        ade = np.concatenate(self.ade)
        fde = np.concatenate(self.fde)
        path = np.concatenate(self.path)
        types = np.concatenate(self.types)
        out = {
            "ade": float(ade.mean()),
            "fde": float(fde.mean()),
            "n_agents": int(ade.shape[0]),
            "per_class": per_class_summary(ade, fde, path, types, min_class_n),
        }
        return out


def masked_ade_fde_np(pred: np.ndarray, gt: np.ndarray, fut_mask: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy masked per-agent errors for single-agent windows.

    pred/gt (N, T, 2), fut_mask (N, T) -> (ade, fde, gt_path_length) per agent.
    ADE over valid steps; FDE at the last valid step."""
    err = np.linalg.norm(pred - gt, axis=-1)                      # (N, T)
    fm = fut_mask.astype(np.float64)
    n_valid = np.maximum(fm.sum(axis=1), 1.0)
    ade = (err * fm).sum(axis=1) / n_valid

    T = err.shape[1]
    last_idx = (np.arange(T)[None, :] * fut_mask).max(axis=1).astype(int)
    fde = err[np.arange(err.shape[0]), last_idx]

    step = np.linalg.norm(gt[:, 1:] - gt[:, :-1], axis=-1)        # (N, T-1)
    rv = (fut_mask[:, 1:] & fut_mask[:, :-1]).astype(np.float64)
    path = (step * rv).sum(axis=1)
    return ade, fde, path


def summarize_np(ade: np.ndarray, fde: np.ndarray, path: np.ndarray,
                 types: np.ndarray, min_class_n: int = 50) -> dict:
    return {
        "ade": float(ade.mean()),
        "fde": float(fde.mean()),
        "n_agents": int(ade.shape[0]),
        "per_class": per_class_summary(ade, fde, path, types, min_class_n),
    }


def per_class_summary(ade: np.ndarray, fde: np.ndarray, path: np.ndarray,
                      types: np.ndarray, min_class_n: int = 50) -> dict:
    per_class: Dict[str, dict] = {}
    for cls in sorted(set(types.tolist())):
        sel = types == cls
        mean_path = float(path[sel].mean())
        per_class[str(cls)] = {
            "n": int(sel.sum()),
            "ade": float(ade[sel].mean()),
            "fde": float(fde[sel].mean()),
            "mean_gt_path_m": mean_path,
            # error per meter actually traveled; static agents -> None
            "ade_per_path": float(ade[sel].mean() / mean_path) if mean_path > 0.5 else None,
        }
    eligible = {c: v for c, v in per_class.items() if v["n"] >= min_class_n}
    disparity = None
    if len(eligible) >= 2:
        vals = [v["ade"] for v in eligible.values()]
        disparity = float(max(vals) / max(min(vals), 1e-9))
    return {"classes": per_class, "min_class_n": min_class_n,
            "ade_disparity_worst_over_best": disparity}
