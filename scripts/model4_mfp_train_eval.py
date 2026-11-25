"""Train/evaluate the K-mode variety decoder on the canonical v2 dataset.

"K-mode variety decoder" (formerly mislabeled "MFP"): predicts K futures per
agent, trained with a masked best-of-K variety loss plus a learned
mode-scoring head (CE to the oracle-best mode) for a deployable top-1.

Benchmark v2 interface (matches model3):
- canonical NPZ + frozen splits/scene_splits.json; seeds affect init only
- masked per-timestep losses/metrics; FDE at last valid step; per-class breakdown
- model selection on val oracle ADE; oracle AND top-1 are then both reported
  from that single best checkpoint on val and test (the v1 practice of quoting
  oracle and top-1 from different epochs is gone).

--init-ckpt starts from another run's weights (fine-tuning); unset, nothing changes.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from dd_data import (
    Batch,
    CanonicalSeqDataset,
    ErrorAccumulator,
    collate_pad,
    masked_variety_loss,
    per_agent_errors,
    per_agent_errors_bestofk,
    rel_valid_mask,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def split_batch(b: Batch, max_slots: int) -> List[Batch]:
    """[b], unless its padded agent slots (sequences x most agents) exceed max_slots (0: never split); then
    runs of consecutive whole sequences, each padded only to its own most agents, of at most max_slots slots
    (a single sequence larger than that is a part by itself)."""
    B, M = b.agent_mask.shape
    if max_slots <= 0 or B * M <= max_slots:
        return [b]
    n = b.agent_mask.sum(dim=1).tolist()                                 # agents, packed first by collate_pad
    parts, i = [], 0
    while i < B:
        j = i + 1
        while j < B and (j + 1 - i) * max(n[i:j + 1]) <= max_slots:
            j += 1
        m = max(n[i:j])
        parts.append(Batch(b.obs_abs[i:j, :m], b.pred_abs[i:j, :m], b.obs_rel[i:j, :m], b.pred_rel[i:j, :m],
                           b.agent_mask[i:j, :m], b.fut_mask[i:j, :m], b.agent_type[i:j, :m]))
        i = j
    return parts


def valid_steps(b: Batch) -> torch.Tensor:
    """The valid displacement count that masked_variety_loss normalizes by (up to its 2 * w factor)."""
    return (b.agent_mask.unsqueeze(-1) & rel_valid_mask(b.fut_mask)).sum()


class AgentSelfAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                         dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(h, h, h, key_padding_mask=~agent_mask, need_weights=False)
        h = self.ln1(h + self.drop(attn_out))
        h2 = self.ff(h)
        return self.ln2(h + self.drop(h2))


class GraphAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(hidden_dim, attn_dim)
        self.k = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(hidden_dim, attn_dim)
        self.out = nn.Linear(hidden_dim + attn_dim, hidden_dim)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, M, _ = h.shape
        q, k, v = self.q(h), self.k(h), self.v(h)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(k.shape[-1])
        key_mask = mask.unsqueeze(1).expand(B, M, M)
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        query_mask = mask.unsqueeze(2).expand(B, M, M)
        scores = scores.masked_fill(~query_mask, 0.0)
        attn = torch.softmax(scores, dim=-1) * query_mask.float()
        ctx = torch.matmul(attn, v)
        return torch.tanh(self.out(torch.cat([h, ctx], dim=-1)))


class MFPModel(nn.Module):
    def __init__(
        self,
        obs_len: int,
        pred_len: int,
        k: int,
        emb_dim: int = 64,
        hidden_dim: int = 256,
        pool: str = "mha",
        attn_dim: int = 128,
        num_heads: int = 4,
        gat_layers: int = 2,
        dropout: float = 0.1,
        mode_emb_dim: int | None = None,
        content_scorer: bool = False,
    ) -> None:
        super().__init__()
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.k = int(k)
        self.content_scorer = bool(content_scorer)

        self.emb = nn.Linear(2, emb_dim)
        self.encoder = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim,
                               num_layers=1, batch_first=True)

        self.pool_type = str(pool)
        if self.pool_type not in {"none", "simple", "mha"}:
            raise ValueError("pool must be one of: none, simple, mha")
        self.simple_pool = GraphAttentionPool(hidden_dim=hidden_dim, attn_dim=attn_dim)
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.mha_layers = nn.ModuleList(
            [AgentSelfAttentionLayer(hidden_dim, num_heads, dropout) for _ in range(gat_layers)]
        )

        if mode_emb_dim is None:
            mode_emb_dim = emb_dim
        self.mode_emb = nn.Embedding(self.k, mode_emb_dim)
        self.dec_in = nn.Linear(2, emb_dim)
        self.mode_to_emb = nn.Linear(mode_emb_dim, emb_dim, bias=False)
        self.decoder_cell = nn.LSTMCell(input_size=emb_dim, hidden_size=hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)
        self.init_h = nn.Linear(hidden_dim, self.k * hidden_dim)
        self.init_c = nn.Linear(hidden_dim, self.k * hidden_dim)

        # Mode-scoring head. The default (context-only) scores each mode from its
        # per-mode initial state h0 -- BEFORE the trajectory exists, so it chooses
        # blind. The content-aware variant scores from concat(h0, traj_enc(feats))
        # where feats summarize the mode's DECODED trajectory (endpoint, waypoints,
        # path length, heading), letting the scorer judge what it is picking.
        self.traj_feat_dim = 3 * 2 + 2 + 3  # 3 waypoints + endpoint + 3 scalars
        traj_emb_dim = hidden_dim // 2
        self.traj_enc = nn.Sequential(
            nn.Linear(self.traj_feat_dim, traj_emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(traj_emb_dim, traj_emb_dim),
        )
        score_in = hidden_dim + (traj_emb_dim if self.content_scorer else 0)
        self.mode_score = nn.Sequential(
            nn.Linear(score_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.drop = nn.Dropout(dropout)

    def _traj_features(self, pred_rel_k: torch.Tensor) -> torch.Tensor:
        """Per-mode content features from decoded displacements. (B,M,K,F).

        Detached from the generator: the scorer reads the trajectory but its
        gradient must not reshape the modes (those are trained by variety loss)."""
        rel = pred_rel_k.detach()                          # (B,M,K,2,T)
        pos = torch.cumsum(rel, dim=-1)                    # absolute-from-start
        T = pos.shape[-1]
        wp = [pos[..., max(0, T // 4 - 1)], pos[..., max(0, T // 2 - 1)],
              pos[..., max(0, (3 * T) // 4 - 1)]]          # 3 x (B,M,K,2)
        endpoint = pos[..., -1]                            # (B,M,K,2)
        path_len = rel.norm(dim=3).sum(dim=-1, keepdim=True)          # (B,M,K,1)
        straight = endpoint.norm(dim=-1, keepdim=True)               # (B,M,K,1)
        last_step = rel[..., -1]
        heading = torch.atan2(last_step[..., 1], last_step[..., 0]).unsqueeze(-1)
        return torch.cat(wp + [endpoint, path_len, straight, heading], dim=-1)

    def forward(self, obs_rel: torch.Tensor, agent_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M, _, obs_len = obs_rel.shape
        assert obs_len == self.obs_len

        x = obs_rel.permute(0, 1, 3, 2).contiguous().view(B * M, obs_len, 2)
        x = self.drop(torch.relu(self.emb(x)))
        _, (h_n, _) = self.encoder(x)
        h_last = h_n[-1].view(B, M, -1)

        if self.pool_type == "none":
            h_ctx = h_last
        elif self.pool_type == "simple":
            h_ctx = self.simple_pool(h_last, agent_mask)
        else:
            h_ctx = h_last
            for layer in self.mha_layers:
                h_ctx = layer(h_ctx, agent_mask)

        h0 = self.init_h(h_ctx).view(B, M, self.k, -1)
        c0 = self.init_c(h_ctx).view(B, M, self.k, -1)

        h = h0.reshape(B * M * self.k, -1)
        c = c0.reshape(B * M * self.k, -1)
        mode_ids = torch.arange(self.k, device=obs_rel.device).view(1, 1, self.k).expand(B, M, self.k)
        mode_e = self.mode_to_emb(self.mode_emb(mode_ids)).reshape(B * M * self.k, -1)

        prev_rel = obs_rel[:, :, :, -1].contiguous().unsqueeze(2).expand(B, M, self.k, 2)
        prev_rel = prev_rel.reshape(B * M * self.k, 2)

        preds: List[torch.Tensor] = []
        for _t in range(self.pred_len):
            inp = self.drop(torch.relu(self.dec_in(prev_rel) + mode_e))
            h, c = self.decoder_cell(inp, (h, c))
            step = self.out(self.drop(h))
            preds.append(step.view(B, M, self.k, 2))
            prev_rel = step

        pred_rel_k = torch.stack(preds, dim=-1)  # (B, M, K, 2, T)

        # Score modes AFTER decoding. Content-aware: from each mode's own
        # trajectory; default: context-only (h0), i.e. blind to the rollout.
        if self.content_scorer:
            feats = self.traj_enc(self._traj_features(pred_rel_k))  # (B,M,K,emb)
            score_in = torch.cat([h0, feats], dim=-1)
        else:
            score_in = h0
        mode_logits = self.mode_score(score_in).squeeze(-1)  # (B, M, K)
        return pred_rel_k, mode_logits


def abs_from_rel_k(obs_abs: torch.Tensor, pred_rel_k: torch.Tensor) -> torch.Tensor:
    start = obs_abs[:, :, :, -1]
    return start.unsqueeze(2).unsqueeze(-1) + torch.cumsum(pred_rel_k, dim=-1)


def top1_abs(pred_rel_k: torch.Tensor, mode_logits: torch.Tensor,
             obs_abs: torch.Tensor) -> torch.Tensor:
    top1_k = mode_logits.argmax(dim=-1)
    pred_rel = pred_rel_k.gather(
        2, top1_k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        .expand(-1, -1, 1, 2, pred_rel_k.shape[-1])).squeeze(2)
    start = obs_abs[:, :, :, -1]
    return start.unsqueeze(-1) + torch.cumsum(pred_rel, dim=-1)


@torch.no_grad()
def evaluate(model: MFPModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    acc_oracle = ErrorAccumulator()
    acc_top1 = ErrorAccumulator()
    for batch in loader:
        b = batch.to(device)
        pred_rel_k, mode_logits = model(b.obs_rel, b.agent_mask)
        pred_abs_k = abs_from_rel_k(b.obs_abs, pred_rel_k)
        ade, fde, path = per_agent_errors_bestofk(pred_abs_k, b.pred_abs, b.agent_mask, b.fut_mask)
        acc_oracle.add(ade, fde, path, b.agent_type, b.agent_mask)

        pred_abs_1 = top1_abs(pred_rel_k, mode_logits, b.obs_abs)
        ade1, fde1, path1 = per_agent_errors(pred_abs_1, b.pred_abs, b.agent_mask, b.fut_mask)
        acc_top1.add(ade1, fde1, path1, b.agent_type, b.agent_mask)
    return {"oracle_best_of_k": acc_oracle.summary(), "top1": acc_top1.summary()}


@torch.no_grad()
def dump_predictions(model: MFPModel, loaders: dict, device: torch.device, out_path: str) -> None:
    """Flat per-agent val+test predictions, one row per real agent in canonical order: the top-1
    pred (N,T,2), all K modes (N,K,T,2) with their scores mode_logits (N,K) (best-of-K over the
    top-scored modes), gt (N,T,2), fut_mask (N,T), agent_type, split and scene_id."""
    model.eval()
    P, Q, L, G, M, C, S, Z = [], [], [], [], [], [], [], []
    for split, loader in loaders.items():
        ds, seq = loader.dataset, 0
        for batch in loader:
            b = batch.to(device)
            pred_rel_k, mode_logits = model(b.obs_rel, b.agent_mask)
            modes = abs_from_rel_k(b.obs_abs, pred_rel_k).permute(0, 1, 2, 4, 3).float().cpu().numpy()
            top1 = top1_abs(pred_rel_k, mode_logits, b.obs_abs).permute(0, 1, 3, 2).float().cpu().numpy()
            gt = b.pred_abs.permute(0, 1, 3, 2).cpu().numpy()
            logits, am, fm = mode_logits.float().cpu().numpy(), b.agent_mask.cpu().numpy(), b.fut_mask.cpu().numpy()
            for i in range(am.shape[0]):
                scene = str(ds.scene_id[int(ds.seq_indices[seq + i])])
                for j in np.flatnonzero(am[i]):
                    P.append(top1[i, j]); Q.append(modes[i, j]); L.append(logits[i, j]); G.append(gt[i, j])
                    M.append(fm[i, j]); C.append(str(b.agent_type[i, j])); S.append(split); Z.append(scene)
            seq += am.shape[0]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, pred=np.asarray(P, dtype=np.float32), modes=np.asarray(Q, dtype=np.float32),
        mode_logits=np.asarray(L, dtype=np.float32), gt=np.asarray(G, dtype=np.float32),
        fut_mask=np.asarray(M, dtype=bool), agent_type=np.asarray(C, dtype=object),
        split=np.asarray(S, dtype=object), scene_id=np.asarray(Z, dtype=object),
        model="K-mode decoder (top-1)")
    print(f"saved val+test predictions: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/canonical_world_obs20_pred60.npz")
    ap.add_argument("--splits-file", default="splits/scene_splits.json")
    ap.add_argument("--save-metrics-json", default="outputs/benchmark_v2/mfp_metrics.json")
    ap.add_argument("--save-ckpt", default="outputs/benchmark_v2/mfp_ckpt.pt")
    ap.add_argument("--init-ckpt", default=None,
                    help="start from these weights (another run's --save-ckpt; fine-tuning)")
    ap.add_argument("--save-pred-npz", default=None,
                    help="also dump flat per-agent val+test predictions (top-1, all K modes, mode_logits)")
    ap.add_argument("--eval-only", action="store_true",
                    help="load --save-ckpt, skip training, evaluate val and test of --data / --splits-file")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--log-every", type=int, default=25)

    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--keep", choices=["best", "last"], default="best",
                    help="best: the checkpoint with the lowest val --select metric; last: the final epoch (no "
                         "val-based choice). Models 2, 3 and 5 call this --select, which here names the metric")
    ap.add_argument("--select-include-init", action="store_true",
                    help="with --keep best: the starting weights (epoch 0) are also a candidate")
    ap.add_argument("--lr-milestones", type=int, nargs="*", default=None,
                    help="fixed schedule: halve the lr at these epochs (replaces the val plateau schedule; given with no epochs: a constant lr)")
    ap.add_argument("--train-repeat", type=int, default=1,
                    help="repeat the training set this many times per epoch (fine-tuning: equal optimizer steps across training-set sizes; schedules and patience stay in epochs)")
    ap.add_argument("--max-batch-slots", type=int, default=0,
                    help="a training batch with more padded agent slots (sequences x most agents) than this runs as "
                         "parts of whole sequences, gradients accumulated with each part's loss weighted by its "
                         "share of the batch, so it is still one optimizer step on the batch's loss (fits the GPU "
                         "memory cap on scenes with many agents); 0: never split")
    ap.add_argument("--lr-plateau", action="store_true", default=True)
    ap.add_argument("--no-lr-plateau", dest="lr_plateau", action="store_false")
    ap.add_argument("--lr-plateau-factor", type=float, default=0.5)
    ap.add_argument("--lr-plateau-patience", type=int, default=10)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--rot-aug", action="store_true", default=False)

    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--pool", choices=["none", "simple", "mha"], default="mha")
    ap.add_argument("--attn-dim", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--gat-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mode-cls-weight", type=float, default=0.1)
    ap.add_argument("--ewta-frac", type=float, default=0.0,
                    help="anneal the winner set K->1 over this fraction of epochs "
                         "(0 = classic WTA variety loss)")
    ap.add_argument("--mode-target", choices=["hard", "soft"], default="hard",
                    help="mode-head CE target: hard argmin index or detached "
                         "softmax(-mode_mse/tau) distribution")
    ap.add_argument("--mode-tau", type=float, default=1.0)
    ap.add_argument("--select", choices=["oracle", "top1"], default="oracle",
                    help="val metric for checkpoint selection; top1 selects for "
                         "the deployable output")
    ap.add_argument("--content-scorer", action="store_true", default=False,
                    help="score modes from their decoded trajectory (endpoint, "
                         "waypoints, path length, heading) instead of context only")

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-mem-frac", type=float, default=None,
                    help="cap the CUDA cache at this fraction of the card (Windows spills an "
                         "uncapped cache into shared system memory instead of raising OOM)")
    args = ap.parse_args()

    if args.eval_only:
        # architecture must match the trained checkpoint
        saved = torch.load(args.save_ckpt, map_location="cpu", weights_only=False)["args"]
        for key in ("k", "emb_dim", "hidden_dim", "pool", "attn_dim", "num_heads", "gat_layers",
                    "dropout", "content_scorer"):
            setattr(args, key, saved[key])
        args.epochs = 0

    device = torch.device(args.device)
    if args.gpu_mem_frac and device.type == "cuda":   # free cached blocks at the cap, never spill
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)   # current device
    set_seed(int(args.seed))

    def make_loader(split: str, shuffle: bool, rot_aug: bool = False) -> DataLoader:
        ds = CanonicalSeqDataset(args.data, split, args.splits_file,
                                 rot_aug=rot_aug, seed=int(args.seed))
        if split == "train" and int(args.train_repeat) > 1:
            ds = ConcatDataset([ds] * int(args.train_repeat))
        return DataLoader(ds, batch_size=int(args.batch_size), shuffle=shuffle,
                          num_workers=0, collate_fn=collate_pad)

    train_loader = make_loader("train", shuffle=True, rot_aug=bool(args.rot_aug))
    val_loader = make_loader("val", shuffle=False)
    test_loader = make_loader("test", shuffle=False)

    obs0, pred0, *_ = (val_loader if args.eval_only else train_loader).dataset[0]
    obs_len, pred_len = obs0.shape[-1], pred0.shape[-1]

    model = MFPModel(
        obs_len=obs_len, pred_len=pred_len, k=int(args.k),
        emb_dim=int(args.emb_dim), hidden_dim=int(args.hidden_dim),
        pool=str(args.pool), attn_dim=int(args.attn_dim),
        num_heads=int(args.num_heads), gat_layers=int(args.gat_layers),
        dropout=float(args.dropout), content_scorer=bool(args.content_scorer),
    ).to(device)
    if args.init_ckpt:                                                     # fine-tuning
        model.load_state_dict(torch.load(args.init_ckpt, map_location=device, weights_only=False)["model"])

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")
    scheduler = None
    if args.lr_milestones is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(args.lr_milestones), gamma=0.5)
    elif bool(args.lr_plateau):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=float(args.lr_plateau_factor),
            patience=int(args.lr_plateau_patience), min_lr=float(args.min_lr))

    def select_ade(val: dict) -> float:
        return val["top1"]["ade"] if str(args.select) == "top1" else val["oracle_best_of_k"]["ade"]

    best_val_ade = float("inf")
    best_epoch = -1
    patience_left = int(args.patience)
    history, steps, seen = [], 0, 0             # per epoch: the lr it trained at, val ADE, cumulative steps / agent-windows
    if args.select_include_init and not args.eval_only:   # epoch 0, the starting weights, is a candidate
        val = evaluate(model, val_loader, device)
        best_val_ade, best_epoch = select_ade(val), 0
        ckpt_path = Path(args.save_ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "epoch": 0, "val": val, "args": vars(args)}, ckpt_path)
        history.append({"epoch": 0, "lr": None, "val_ade": best_val_ade, "steps": 0, "agent_windows": 0})

    # EWTA schedule: geometric stages [K, K/2, ..., 2, 1] spread over the
    # first ewta_frac of epochs; pure WTA (w=1) afterwards / when frac=0.
    k = int(args.k)
    stages = []
    w = k
    while w > 1:
        stages.append(w)
        w //= 2
    stages.append(1)

    def ewta_w(epoch: int) -> int:
        anneal = int(round(float(args.ewta_frac) * int(args.epochs)))
        if anneal <= 0 or epoch > anneal:
            return 1
        i = min(len(stages) - 1, (epoch - 1) * len(stages) // anneal)
        return stages[i]

    def soft_mode_targets(pred_rel_k: torch.Tensor, b) -> torch.Tensor:
        """Detached per-agent mode distribution: softmax(-masked mode MSE / tau)."""
        with torch.no_grad():
            rv = (b.agent_mask.unsqueeze(-1) & rel_valid_mask(b.fut_mask))
            m = rv.unsqueeze(2).unsqueeze(3).float()
            diff = (pred_rel_k - b.pred_rel.unsqueeze(2)) * m
            denom = (rv.float().sum(dim=-1) * 2.0).clamp_min(1.0)  # (B,M)
            per_mode = (diff * diff).sum(dim=(3, 4)) / denom.unsqueeze(-1)  # (B,M,K)
            return torch.softmax(-per_mode / float(args.mode_tau), dim=-1)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses = []
        w_now = ewta_w(epoch)
        for batch in train_loader:
            b = batch.to(device)
            opt.zero_grad(set_to_none=True)
            parts = split_batch(b, int(args.max_batch_slots))
            batch_loss = 0.0
            for p in parts:
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    pred_rel_k, mode_logits = model(p.obs_rel, p.agent_mask)
                    reg_loss, best_mode = masked_variety_loss(pred_rel_k, p.pred_rel,
                                                              p.agent_mask, p.fut_mask,
                                                              top_w=w_now)
                    if len(parts) > 1:                                   # this part's share of the batch mean
                        reg_loss = reg_loss * (valid_steps(p) / valid_steps(b).clamp_min(1))
                    if float(args.mode_cls_weight) > 0.0 and bool(p.agent_mask.any()):
                        if str(args.mode_target) == "soft":
                            tgt = soft_mode_targets(pred_rel_k, p)
                            logp = torch.log_softmax(mode_logits[p.agent_mask], dim=-1)
                            ce = -(tgt[p.agent_mask] * logp).sum(dim=-1).mean()
                        else:
                            ce = F.cross_entropy(mode_logits[p.agent_mask],
                                                 best_mode[p.agent_mask])
                        if len(parts) > 1:
                            ce = ce * (p.agent_mask.sum() / b.agent_mask.sum())
                        loss = reg_loss + float(args.mode_cls_weight) * ce
                    else:
                        loss = reg_loss
                scaler.scale(loss).backward()
                batch_loss += float(loss.item())
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            train_losses.append(batch_loss)
            steps += 1
            seen += int(b.agent_mask.sum().item())

        val = evaluate(model, val_loader, device)
        val_ade = select_ade(val)
        history.append({"epoch": epoch, "lr": opt.param_groups[0]["lr"], "val_ade": val_ade, "steps": steps,
                        "agent_windows": seen})
        if args.lr_milestones is not None:
            scheduler.step()
        elif scheduler is not None:
            scheduler.step(val_ade)

        improved = val_ade < best_val_ade - 1e-6 if args.keep == "best" else epoch == int(args.epochs)
        if improved:
            best_val_ade = val_ade
            best_epoch = epoch
            patience_left = int(args.patience)
            ckpt_path = Path(args.save_ckpt)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val": val, "args": vars(args)}, ckpt_path)
        elif int(args.patience) > 0:
            patience_left -= 1

        log_every = max(1, int(args.log_every))
        if epoch == 1 or epoch % log_every == 0 or improved:
            print(f"epoch {epoch:3d} | lr={opt.param_groups[0]['lr']:.2e} w={w_now} "
                  f"| train_loss={float(np.mean(train_losses)):.6f} "
                  f"| val_bestK_ADE={val['oracle_best_of_k']['ade']:.4f} "
                  f"| val_top1_ADE={val['top1']['ade']:.4f}"
                  + ("  (best)" if improved else ""))

        if args.keep == "best" and int(args.patience) > 0 and patience_left <= 0:
            print(f"early stopping at epoch {epoch} (best_epoch={best_epoch})")
            break

    # Final: single best checkpoint, evaluated on val + test (oracle AND top-1).
    ckpt = torch.load(Path(args.save_ckpt), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_summary = evaluate(model, val_loader, device)
    test_summary = evaluate(model, test_loader, device)
    if args.eval_only:
        best_epoch = int(ckpt["epoch"])

    metrics = {
        "model": "KModeVarietyDecoder_v2",
        "data": str(args.data),
        "splits_file": str(args.splits_file),
        "init_ckpt": args.init_ckpt,
        "obs_len": int(obs_len),
        "pred_len": int(pred_len),
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "eval_only": bool(args.eval_only),
        "ckpt": str(args.save_ckpt),
        "history": history,
        "val": val_summary,
        "test": test_summary,
        "hyper": {k: v for k, v in vars(args).items()
                  if k not in {"data", "splits_file", "save_metrics_json", "save_ckpt", "save_pred_npz",
                               "eval_only", "init_ckpt"}},
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": "Benchmark v2: oracle and top-1 both from the single best-val checkpoint; "
                "masked per-timestep metrics; FDE at last valid step.",
    }
    out = Path(args.save_metrics_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"val  oracle ADE={val_summary['oracle_best_of_k']['ade']:.4f} "
          f"top1 ADE={val_summary['top1']['ade']:.4f}")
    print(f"test oracle ADE={test_summary['oracle_best_of_k']['ade']:.4f} "
          f"top1 ADE={test_summary['top1']['ade']:.4f}")
    print(f"saved metrics: {out}")
    if args.save_pred_npz:
        dump_predictions(model, {"val": val_loader, "test": test_loader}, device, args.save_pred_npz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
