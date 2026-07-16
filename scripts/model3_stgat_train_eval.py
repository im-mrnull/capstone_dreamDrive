"""Train/evaluate the ST-GAT-inspired model on the canonical v2 dataset.

Benchmark v2 interface:
- data: outputs/canonical_world_obs20_pred60.npz (per-timestep fut_mask, agent_type)
- split: frozen splits/scene_splits.json (train/val/test); seeds affect init only
- losses/metrics: masked per valid future step; FDE at last valid step;
  sum-and-divide aggregation over the whole eval set; per-class breakdown
- model selection on val; final metrics reported for val AND test

Architecture (unchanged from v1): LSTM encoder per agent, agent self-attention
(simple QKV pool or stacked MHA blocks), LSTMCell decoder emitting per-step
displacements. This is ST-GAT-*inspired*, not the published ST-GAT.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dd_data import (
    Batch,
    CanonicalSeqDataset,
    ErrorAccumulator,
    abs_from_rel,
    collate_pad,
    masked_reg_loss,
    per_agent_errors,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class AgentSelfAttentionLayer(nn.Module):
    """Multi-head self-attention over agents: MHA + residual + LayerNorm + FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
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
        key_padding_mask = ~agent_mask
        attn_out, _ = self.mha(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.ln1(h + self.drop(attn_out))
        h2 = self.ff(h)
        h = self.ln2(h + self.drop(h2))
        return h


class GraphAttentionPool(nn.Module):
    """Single-step attention pooling (simple variant)."""

    def __init__(self, hidden_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(hidden_dim, attn_dim)
        self.k = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(hidden_dim, attn_dim)
        self.out = nn.Linear(hidden_dim + attn_dim, hidden_dim)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, M, _ = h.shape
        q, k, v = self.q(h), self.k(h), self.v(h)
        scale = 1.0 / math.sqrt(k.shape[-1])
        scores = torch.matmul(q, k.transpose(1, 2)) * scale
        key_mask = mask.unsqueeze(1).expand(B, M, M)
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        query_mask = mask.unsqueeze(2).expand(B, M, M)
        scores = scores.masked_fill(~query_mask, 0.0)
        attn = torch.softmax(scores, dim=-1) * query_mask.float()
        ctx = torch.matmul(attn, v)
        return torch.tanh(self.out(torch.cat([h, ctx], dim=-1)))


class STGATModel(nn.Module):
    def __init__(
        self,
        obs_len: int,
        pred_len: int,
        emb_dim: int = 32,
        hidden_dim: int = 128,
        pool: str = "simple",
        attn_dim: int = 64,
        num_heads: int = 4,
        gat_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len

        self.emb = nn.Linear(2, emb_dim)
        self.encoder = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim,
                               num_layers=1, batch_first=True)
        self.pool_type = str(pool)
        if self.pool_type not in {"simple", "mha"}:
            raise ValueError("pool must be one of: simple, mha")

        self.simple_pool = GraphAttentionPool(hidden_dim=hidden_dim, attn_dim=attn_dim)
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.mha_layers = nn.ModuleList(
            [AgentSelfAttentionLayer(hidden_dim, num_heads, dropout) for _ in range(gat_layers)]
        )

        self.dec_in = nn.Linear(2, emb_dim)
        self.decoder_cell = nn.LSTMCell(input_size=emb_dim, hidden_size=hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        obs_rel: torch.Tensor,
        agent_mask: torch.Tensor,
        teacher_forcing: bool,
        gt_pred_rel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, M, _, obs_len = obs_rel.shape
        assert obs_len == self.obs_len

        x = obs_rel.permute(0, 1, 3, 2).contiguous().view(B * M, obs_len, 2)
        x = self.drop(torch.relu(self.emb(x)))
        _, (h_n, _) = self.encoder(x)
        h_last = h_n[-1].view(B, M, -1)

        if self.pool_type == "simple":
            h0 = self.simple_pool(h_last, agent_mask)
        else:
            h0 = h_last
            for layer in self.mha_layers:
                h0 = layer(h0, agent_mask)

        h = h0.reshape(B * M, -1)
        c = torch.zeros_like(h)
        preds: List[torch.Tensor] = []
        prev_rel = obs_rel[:, :, :, -1].contiguous().view(B * M, 2)

        for t in range(self.pred_len):
            inp = self.drop(torch.relu(self.dec_in(prev_rel)))
            h, c = self.decoder_cell(inp, (h, c))
            step = self.out(self.drop(h))
            preds.append(step.view(B, M, 2))
            if teacher_forcing and gt_pred_rel is not None:
                prev_rel = gt_pred_rel[:, :, :, t].contiguous().view(B * M, 2)
            else:
                prev_rel = step

        return torch.stack(preds, dim=-1)  # (B, M, 2, pred_len)


def teacher_forcing_ratio(epoch: int, tf_start: float, tf_end: float, tf_decay_epochs: int) -> float:
    if tf_decay_epochs <= 0:
        return float(tf_end)
    if epoch <= 1:
        return float(tf_start)
    t = min(1.0, max(0.0, (epoch - 1) / float(tf_decay_epochs)))
    return float(tf_start + (tf_end - tf_start) * t)


@torch.no_grad()
def evaluate(model: STGATModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    acc = ErrorAccumulator()
    for batch in loader:
        b = batch.to(device)
        pred_rel_hat = model(b.obs_rel, b.agent_mask, teacher_forcing=False)
        pred_abs = abs_from_rel(b.obs_abs, pred_rel_hat)
        ade, fde, path = per_agent_errors(pred_abs, b.pred_abs, b.agent_mask, b.fut_mask)
        acc.add(ade, fde, path, b.agent_type, b.agent_mask)
    return acc.summary()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/canonical_world_obs20_pred60.npz")
    ap.add_argument("--splits-file", default="splits/scene_splits.json")
    ap.add_argument("--save-metrics-json", default="outputs/benchmark_v2/stgat_metrics.json")
    ap.add_argument("--save-ckpt", default="outputs/benchmark_v2/stgat_ckpt.pt")
    ap.add_argument("--save-pred-npz", default=None,
                    help="optional: save test-split predictions NPZ")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--loss", choices=["mse", "huber"], default="mse")
    ap.add_argument("--patience", type=int, default=0, help="early stopping on val ADE (0 disables)")
    ap.add_argument("--lr-plateau", action="store_true", default=True)
    ap.add_argument("--no-lr-plateau", dest="lr_plateau", action="store_false")
    ap.add_argument("--lr-plateau-factor", type=float, default=0.5)
    ap.add_argument("--lr-plateau-patience", type=int, default=10)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--rot-aug", action="store_true", default=False,
                    help="random scene rotation augmentation at train time")

    ap.add_argument("--tf-start", type=float, default=1.0)
    ap.add_argument("--tf-end", type=float, default=1.0)
    ap.add_argument("--tf-decay-epochs", type=int, default=1)

    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--pool", choices=["simple", "mha"], default="mha")
    ap.add_argument("--attn-dim", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--gat-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    set_seed(int(args.seed))

    def make_loader(split: str, shuffle: bool, rot_aug: bool = False) -> DataLoader:
        ds = CanonicalSeqDataset(args.data, split, args.splits_file,
                                 rot_aug=rot_aug, seed=int(args.seed))
        return DataLoader(ds, batch_size=int(args.batch_size), shuffle=shuffle,
                          num_workers=0, collate_fn=collate_pad)

    train_loader = make_loader("train", shuffle=True, rot_aug=bool(args.rot_aug))
    val_loader = make_loader("val", shuffle=False)
    test_loader = make_loader("test", shuffle=False)

    obs0, pred0, *_ = train_loader.dataset[0]
    obs_len, pred_len = obs0.shape[-1], pred0.shape[-1]

    model = STGATModel(
        obs_len=obs_len, pred_len=pred_len,
        emb_dim=int(args.emb_dim), hidden_dim=int(args.hidden_dim),
        pool=str(args.pool), attn_dim=int(args.attn_dim),
        num_heads=int(args.num_heads), gat_layers=int(args.gat_layers),
        dropout=float(args.dropout),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")
    scheduler = None
    if bool(args.lr_plateau):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=float(args.lr_plateau_factor),
            patience=int(args.lr_plateau_patience), min_lr=float(args.min_lr))

    best_val_ade = float("inf")
    best_epoch = -1
    patience_left = int(args.patience)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses = []
        tf_ratio = teacher_forcing_ratio(epoch, float(args.tf_start), float(args.tf_end),
                                         int(args.tf_decay_epochs))
        for batch in train_loader:
            b = batch.to(device)
            use_tf = (torch.rand(()) < tf_ratio).item()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred_rel_hat = model(b.obs_rel, b.agent_mask, teacher_forcing=bool(use_tf),
                                     gt_pred_rel=b.pred_rel)
                loss = masked_reg_loss(pred_rel_hat, b.pred_rel, b.agent_mask, b.fut_mask,
                                       kind=str(args.loss))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            train_losses.append(float(loss.item()))

        val = evaluate(model, val_loader, device)
        val_ade, val_fde = val["ade"], val["fde"]
        if scheduler is not None:
            scheduler.step(val_ade)

        improved = val_ade < best_val_ade - 1e-6
        if improved:
            best_val_ade = val_ade
            best_epoch = epoch
            patience_left = int(args.patience)
            ckpt_path = Path(args.save_ckpt)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_ade": val_ade, "val_fde": val_fde, "args": vars(args)}, ckpt_path)
        elif int(args.patience) > 0:
            patience_left -= 1

        if epoch == 1 or epoch % 5 == 0 or improved:
            print(f"epoch {epoch:3d} | lr={opt.param_groups[0]['lr']:.2e} | tf={tf_ratio:.2f} "
                  f"| train_loss={float(np.mean(train_losses)):.6f} "
                  f"| val_ADE={val_ade:.4f} | val_FDE={val_fde:.4f}"
                  + ("  (best)" if improved else ""))

        if int(args.patience) > 0 and patience_left <= 0:
            print(f"early stopping at epoch {epoch} (best_epoch={best_epoch})")
            break

    # Final: reload best, evaluate val + test.
    ckpt = torch.load(Path(args.save_ckpt), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_summary = evaluate(model, val_loader, device)
    test_summary = evaluate(model, test_loader, device)

    metrics = {
        "model": "STGAT_inspired_v2",
        "data": str(args.data),
        "splits_file": str(args.splits_file),
        "obs_len": int(obs_len),
        "pred_len": int(pred_len),
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "val": val_summary,
        "test": test_summary,
        "hyper": {k: v for k, v in vars(args).items()
                  if k not in {"data", "splits_file", "save_metrics_json", "save_ckpt", "save_pred_npz"}},
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": "Benchmark v2: canonical dataset, frozen splits, masked per-timestep "
                "metrics, FDE at last valid step, model selection on val only.",
    }
    out = Path(args.save_metrics_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"val  ADE={val_summary['ade']:.4f} FDE={val_summary['fde']:.4f} (n={val_summary['n_agents']})")
    print(f"test ADE={test_summary['ade']:.4f} FDE={test_summary['fde']:.4f} (n={test_summary['n_agents']})")
    print(f"saved metrics: {out}")

    if args.save_pred_npz:
        model.eval()
        preds, gts, masks, fmasks, types = [], [], [], [], []
        global_max_m = max(int(b.agent_mask.shape[1]) for b in test_loader)
        with torch.no_grad():
            for batch in test_loader:
                b = batch.to(device)
                pred_abs = abs_from_rel(b.obs_abs, model(b.obs_rel, b.agent_mask, False))
                B, m, _, T = pred_abs.shape
                pad = lambda x, fill=0: torch.cat(
                    [x, torch.full((B, global_max_m - m, *x.shape[2:]), fill,
                                   dtype=x.dtype, device=x.device)], dim=1) if m < global_max_m else x
                preds.append(pad(pred_abs).cpu())
                gts.append(pad(b.pred_abs).cpu())
                masks.append(pad(b.agent_mask).cpu())
                fmasks.append(pad(b.fut_mask).cpu())
                at = np.full((B, global_max_m), "", dtype=object)
                at[:, :m] = b.agent_type
                types.append(at)
        np.savez_compressed(
            args.save_pred_npz,
            pred_abs=torch.cat(preds).numpy().astype(np.float32),
            gt_abs=torch.cat(gts).numpy().astype(np.float32),
            mask=torch.cat(masks).numpy().astype(bool),
            fut_mask=torch.cat(fmasks).numpy().astype(bool),
            agent_type=np.concatenate(types),
        )
        print(f"saved test predictions: {args.save_pred_npz}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
