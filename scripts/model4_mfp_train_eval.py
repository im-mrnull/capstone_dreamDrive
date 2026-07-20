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
from torch.utils.data import DataLoader

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/canonical_world_obs20_pred60.npz")
    ap.add_argument("--splits-file", default="splits/scene_splits.json")
    ap.add_argument("--save-metrics-json", default="outputs/benchmark_v2/mfp_metrics.json")
    ap.add_argument("--save-ckpt", default="outputs/benchmark_v2/mfp_ckpt.pt")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--log-every", type=int, default=25)

    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--patience", type=int, default=0)
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

    model = MFPModel(
        obs_len=obs_len, pred_len=pred_len, k=int(args.k),
        emb_dim=int(args.emb_dim), hidden_dim=int(args.hidden_dim),
        pool=str(args.pool), attn_dim=int(args.attn_dim),
        num_heads=int(args.num_heads), gat_layers=int(args.gat_layers),
        dropout=float(args.dropout), content_scorer=bool(args.content_scorer),
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
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred_rel_k, mode_logits = model(b.obs_rel, b.agent_mask)
                reg_loss, best_mode = masked_variety_loss(pred_rel_k, b.pred_rel,
                                                          b.agent_mask, b.fut_mask,
                                                          top_w=w_now)
                if float(args.mode_cls_weight) > 0.0 and bool(b.agent_mask.any()):
                    if str(args.mode_target) == "soft":
                        tgt = soft_mode_targets(pred_rel_k, b)
                        logp = torch.log_softmax(mode_logits[b.agent_mask], dim=-1)
                        ce = -(tgt[b.agent_mask] * logp).sum(dim=-1).mean()
                    else:
                        ce = F.cross_entropy(mode_logits[b.agent_mask],
                                             best_mode[b.agent_mask])
                    loss = reg_loss + float(args.mode_cls_weight) * ce
                else:
                    loss = reg_loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            train_losses.append(float(loss.item()))

        val = evaluate(model, val_loader, device)
        val_ade = (val["top1"]["ade"] if str(args.select) == "top1"
                   else val["oracle_best_of_k"]["ade"])
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

        if int(args.patience) > 0 and patience_left <= 0:
            print(f"early stopping at epoch {epoch} (best_epoch={best_epoch})")
            break

    # Final: single best checkpoint, evaluated on val + test (oracle AND top-1).
    ckpt = torch.load(Path(args.save_ckpt), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_summary = evaluate(model, val_loader, device)
    test_summary = evaluate(model, test_loader, device)

    metrics = {
        "model": "KModeVarietyDecoder_v2",
        "data": str(args.data),
        "splits_file": str(args.splits_file),
        "obs_len": int(obs_len),
        "pred_len": int(pred_len),
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "val": val_summary,
        "test": test_summary,
        "hyper": {k: v for k, v in vars(args).items()
                  if k not in {"data", "splits_file", "save_metrics_json", "save_ckpt"}},
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
