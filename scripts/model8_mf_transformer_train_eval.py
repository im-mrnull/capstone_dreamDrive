"""Train/evaluate the multi-future map-free transformer on the canonical v2 dataset.

The map-free transformer (model5) with K futures per agent and a learned
probability for each: the same per-agent temporal encoder, agent self-attention
and query decoder, but the decoder's output feeds K regression heads (one per
future) and the agent's social summary feeds a mode-probability head. Trained
exactly like the K-mode decoder (model4): masked best-of-K (winner-takes-all)
variety loss + 0.1 x cross-entropy of the probability head to the best mode;
checkpoint selected on val top-1 ADE (model4's --select top1, its better
deployable setting). Same map-free input contract and per-agent heading
canonicalization as model5; every mode is rotated back to the world frame.

Outputs follow model4's interface -- forward() returns (pred_rel_k (B,M,K,2,T),
mode_logits (B,M,K)) and the JSON has "oracle_best_of_k" and "top1" -- so the
evaluation, dumps and closed-loop code treat it like the K-mode decoder.

Usage:
    python scripts/model8_mf_transformer_train_eval.py --device cuda --seed 0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import model4_mfp_train_eval as m4
from dd_data import CanonicalSeqDataset, collate_pad, masked_variety_loss
from model5_transformer_train_eval import HEADING_K_DEFAULT, MapFreeTransformer, set_seed


class MultiFutureTransformer(MapFreeTransformer):
    """model5's encoder/decoder with K output heads and a mode-probability head."""

    def __init__(self, obs_len: int, pred_len: int, k: int = 10, **kw) -> None:
        super().__init__(obs_len, pred_len, **kw)
        self.k = int(k)
        self.out = nn.Linear(self.d_model, 2 * self.k)          # K futures from one decode
        self.mode_head = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.GELU(),
                                       nn.Linear(self.d_model, self.k))

    def forward(self, obs_rel: torch.Tensor, agent_mask: torch.Tensor):
        B, M, _, T = obs_rel.shape
        assert T == self.obs_len, f"obs_len {T} != {self.obs_len}"
        R = self._canonical_rotation(obs_rel)                          # (B,M,2,2)
        obs_c = torch.einsum("bmij,bmjt->bmit", R, obs_rel)
        x = obs_c.permute(0, 1, 3, 2).reshape(B * M, T, 2)
        x = self.drop(self.in_emb(x) + self.obs_pe.unsqueeze(0))
        mem = self.temporal(x)
        h = mem.mean(dim=1).view(B, M, self.d_model)
        h = self.social(h, src_key_padding_mask=~agent_mask)          # (B,M,d)
        q = (self.query_emb.unsqueeze(0) + self.pred_pe.unsqueeze(0))
        q = q.expand(B * M, -1, -1) + h.reshape(B * M, 1, self.d_model)
        dec = self.decoder(self.drop(q), mem)                         # (B*M,P,d)
        pred_c = self.out(dec).view(B, M, self.pred_len, self.k, 2).permute(0, 1, 3, 4, 2)  # (B,M,K,2,P)
        pred_rel_k = torch.einsum("bmij,bmkit->bmkjt", R, pred_c)    # back to world
        mode_logits = self.mode_head(h)                               # (B,M,K)
        return pred_rel_k, mode_logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/canonical_world_obs20_pred60.npz")
    ap.add_argument("--splits-file", default="splits/scene_splits.json")
    ap.add_argument("--save-metrics-json", default="outputs/benchmark_v2/mftf_metrics.json")
    ap.add_argument("--save-ckpt", default="outputs/benchmark_v2/mftf_ckpt.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--mode-cls-weight", type=float, default=0.1)
    ap.add_argument("--select", choices=["oracle", "top1"], default="top1")
    ap.add_argument("--lr-plateau-factor", type=float, default=0.5)
    ap.add_argument("--lr-plateau-patience", type=int, default=10)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-temporal-layers", type=int, default=2)
    ap.add_argument("--n-social-layers", type=int, default=2)
    ap.add_argument("--n-decoder-layers", type=int, default=2)
    ap.add_argument("--ff-mult", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--heading-k", type=int, default=HEADING_K_DEFAULT)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    set_seed(int(args.seed))

    def make_loader(split: str, shuffle: bool) -> DataLoader:
        ds = CanonicalSeqDataset(args.data, split, args.splits_file, rot_aug=False, seed=int(args.seed))
        return DataLoader(ds, batch_size=int(args.batch_size), shuffle=shuffle, num_workers=0,
                          collate_fn=collate_pad)

    train_loader, val_loader, test_loader = make_loader("train", True), make_loader("val", False), make_loader("test", False)
    obs0, pred0, *_ = train_loader.dataset[0]
    model = MultiFutureTransformer(
        obs0.shape[-1], pred0.shape[-1], k=int(args.k), d_model=int(args.d_model), n_heads=int(args.n_heads),
        n_temporal_layers=int(args.n_temporal_layers), n_social_layers=int(args.n_social_layers),
        n_decoder_layers=int(args.n_decoder_layers), ff_mult=int(args.ff_mult),
        dropout=float(args.dropout), heading_k=int(args.heading_k)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp) and device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=float(args.lr_plateau_factor), patience=int(args.lr_plateau_patience),
        min_lr=float(args.min_lr))

    best, best_epoch = float("inf"), -1
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            b = batch.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                pred_rel_k, mode_logits = model(b.obs_rel, b.agent_mask)
                reg, best_mode = masked_variety_loss(pred_rel_k.float(), b.pred_rel, b.agent_mask, b.fut_mask)
                loss = reg
                if float(args.mode_cls_weight) > 0 and bool(b.agent_mask.any()):
                    loss = loss + float(args.mode_cls_weight) * F.cross_entropy(
                        mode_logits[b.agent_mask].float(), best_mode[b.agent_mask])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
        val = m4.evaluate(model, val_loader, device)
        sel = val["top1"]["ade"] if args.select == "top1" else val["oracle_best_of_k"]["ade"]
        scheduler.step(sel)
        if sel < best - 1e-6:
            best, best_epoch = sel, epoch
            Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val, "args": vars(args)},
                       args.save_ckpt)
        if epoch == 1 or epoch % 25 == 0 or best_epoch == epoch:
            print(f"epoch {epoch:3d} | loss {np.mean(losses):.4f} | val top1 {val['top1']['ade']:.4f} "
                  f"| val best-of-{args.k} {val['oracle_best_of_k']['ade']:.4f}"
                  + ("  (best)" if best_epoch == epoch else ""), flush=True)

    ck = torch.load(args.save_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    val, test = m4.evaluate(model, val_loader, device), m4.evaluate(model, test_loader, device)
    metrics = {"model": "MultiFutureTransformer_v2", "data": str(args.data), "splits_file": str(args.splits_file),
               "seed": int(args.seed), "best_epoch": best_epoch, "k": int(args.k), "val": val, "test": test,
               "hyper": {k: v for k, v in vars(args).items() if k not in {"data", "splits_file", "save_metrics_json", "save_ckpt"}},
               "when": datetime.now().isoformat(timespec="seconds"),
               "note": "Benchmark v2: multi-future map-free transformer (model5 backbone, K heads + mode-probability "
                       "head), best-of-K variety loss + 0.1 CE, selected on val top-1 ADE. top1 = most probable mode "
                       "(deployable); oracle_best_of_k = min over K after seeing the ground truth."}
    out = Path(args.save_metrics_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for split, s in (("val", val), ("test", test)):
        print(f"{split}: top1 ADE={s['top1']['ade']:.4f} | best-of-{args.k} ADE={s['oracle_best_of_k']['ade']:.4f}")
    print(f"saved metrics: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
