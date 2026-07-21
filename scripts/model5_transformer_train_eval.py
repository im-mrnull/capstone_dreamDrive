"""Train/evaluate a map-free trajectory Transformer on the canonical v2 dataset.

Deterministic, single-output — the Transformer counterpart to the deployable
rows (CV / MLP / ST-GAT-inspired / Trajectron++ most-likely). This is a *plain
map-free Transformer, NOT a QCNet port*: IDD-3D is lane-free, so there is no map
branch and no map-anchored reference frame. Honest name: "map-free transformer".

Map-free INPUT contract (enforced by tests/test_map_free_transformer.py):
- inputs are agent-relative displacements only (obs_rel); translation-invariant
  by construction, and rotation-equivariant to each agent's last-observed pose
  via per-agent heading canonicalization (each agent is rotated so its last
  heading points +x, predictions are rotated back to world);
- no map/lane channel, no agent-class one-hot as an input feature;
- obs/pred horizons and fps are constructor arguments, never hard-coded, so the
  identical architecture transfers to nuScenes/WOMD later (Chaos-Gap backbone).

Architecture: per-agent temporal Transformer encoder over the observed
displacement sequence, agent self-attention (social) across agents, and a
parallel query-based Transformer decoder that cross-attends to each agent's
temporal memory to emit pred_len future displacements in one shot.

Benchmark v2 interface (matches model3): canonical NPZ + frozen splits; seeds
affect init only; masked per-timestep losses/metrics; FDE at last valid step;
model selection on val ADE; per-class breakdown; JSON schema of the deployable
rows ({"val": {ade, fde, ...}, "test": {...}}).
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
    CanonicalSeqDataset,
    ErrorAccumulator,
    abs_from_rel,
    collate_pad,
    masked_reg_loss,
    per_agent_errors,
)

# Heading canonicalization (mirrors scripts/9_make_agent_windows_from_canonical.py
# so the model's internal frame matches the per-agent windows used elsewhere).
HEADING_K_DEFAULT = 5
STATIC_SPEED = 0.01  # m per frame (0.1 m/s at 10 fps): below this keep identity


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def sinusoidal_pe(length: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding, (length, d_model)."""
    pe = torch.zeros(length, d_model)
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class MapFreeTransformer(nn.Module):
    def __init__(
        self,
        obs_len: int,
        pred_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_temporal_layers: int = 2,
        n_social_layers: int = 2,
        n_decoder_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
        heading_k: int = HEADING_K_DEFAULT,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.d_model = int(d_model)
        self.heading_k = int(heading_k)
        self.input_dim = 2  # (x, y) displacement only — no map, no class one-hot

        self.in_emb = nn.Linear(2, d_model)
        self.register_buffer("obs_pe", sinusoidal_pe(obs_len, d_model), persistent=False)
        self.register_buffer("pred_pe", sinusoidal_pe(pred_len, d_model), persistent=False)

        def enc(n_layers: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ff_mult,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

        self.temporal = enc(n_temporal_layers)     # per-agent, over time
        self.social = enc(n_social_layers)         # across agents

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ff_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, n_decoder_layers)
        self.query_emb = nn.Parameter(torch.randn(pred_len, d_model) * 0.02)
        self.out = nn.Linear(d_model, 2)
        self.drop = nn.Dropout(dropout)

    def _canonical_rotation(self, obs_rel: torch.Tensor) -> torch.Tensor:
        """(B,M,2,T) displacements -> (B,M,2,2) rotation mapping each agent's
        last-heading to +x. Near-static agents keep the identity rotation."""
        k = min(self.heading_k, obs_rel.shape[-1])
        heading = obs_rel[..., -k:].sum(dim=-1)                 # (B,M,2)
        norm = torch.linalg.vector_norm(heading, dim=-1)       # (B,M)
        moving = norm > (STATIC_SPEED * k)
        safe = norm.clamp_min(1e-8)
        cos = torch.where(moving, heading[..., 0] / safe, torch.ones_like(norm))
        sin = torch.where(moving, heading[..., 1] / safe, torch.zeros_like(norm))
        # Rotate by -theta so heading -> +x: [[cos, sin], [-sin, cos]]
        row0 = torch.stack([cos, sin], dim=-1)
        row1 = torch.stack([-sin, cos], dim=-1)
        return torch.stack([row0, row1], dim=-2)               # (B,M,2,2)

    def forward(self, obs_rel: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        B, M, _, T = obs_rel.shape
        assert T == self.obs_len, f"obs_len {T} != {self.obs_len}"

        R = self._canonical_rotation(obs_rel)                          # (B,M,2,2)
        obs_c = torch.einsum("bmij,bmjt->bmit", R, obs_rel)            # canonical frame

        # Per-agent temporal encoder over the observed displacement sequence.
        x = obs_c.permute(0, 1, 3, 2).reshape(B * M, T, 2)            # (B*M,T,2)
        x = self.drop(self.in_emb(x) + self.obs_pe.unsqueeze(0))
        mem = self.temporal(x)                                        # (B*M,T,d)
        h = mem.mean(dim=1).view(B, M, self.d_model)                 # per-agent summary

        # Social self-attention across agents (padding masked out as keys).
        h = self.social(h, src_key_padding_mask=~agent_mask)         # (B,M,d)

        # Parallel query decoder: pred_len learned queries per agent, conditioned
        # on the social summary, cross-attending to that agent's temporal memory.
        q = (self.query_emb.unsqueeze(0) + self.pred_pe.unsqueeze(0))  # (1,P,d)
        q = q.expand(B * M, -1, -1) + h.reshape(B * M, 1, self.d_model)
        dec = self.decoder(self.drop(q), mem)                        # (B*M,P,d)
        pred_c = self.out(dec).view(B, M, self.pred_len, 2).permute(0, 1, 3, 2)

        # Rotate predictions back from the per-agent canonical frame to world.
        pred_rel = torch.einsum("bmij,bmit->bmjt", R, pred_c)        # R^T @ pred_c
        return pred_rel


@torch.no_grad()
def evaluate(model: MapFreeTransformer, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    acc = ErrorAccumulator()
    for batch in loader:
        b = batch.to(device)
        pred_rel = model(b.obs_rel, b.agent_mask)
        pred_abs = abs_from_rel(b.obs_abs, pred_rel)
        ade, fde, path = per_agent_errors(pred_abs, b.pred_abs, b.agent_mask, b.fut_mask)
        acc.add(ade, fde, path, b.agent_type, b.agent_mask)
    return acc.summary()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/canonical_world_obs20_pred60.npz")
    ap.add_argument("--splits-file", default="splits/scene_splits.json")
    ap.add_argument("--save-metrics-json", default="outputs/benchmark_v2/transformer_metrics.json")
    ap.add_argument("--save-ckpt", default="outputs/benchmark_v2/transformer_ckpt.pt")
    ap.add_argument("--save-pred-npz", default=None,
                    help="optional: save val+test predictions NPZ (conformal dump)")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["mse", "huber"], default="huber")
    ap.add_argument("--patience", type=int, default=0, help="early stopping on val ADE (0 disables)")
    ap.add_argument("--lr-plateau", action="store_true", default=True)
    ap.add_argument("--no-lr-plateau", dest="lr_plateau", action="store_false")
    ap.add_argument("--lr-plateau-factor", type=float, default=0.5)
    ap.add_argument("--lr-plateau-patience", type=int, default=10)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--rot-aug", action="store_true", default=False,
                    help="random scene rotation augmentation (a no-op under the "
                         "model's own per-agent canonicalization; kept for parity)")

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

    model = MapFreeTransformer(
        obs_len=obs_len, pred_len=pred_len,
        d_model=int(args.d_model), n_heads=int(args.n_heads),
        n_temporal_layers=int(args.n_temporal_layers),
        n_social_layers=int(args.n_social_layers),
        n_decoder_layers=int(args.n_decoder_layers),
        ff_mult=int(args.ff_mult), dropout=float(args.dropout),
        heading_k=int(args.heading_k),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr),
                            weight_decay=float(args.weight_decay))
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
        for batch in train_loader:
            b = batch.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred_rel = model(b.obs_rel, b.agent_mask)
                loss = masked_reg_loss(pred_rel, b.pred_rel, b.agent_mask, b.fut_mask,
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
            print(f"epoch {epoch:3d} | lr={opt.param_groups[0]['lr']:.2e} "
                  f"| train_loss={float(np.mean(train_losses)):.6f} "
                  f"| val_ADE={val_ade:.4f} | val_FDE={val_fde:.4f}"
                  + ("  (best)" if improved else ""))

        if int(args.patience) > 0 and patience_left <= 0:
            print(f"early stopping at epoch {epoch} (best_epoch={best_epoch})")
            break

    # Final: reload best checkpoint, evaluate val + test.
    ckpt = torch.load(Path(args.save_ckpt), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_summary = evaluate(model, val_loader, device)
    test_summary = evaluate(model, test_loader, device)

    metrics = {
        "model": "MapFreeTransformer_v2",
        "data": str(args.data),
        "splits_file": str(args.splits_file),
        "obs_len": int(obs_len),
        "pred_len": int(pred_len),
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "val": val_summary,
        "test": test_summary,
        "hyper": {k: v for k, v in vars(args).items()
                  if k not in {"data", "splits_file", "save_metrics_json",
                               "save_ckpt", "save_pred_npz"}},
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": "Benchmark v2: plain map-free Transformer (deterministic single "
                "output), per-agent heading canonicalization, masked per-timestep "
                "metrics, FDE at last valid step, model selection on val only.",
    }
    out = Path(args.save_metrics_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"val  ADE={val_summary['ade']:.4f} FDE={val_summary['fde']:.4f} (n={val_summary['n_agents']})")
    print(f"test ADE={test_summary['ade']:.4f} FDE={test_summary['fde']:.4f} (n={test_summary['n_agents']})")
    print(f"saved metrics: {out}")

    if args.save_pred_npz:
        _dump_predictions(model, val_loader, test_loader, device, args.save_pred_npz)

    return 0


@torch.no_grad()
def _dump_predictions(model, val_loader, test_loader, device, out_path) -> None:
    """Flat per-agent val+test predictions for the conformal wrapper (one row per
    real agent): pred/gt (N,60,2), fut_mask (N,60), agent_type (N,), split (N,)."""
    model.eval()
    P, G, M, C, S = [], [], [], [], []
    for split, loader in (("val", val_loader), ("test", test_loader)):
        for batch in loader:
            b = batch.to(device)
            pred_abs = abs_from_rel(b.obs_abs, model(b.obs_rel, b.agent_mask))
            pred = pred_abs.permute(0, 1, 3, 2).cpu().numpy()   # (B,M,T,2)
            gt = b.pred_abs.permute(0, 1, 3, 2).cpu().numpy()
            am = b.agent_mask.cpu().numpy()
            fm = b.fut_mask.cpu().numpy()
            at = np.asarray(b.agent_type, dtype=object)
            for i in range(am.shape[0]):
                for j in range(am.shape[1]):
                    if am[i, j]:
                        P.append(pred[i, j]); G.append(gt[i, j]); M.append(fm[i, j])
                        C.append(str(at[i, j])); S.append(split)
    np.savez_compressed(
        out_path,
        pred=np.asarray(P, dtype=np.float32), gt=np.asarray(G, dtype=np.float32),
        fut_mask=np.asarray(M, dtype=bool), agent_type=np.asarray(C, dtype=object),
        split=np.asarray(S, dtype=object), model="map-free transformer")
    print(f"saved test+val predictions: {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
