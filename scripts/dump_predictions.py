"""Dump deployable predictions from trained ST-GAT / K-mode checkpoints.

Rebuilds each model from its committed checkpoint (state + saved args), runs
inference on the val and test splits of the canonical dataset, and writes a flat
conformal-ready NPZ per model:

    pred (N,60,2) · gt (N,60,2) · fut_mask (N,60) · agent_type (N,) · split (N,)

one row per real agent (padding dropped). ST-GAT dumps its single output;
K-mode dumps its deployable learned top-1. These feed scripts/conformal_wrapper.py
so the neural models get the same split-conformal treatment as CV/MLP.

Usage:
    python scripts/dump_predictions.py --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import model3_stgat_train_eval as m3
import model4_mfp_train_eval as m4
import model5_transformer_train_eval as m5
from dd_data import CanonicalSeqDataset, collate_pad, abs_from_rel
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "benchmark_v2"
DATA = REPO / "outputs" / "canonical_world_obs20_pred60.npz"
SPLITS = "splits/scene_splits.json"


def make_loader(split: str) -> DataLoader:
    ds = CanonicalSeqDataset(str(DATA), split, SPLITS, rot_aug=False, seed=0)
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_pad)


@torch.no_grad()
def flatten(loader: DataLoader, predict_fn, device: torch.device, split: str):
    P, G, M, C, S = [], [], [], [], []
    for batch in loader:
        b = batch.to(device)
        pred_abs = predict_fn(b)                          # (B,M,2,T)
        pred = pred_abs.permute(0, 1, 3, 2).cpu().numpy()  # (B,M,T,2)
        gt = b.pred_abs.permute(0, 1, 3, 2).cpu().numpy()
        am = b.agent_mask.cpu().numpy()
        fm = b.fut_mask.cpu().numpy()
        at = np.asarray(b.agent_type, dtype=object)
        for i in range(am.shape[0]):
            for j in range(am.shape[1]):
                if am[i, j]:
                    P.append(pred[i, j]); G.append(gt[i, j]); M.append(fm[i, j])
                    C.append(str(at[i, j])); S.append(split)
    return P, G, M, C, S


def dump(name: str, predict_fn, device: torch.device, out_path: Path) -> None:
    P, G, M, C, S = [], [], [], [], []
    for split in ("val", "test"):
        p, g, m, c, s = flatten(make_loader(split), predict_fn, device, split)
        P += p; G += g; M += m; C += c; S += s
    np.savez_compressed(
        out_path,
        pred=np.asarray(P, dtype=np.float32), gt=np.asarray(G, dtype=np.float32),
        fut_mask=np.asarray(M, dtype=bool), agent_type=np.asarray(C, dtype=object),
        split=np.asarray(S, dtype=object), model=name)
    print(f"{name}: {len(P)} agents (val+test) -> {out_path}")


def load_stgat(device: torch.device):
    ck = torch.load(OUT / "stgat_ckpt_seed0.pt", map_location=device, weights_only=False)
    a = ck["args"]
    model = m3.STGATModel(
        obs_len=20, pred_len=60, emb_dim=a["emb_dim"], hidden_dim=a["hidden_dim"],
        pool=a["pool"], attn_dim=a["attn_dim"], num_heads=a["num_heads"],
        gat_layers=a["gat_layers"], dropout=a["dropout"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return lambda b: abs_from_rel(b.obs_abs, model(b.obs_rel, b.agent_mask, False))


def load_mfp(device: torch.device):
    # The top-1-selected checkpoint is the better deployable K-mode (test 4.15).
    # It predates the sprint-3 traj_enc module, so load non-strictly: the content
    # scorer is off, traj_enc is unused, and every real weight matches.
    ck = torch.load(OUT / "mfp_top1sel_ckpt_seed0.pt", map_location=device, weights_only=False)
    a = ck["args"]
    model = m4.MFPModel(
        obs_len=20, pred_len=60, k=a["k"], emb_dim=a["emb_dim"], hidden_dim=a["hidden_dim"],
        pool=a["pool"], attn_dim=a["attn_dim"], num_heads=a["num_heads"],
        gat_layers=a["gat_layers"], dropout=a["dropout"],
        content_scorer=a.get("content_scorer", False)).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert all(k.startswith("traj_enc") for k in missing), f"unexpected missing: {missing}"
    model.eval()

    def predict(b):
        pred_rel_k, mode_logits = model(b.obs_rel, b.agent_mask)
        return m4.top1_abs(pred_rel_k, mode_logits, b.obs_abs)
    return predict


def load_transformer(device: torch.device):
    ck = torch.load(OUT / "transformer_ckpt_seed0.pt", map_location=device, weights_only=False)
    a = ck["args"]
    model = m5.MapFreeTransformer(
        obs_len=20, pred_len=60, d_model=a["d_model"], n_heads=a["n_heads"],
        n_temporal_layers=a["n_temporal_layers"], n_social_layers=a["n_social_layers"],
        n_decoder_layers=a["n_decoder_layers"], ff_mult=a["ff_mult"],
        dropout=a["dropout"], heading_k=a["heading_k"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return lambda b: abs_from_rel(b.obs_abs, model(b.obs_rel, b.agent_mask))


# key -> (display name, checkpoint loader, output filename). The display name is
# what conformal_wrapper.py keys each model by (via the NPZ `model` field).
REGISTRY = {
    "stgat": ("ST-GAT-inspired (MHA)", load_stgat, "preds_stgat.npz"),
    "mfp": ("K-mode decoder (top-1-sel)", load_mfp, "preds_mfp.npz"),
    "transformer": ("map-free transformer", load_transformer, "preds_transformer.npz"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", nargs="+", choices=sorted(REGISTRY), default=sorted(REGISTRY),
                    help="which model dumps to (re)generate; default all. Use a subset to "
                         "add one model without overwriting the committed dumps of others.")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    for key in args.models:
        name, loader, fname = REGISTRY[key]
        dump(name, loader(device), device, OUT / fname)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
