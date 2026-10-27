"""Dump deployable predictions from trained ST-GAT / K-mode checkpoints.

Rebuilds each model from its committed checkpoint (state + saved args), runs
inference on the val and test splits of the canonical dataset, and writes a flat
conformal-ready NPZ per model:

    pred (N,60,2) · gt (N,60,2) · fut_mask (N,60) · agent_type (N,) · split (N,)
    · scene_id (N,)  (for scene-level conformal folds and bootstrap CIs)

one row per real agent (padding dropped). ST-GAT dumps its single output;
K-mode dumps its deployable learned top-1. These feed scripts/conformal_wrapper.py
so the neural models get the same split-conformal treatment as CV/MLP.

Usage:
    python scripts/dump_predictions.py --device cuda
    # other seeds (for the seed-pooled bootstrap), kept apart from the conformal inputs:
    python scripts/dump_predictions.py --models stgat transformer --seed 1 \
        --out-dir outputs/benchmark_v2/seed_dumps/seed1
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


def agent_scene_ids(ds: CanonicalSeqDataset) -> list:
    """Scene id per real agent, in the order an unshuffled loader yields them
    (sequences in seq_indices order, agents in order within each sequence)."""
    out = []
    for k in ds.seq_indices:
        s, e = ds.seq_start_end[int(k)]
        out += [str(ds.scene_id[int(k)])] * int(e - s)
    return out


@torch.no_grad()
def flatten_samples(loader: DataLoader, samples_fn, device: torch.device) -> list:
    """Per real agent (same order as flatten): samples (K,T,2) from samples_fn(b) -> (B,M,K,2,T)."""
    out = []
    for batch in loader:
        b = batch.to(device)
        k = samples_fn(b).permute(0, 1, 2, 4, 3).cpu().numpy()   # (B,M,K,T,2)
        am = b.agent_mask.cpu().numpy()
        out += [k[i, j] for i in range(am.shape[0]) for j in range(am.shape[1]) if am[i, j]]
    return out


def dump(name: str, predict_fn, device: torch.device, out_path: Path,
         save_samples: bool = False) -> None:
    P, G, M, C, S, Z, K = [], [], [], [], [], [], []
    samples_fn = getattr(predict_fn, "samples_fn", None) if save_samples else None
    for split in ("val", "test"):
        loader = make_loader(split)
        p, g, m, c, s = flatten(loader, predict_fn, device, split)
        z = agent_scene_ids(loader.dataset)
        assert len(z) == len(p), "scene-id order must match the flattened agents"
        P += p; G += g; M += m; C += c; S += s; Z += z
        if samples_fn is not None:
            K += flatten_samples(loader, samples_fn, device)
    extra = {"samples": np.asarray(K, dtype=np.float32)} if samples_fn is not None else {}
    np.savez_compressed(
        out_path, **extra,
        pred=np.asarray(P, dtype=np.float32), gt=np.asarray(G, dtype=np.float32),
        fut_mask=np.asarray(M, dtype=bool), agent_type=np.asarray(C, dtype=object),
        split=np.asarray(S, dtype=object), scene_id=np.asarray(Z, dtype=object),
        model=name)
    print(f"{name}: {len(P)} agents (val+test) -> {out_path}")


def load_stgat(device: torch.device, seed: int = 0, ckpt_dir: Path = OUT):
    ck = torch.load(Path(ckpt_dir) / f"stgat_ckpt_seed{seed}.pt", map_location=device, weights_only=False)
    a = ck["args"]
    model = m3.STGATModel(
        obs_len=20, pred_len=60, emb_dim=a["emb_dim"], hidden_dim=a["hidden_dim"],
        pool=a["pool"], attn_dim=a["attn_dim"], num_heads=a["num_heads"],
        gat_layers=a["gat_layers"], dropout=a["dropout"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return lambda b: abs_from_rel(b.obs_abs, model(b.obs_rel, b.agent_mask, False))


def load_mfp(device: torch.device, seed: int = 0, ckpt_dir: Path = OUT):
    # The top-1-selected checkpoint is the better deployable K-mode (test 4.15).
    # It predates the sprint-3 traj_enc module, so load non-strictly: the content
    # scorer is off, traj_enc is unused, and every real weight matches.
    ck = torch.load(Path(ckpt_dir) / f"mfp_top1sel_ckpt_seed{seed}.pt", map_location=device, weights_only=False)
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

    def samples(b):   # all K modes, absolute: (B,M,K,2,T)
        return m4.abs_from_rel_k(b.obs_abs, model(b.obs_rel, b.agent_mask)[0])
    predict.samples_fn = samples
    return predict


def load_transformer(device: torch.device, seed: int = 0, ckpt_dir: Path = OUT):
    ck = torch.load(Path(ckpt_dir) / f"transformer_ckpt_seed{seed}.pt", map_location=device, weights_only=False)
    a = ck["args"]
    model = m5.MapFreeTransformer(
        obs_len=20, pred_len=60, d_model=a["d_model"], n_heads=a["n_heads"],
        n_temporal_layers=a["n_temporal_layers"], n_social_layers=a["n_social_layers"],
        n_decoder_layers=a["n_decoder_layers"], ff_mult=a["ff_mult"],
        dropout=a["dropout"], heading_k=a["heading_k"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return lambda b: abs_from_rel(b.obs_abs, model(b.obs_rel, b.agent_mask))


def load_mftf(device: torch.device, seed: int = 0, ckpt_dir: Path = OUT):
    import model8_mf_transformer_train_eval as m8
    ck = torch.load(Path(ckpt_dir) / f"mftf_ckpt_seed{seed}.pt", map_location=device, weights_only=False)
    a = ck["args"]
    model = m8.MultiFutureTransformer(
        obs_len=20, pred_len=60, k=a["k"], d_model=a["d_model"], n_heads=a["n_heads"],
        n_temporal_layers=a["n_temporal_layers"], n_social_layers=a["n_social_layers"],
        n_decoder_layers=a["n_decoder_layers"], ff_mult=a["ff_mult"], dropout=a["dropout"],
        heading_k=a["heading_k"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    def predict(b):
        pred_rel_k, mode_logits = model(b.obs_rel, b.agent_mask)
        return m4.top1_abs(pred_rel_k, mode_logits, b.obs_abs)

    def samples(b):   # all K modes, absolute: (B,M,K,2,T)
        return m4.abs_from_rel_k(b.obs_abs, model(b.obs_rel, b.agent_mask)[0])
    predict.samples_fn = samples
    return predict


# key -> (display name, checkpoint loader, output filename). The display name is
# what conformal_wrapper.py keys each model by (via the NPZ `model` field).
REGISTRY = {
    "stgat": ("ST-GAT-inspired (MHA)", load_stgat, "preds_stgat.npz"),
    "mfp": ("K-mode decoder (top-1-sel)", load_mfp, "preds_mfp.npz"),
    "transformer": ("map-free transformer", load_transformer, "preds_transformer.npz"),
    "mftf": ("multi-future transformer (top-1)", load_mftf, "preds_mftf.npz"),
}


def main() -> int:
    global DATA, SPLITS
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", nargs="+", choices=sorted(REGISTRY), default=sorted(REGISTRY),
                    help="which model dumps to (re)generate; default all. Use a subset to "
                         "add one model without overwriting the committed dumps of others.")
    ap.add_argument("--seed", type=int, default=0, help="which seed's checkpoint to dump")
    ap.add_argument("--ckpt-dir", default=str(OUT), help="where the *_ckpt_seed<k>.pt files are")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--splits-file", default=SPLITS)
    ap.add_argument("--save-samples", action="store_true",
                    help="also store every mode of multi-output models (K-mode) as samples")
    ap.add_argument("--out-dir", default=str(OUT),
                    help="output directory; keep non-seed-0 dumps out of outputs/benchmark_v2 "
                         "itself, whose preds_*.npz are the conformal inputs")
    args = ap.parse_args()
    DATA, SPLITS = Path(args.data), args.splits_file
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    for key in args.models:
        name, loader, fname = REGISTRY[key]
        dump(name, loader(device, args.seed, args.ckpt_dir), device, out_dir / fname, args.save_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
