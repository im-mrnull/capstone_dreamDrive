"""Train/evaluate upstream AgentFormer (base CVAE) on the canonical v2 dataset.

AgentFormer (Yuan et al., ICCV 2021; https://github.com/Khrylx/AgentFormer),
patched for modern Python/PyTorch by scripts/patch_agentformer.py and driven here
through its model interface only — fed canonical IDD-3D sequences directly by
scripts/dd_agentformer.py so the eval population is identical to every other
benchmark-v2 model (the shared eval-population contract). This is the *base CVAE*
scope (no DLow diversity head): the deployable most-likely output is the z=0
prior-mean rollout, and the oracle best-of-K row is the min over K prior samples.

Map-free by construction: no map/lane channel, no per-agent heading input
(input_type=['scene_norm','vel'], use_map=False) — IDD-3D is lane-free and this
repo's parsed data has no yaw.

Benchmark v2 interface (matches Trajectron++/model5): canonical NPZ + frozen
splits; seeds affect init/shuffle only; masked per-timestep ADE/FDE (dd_data,
FDE at last valid step); model selection on val most-likely ADE; multimodal JSON
schema {"val": {"most_likely": {...}, "best_of_k": {...}}, "test": {...}} that
run_benchmark_v2.py aggregates.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from dd_data import (
    CanonicalSeqDataset,
    ErrorAccumulator,
    per_agent_errors,
    per_agent_errors_bestofk,
)
from dd_agentformer import (
    add_af_to_path,
    build_map_free_cfg,
    keyframe_indices,
    predict_seq,
    seq_to_af_data,
)

add_af_to_path()
from model.model_lib import model_dict  # noqa: E402  (needs AF on sys.path first)


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def _seq_batch1(obs, pred, fm, at):
    """One canonical sequence -> the (batch-of-1) tensors dd_data's metrics want."""
    M = obs.shape[0]
    gt = torch.from_numpy(pred).float().unsqueeze(0)              # (1,M,2,T)
    am = torch.ones(1, M, dtype=torch.bool)
    fmt = torch.from_numpy(fm.astype(bool)).unsqueeze(0)         # (1,M,T)
    at2 = np.asarray(at, dtype=object).reshape(1, M)
    return gt, am, fmt, at2


def scene_sizes(ds) -> np.ndarray:
    """Agent count per kept sequence, without loading the trajectory arrays."""
    sse = ds.seq_start_end[ds.seq_indices]
    return (sse[:, 1] - sse[:, 0]).astype(int)


def pick_indices(ds, limit=None, smallest=False, max_agents=None):
    """Sequence indices to iterate: all in natural order by default; when
    smallest/max_agents are set (smoke), the smallest scenes first (cheap AR
    decode) capped at max_agents; truncated to `limit`."""
    idx = np.arange(len(ds))
    if smallest or max_agents is not None:
        sizes = scene_sizes(ds)
        idx = np.argsort(sizes, kind="stable")
        if max_agents is not None:
            idx = idx[sizes[idx] <= max_agents]
    if limit is not None:
        idx = idx[:limit]
    return [int(i) for i in idx]


@torch.no_grad()
def evaluate(model, ds, cfg, eval_k: int, device, indices=None) -> dict:
    """Most-likely (z=0) and oracle best-of-eval_k over the given sequence
    indices (all sequences when indices is None), masked metrics."""
    model.eval()
    acc_ml, acc_bok = ErrorAccumulator(), ErrorAccumulator()
    idx_list = range(len(ds)) if indices is None else indices
    for i in idx_list:
        obs, pred, _, _, fm, at = ds[int(i)]
        data = seq_to_af_data(obs, pred, fm, cfg.traj_scale, stride=cfg.get("stride", 1))
        ml, samp = predict_seq(model, data, nz=cfg.nz, sample_k=eval_k,
                               traj_scale=cfg.traj_scale, stride=cfg.get("stride", 1))
        gt, am, fmt, at2 = _seq_batch1(obs, pred, fm, at)
        gt, am, fmt = gt.to(device), am.to(device), fmt.to(device)

        pred_ml = ml.permute(0, 2, 1).unsqueeze(0)                       # (1,M,2,T)
        ade, fde, path = per_agent_errors(pred_ml, gt, am, fmt)
        acc_ml.add(ade, fde, path, at2, am)

        pred_k = samp.permute(0, 1, 3, 2).unsqueeze(0)                   # (1,M,K,2,T)
        ade_k, fde_k, path_k = per_agent_errors_bestofk(pred_k, gt, am, fmt)
        acc_bok.add(ade_k, fde_k, path_k, at2, am)
    return {"most_likely": acc_ml.summary(), "best_of_k": acc_bok.summary()}


@torch.no_grad()
def dump_most_likely(model, splits, cfg, device, out_path: str, sample_k: int = 0) -> None:
    """Flat per-agent val+test most-likely predictions for the conformal wrapper
    (one row per real agent): pred/gt (N,60,2), fut_mask (N,60), agent_type, split.
    sample_k > 0 also stores that many prior samples per agent as samples (N,K,60,2)."""
    model.eval()
    P, G, M, C, S, Z, K = [], [], [], [], [], [], []
    for split, ds in splits.items():
        for i in range(len(ds)):
            obs, pred, _, _, fm, at = ds[i]
            data = seq_to_af_data(obs, pred, fm, cfg.traj_scale, stride=cfg.get("stride", 1))
            ml, samp = predict_seq(model, data, nz=cfg.nz, sample_k=max(sample_k, 1),
                                   traj_scale=cfg.traj_scale, stride=cfg.get("stride", 1))
            ml, samp = ml.cpu().numpy(), samp.cpu().numpy()              # (M,T,2), (M,K,T,2)
            gt = np.transpose(pred, (0, 2, 1))                           # (M,T,2)
            scene = str(ds.scene_id[int(ds.seq_indices[i])])
            for a in range(obs.shape[0]):
                P.append(ml[a]); G.append(gt[a]); M.append(fm[a].astype(bool)); K.append(samp[a])
                C.append(str(at[a])); S.append(split); Z.append(scene)
    extra = {"samples": np.asarray(K, dtype=np.float32)} if sample_k > 0 else {}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, **extra,
        pred=np.asarray(P, dtype=np.float32), gt=np.asarray(G, dtype=np.float32),
        fut_mask=np.asarray(M, dtype=bool), agent_type=np.asarray(C, dtype=object),
        split=np.asarray(S, dtype=object), scene_id=np.asarray(Z, dtype=object),
        model="AgentFormer (most-likely)")
    print(f"saved most-likely predictions: {out_path}")


def build_model(cfg, device):
    model = model_dict["agentformer"](cfg)
    model.set_device(device)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/canonical_world_obs20_pred60.npz")
    ap.add_argument("--splits-file", default="splits/scene_splits.json")
    ap.add_argument("--save-metrics-json", default="outputs/benchmark_v2/agentformer_metrics.json")
    ap.add_argument("--save-ckpt", default="outputs/benchmark_v2/agentformer_ckpt.pt")
    ap.add_argument("--save-pred-npz", default=None,
                    help="optional: dump val+test most-likely predictions (conformal)")
    ap.add_argument("--save-samples", action="store_true",
                    help="with --save-pred-npz: also store --eval-k prior samples (N,K,T,2)")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training: rebuild the model from --save-ckpt (its saved args) "
                         "and evaluate val + test")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--decay-step", type=int, default=10)
    ap.add_argument("--decay-gamma", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--nz", type=int, default=32)
    ap.add_argument("--sample-loss-k", type=int, default=5, help="k for the training variety loss")
    ap.add_argument("--eval-k", type=int, default=20, help="K for oracle best-of-K eval (matches T++)")
    ap.add_argument("--tf-model-dim", type=int, default=256)
    ap.add_argument("--tf-ff-dim", type=int, default=512)
    ap.add_argument("--tf-nhead", type=int, default=8)
    ap.add_argument("--tf-dropout", type=float, default=0.1)
    ap.add_argument("--nlayer", type=int, default=2)
    ap.add_argument("--no-out-mlp", action="store_true", help="disable decoder/encoder out MLP")
    ap.add_argument("--traj-scale", type=float, default=10.0)
    ap.add_argument("--no-rot-aug", dest="rot_aug", action="store_false", default=True)
    ap.add_argument("--stride", type=int, default=5,
                    help="run AgentFormer on every stride-th frame (5 -> its native 2 Hz: "
                         "4 obs + 12 future keyframes), predictions linearly upsampled "
                         "to 10 Hz for scoring; 1 = full 10 Hz (memory-infeasible for "
                         "the largest IDD-3D scenes on a 24 GB GPU)")

    ap.add_argument("--smoke", action="store_true",
                    help="tiny model + few seqs/epochs: plumbing check only")
    ap.add_argument("--train-seq-limit", type=int, default=None)
    ap.add_argument("--eval-seq-limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.eval_only:
        # architecture + data settings must match the trained checkpoint
        saved = torch.load(args.save_ckpt, map_location="cpu", weights_only=False)["args"]
        for key in ("nz", "sample_loss_k", "eval_k", "tf_model_dim", "tf_ff_dim", "tf_nhead",
                    "tf_dropout", "nlayer", "no_out_mlp", "traj_scale", "rot_aug", "stride"):
            setattr(args, key, saved[key])
        args.epochs = 0

    if args.smoke:
        # tiny model + smallest scenes: AgentFormer's 60-step AR decode is slow on
        # CPU, so keep everything small — this is a plumbing check, not a number.
        args.tf_model_dim = min(args.tf_model_dim, 32)
        args.tf_ff_dim = min(args.tf_ff_dim, 64)
        args.tf_nhead = 4
        args.nlayer = 1
        args.no_out_mlp = True
        args.nz = min(args.nz, 8)
        args.epochs = min(args.epochs, 2)
        args.sample_loss_k = min(args.sample_loss_k, 2)
        args.eval_k = min(args.eval_k, 2)
        args.train_seq_limit = args.train_seq_limit or 8
        args.eval_seq_limit = args.eval_seq_limit or 6

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    set_seed(args.seed)

    obs_len, pred_len = 20, 60
    obs_idx, fut_idx = keyframe_indices(obs_len, pred_len, args.stride)
    cfg = build_map_free_cfg(
        len(obs_idx), len(fut_idx), nz=args.nz, sample_k=args.sample_loss_k,
        tf_model_dim=args.tf_model_dim, tf_ff_dim=args.tf_ff_dim,
        tf_nhead=args.tf_nhead, tf_dropout=args.tf_dropout, nlayer=args.nlayer,
        out_mlp_dim=None if args.no_out_mlp else (512, 256),
        rand_rot_scene=bool(args.rot_aug), traj_scale=args.traj_scale)
    cfg.stride = int(args.stride)

    def make_ds(split):
        return CanonicalSeqDataset(args.data, split, args.splits_file,
                                   rot_aug=False, seed=args.seed)
    train_ds, val_ds, test_ds = make_ds("train"), make_ds("val"), make_ds("test")

    smoke_cap = 6 if args.smoke else None
    train_indices = pick_indices(train_ds, args.train_seq_limit, smallest=args.smoke, max_agents=smoke_cap)
    val_indices = pick_indices(val_ds, args.eval_seq_limit, smallest=args.smoke, max_agents=smoke_cap)
    test_indices = pick_indices(test_ds, args.eval_seq_limit, smallest=args.smoke, max_agents=smoke_cap)

    model = build_model(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.decay_step, gamma=args.decay_gamma)

    rng = np.random.default_rng(args.seed)
    best_val_ade, best_epoch = float("inf"), -1
    Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(train_indices)
        rng.shuffle(order)
        losses = []
        for idx in order:
            obs, pred, _, _, fm, at = train_ds[int(idx)]
            data = seq_to_af_data(obs, pred, fm, cfg.traj_scale, stride=cfg.get("stride", 1))
            model.set_data(data)
            model()
            loss, _, _ = model.compute_loss()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            model.step_annealer()
            losses.append(float(loss.item()))
        sched.step()

        val = evaluate(model, val_ds, cfg, args.eval_k, device, indices=val_indices)
        if device.type == "cuda":
            # Hand the K-sample eval's cached blocks back to the device: on a shared
            # GPU a process pinned near the limit stalls in allocator retries.
            torch.cuda.empty_cache()
        val_ml_ade = val["most_likely"]["ade"]
        improved = val_ml_ade < best_val_ade - 1e-6
        if improved:
            best_val_ade, best_epoch = val_ml_ade, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_ml_ade": val_ml_ade, "args": vars(args)}, args.save_ckpt)
        print(f"epoch {epoch:3d} | train_loss={np.mean(losses):.3f} "
              f"| val_ML_ADE={val_ml_ade:.4f} | val_bestof{args.eval_k}_ADE="
              f"{val['best_of_k']['ade']:.4f}" + ("  (best)" if improved else ""), flush=True)

    # Final: reload best, evaluate val + test in full (population contract holds here).
    ckpt = torch.load(args.save_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if args.eval_only:
        best_epoch = ckpt["epoch"]
    val_summary = evaluate(model, val_ds, cfg, args.eval_k, device, indices=val_indices)
    test_summary = evaluate(model, test_ds, cfg, args.eval_k, device, indices=test_indices)

    metrics = {
        "model": "AgentFormer_v2_mapfree",
        "upstream": "Khrylx/AgentFormer (Yuan et al., ICCV 2021), base CVAE, patched",
        "data": args.data, "splits_file": args.splits_file,
        "obs_len": obs_len, "pred_len": pred_len, "seed": args.seed,
        "best_epoch": best_epoch, "eval_k": args.eval_k,
        "val": val_summary, "test": test_summary,
        "hyper": {k: v for k, v in vars(args).items()
                  if k not in {"data", "splits_file", "save_metrics_json",
                               "save_ckpt", "save_pred_npz", "save_samples", "eval_only"}},
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": "Benchmark v2: upstream AgentFormer base CVAE, map-free "
                "(scene_norm+vel, no map, no heading), standard-normal prior. "
                "most_likely = z=0 deployable rollout; best_of_k = oracle min over "
                f"K={args.eval_k} prior samples. Masked per-timestep metrics, FDE "
                "at last valid step, model selection on val most-likely ADE. "
                + (f"Run at stride {args.stride} (10/{args.stride} Hz keyframes, AgentFormer's "
                   "native rate); predictions linearly upsampled to 10 Hz, so it is scored "
                   "on the same agents and frames as every other model. "
                   if args.stride > 1 else "")
                + ("SMOKE run (tiny model, seq-limited) — plumbing only, not a benchmark number."
                   if args.smoke else ""),
    }
    out = Path(args.save_metrics_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for split, s in (("val", val_summary), ("test", test_summary)):
        print(f"{split}: most-likely ADE={s['most_likely']['ade']:.4f} "
              f"FDE={s['most_likely']['fde']:.4f} (n={s['most_likely']['n_agents']}) | "
              f"best-of-{args.eval_k} ADE={s['best_of_k']['ade']:.4f} "
              f"FDE={s['best_of_k']['fde']:.4f}")
    print(f"saved metrics: {out}")

    if args.save_pred_npz:
        dump_most_likely(model, {"val": val_ds, "test": test_ds}, cfg, device, args.save_pred_npz,
                         sample_k=args.eval_k if args.save_samples else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
