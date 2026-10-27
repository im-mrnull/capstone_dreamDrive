"""Train/evaluate upstream LED (Leapfrog Diffusion) on the canonical v2 dataset.

LED (Mao et al., CVPR 2023; https://github.com/MediaBrain-SJTU/LED), driven
through its model modules by scripts/dd_led.py so the eval population is the
shared canonical one. Two stages, as in the paper:

  1. core denoiser: DDPM noise-estimation loss (upstream ships only an NBA-
     pretrained core, so it is trained here); best epoch by val noise loss.
  2. leapfrog initializer: core frozen, best-of-K + uncertainty loss through the
     last n_tau reverse steps; best epoch by val most-likely ADE (deployable),
     the same selection rule as every other benchmark-v2 model.

Benchmark v2 interface (matches AgentFormer/Trajectron++): canonical NPZ + frozen
splits; seeds affect init/shuffle only; masked per-timestep ADE/FDE (dd_data,
FDE at last valid step); multimodal JSON schema {"val": {"most_likely": ...,
"best_of_k": ...}, "test": {...}} that run_benchmark_v2.py aggregates.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from dd_data import CanonicalSeqDataset, masked_ade_fde_np, summarize_np
from dd_led import (
    Diffusion,
    build_batch,
    build_models,
    initializer_loss,
    leapfrog_loc,
    predict_batch,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def batches(ds, indices, batch_seqs: int):
    """Yield (items, scene ids) for consecutive groups of sequences."""
    for b in range(0, len(indices), batch_seqs):
        chunk = indices[b:b + batch_seqs]
        items, scenes = [], []
        for i in chunk:
            obs, pred, _, _, fm, at = ds[int(i)]
            items.append((obs, pred, fm, at))
            scenes += [str(ds.scene_id[int(ds.seq_indices[int(i)])])] * obs.shape[0]
        yield items, scenes


@torch.no_grad()
def run_split(core, init, diff, ds, args, device, indices=None, collect=False):
    """Most-likely + oracle best-of-K summaries over the given sequences (all by
    default). collect=True also returns the flat most-likely dump rows."""
    core.eval(); init.eval()
    idx = list(range(len(ds))) if indices is None else list(indices)
    ade_ml, fde_ml, ade_k, fde_k, paths, types = [], [], [], [], [], []
    dump = {"pred": [], "gt": [], "fut_mask": [], "agent_type": [], "scene_id": []}
    for items, scenes in batches(ds, idx, args.batch_seqs):
        b = build_batch(items, args.traj_scale, device)
        ml, samp = predict_batch(core, init, diff, b, args.traj_scale, args.n_tau)
        ml, samp = ml.cpu().numpy(), samp.cpu().numpy()
        gt = np.concatenate([np.transpose(p, (0, 2, 1)) for _, p, _, _ in items])
        fm = np.concatenate([f for _, _, f, _ in items]).astype(bool)
        a, f, path = masked_ade_fde_np(ml, gt, fm)
        ak = np.stack([masked_ade_fde_np(samp[:, k], gt, fm)[0] for k in range(samp.shape[1])], 1)
        fk = np.stack([masked_ade_fde_np(samp[:, k], gt, fm)[1] for k in range(samp.shape[1])], 1)
        ade_ml.append(a); fde_ml.append(f); paths.append(path); types.append(b["agent_type"])
        ade_k.append(ak.min(1)); fde_k.append(fk.min(1))
        if collect:
            dump["pred"].append(ml.astype(np.float32)); dump["gt"].append(gt.astype(np.float32))
            dump["fut_mask"].append(fm); dump["agent_type"].append(b["agent_type"])
            dump["scene_id"].append(np.asarray(scenes, dtype=object))
    cat = np.concatenate
    path, typ = cat(paths), cat(types)
    out = {"most_likely": summarize_np(cat(ade_ml), cat(fde_ml), path, typ),
           "best_of_k": summarize_np(cat(ade_k), cat(fde_k), path, typ)}
    return (out, {k: cat(v) for k, v in dump.items()}) if collect else out


@torch.no_grad()
def val_noise_loss(core, diff, ds, args, device, indices) -> float:
    core.eval()
    torch.manual_seed(0)   # same t / eps draws every epoch -> comparable losses
    losses = []
    for items, _ in batches(ds, indices, args.batch_seqs):
        b = build_batch(items, args.traj_scale, device)
        losses.append(float(diff.noise_estimation_loss(core, b["past"], b["fut"],
                                                       b["fut_mask"], b["mask"])))
    return float(np.mean(losses))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/canonical_world_obs20_pred60.npz")
    ap.add_argument("--splits-file", default="splits/scene_splits.json")
    ap.add_argument("--save-metrics-json", default="outputs/benchmark_v2/led_metrics.json")
    ap.add_argument("--save-ckpt", default="outputs/benchmark_v2/led_ckpt.pt")
    ap.add_argument("--save-pred-npz", default=None,
                    help="optional: dump val+test most-likely predictions (conformal)")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--core-epochs", type=int, default=100)
    ap.add_argument("--init-epochs", type=int, default=100)
    ap.add_argument("--core-lr", type=float, default=1e-3)
    ap.add_argument("--init-lr", type=float, default=1e-3, help="upstream cfg lr")
    ap.add_argument("--decay-step", type=int, default=8, help="upstream cfg")
    ap.add_argument("--decay-gamma", type=float, default=0.5, help="upstream cfg")
    ap.add_argument("--batch-seqs", type=int, default=8,
                    help="canonical sequences per batch for stage 1 and eval "
                         "(~upstream's 10 NBA scenes)")
    ap.add_argument("--init-batch-seqs", type=int, default=4,
                    help="sequences per stage-2 batch (backprop through n_tau denoising "
                         "steps x K samples x 60 frames is the memory peak)")
    ap.add_argument("--k", type=int, default=20, help="leapfrog samples (oracle best-of-K)")
    ap.add_argument("--n-tau", type=int, default=5, help="reverse steps after the leapfrog")
    ap.add_argument("--diffusion-steps", type=int, default=100)
    ap.add_argument("--traj-scale", type=float, default=10.0)
    ap.add_argument("--no-rot-aug", dest="rot_aug", action="store_false", default=True)

    ap.add_argument("--smoke", action="store_true", help="few seqs/epochs: plumbing only")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training: load --save-ckpt and evaluate val + test (with "
                         "--save-pred-npz, also dump the most-likely predictions)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.eval_only:
        args.core_epochs = args.init_epochs = 0
    if args.smoke:
        args.core_epochs = min(args.core_epochs, 2)
        args.init_epochs = min(args.init_epochs, 2)
        args.k = min(args.k, 4)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    set_seed(args.seed)

    def make_ds(split, rot=False):
        return CanonicalSeqDataset(args.data, split, args.splits_file, rot_aug=rot, seed=args.seed)
    train_ds = make_ds("train", rot=bool(args.rot_aug))
    val_ds, test_ds = make_ds("val"), make_ds("test")
    lim = 6 if args.smoke else None
    train_idx = list(range(len(train_ds)))[:lim]
    val_idx = list(range(len(val_ds)))[:lim]
    test_idx = list(range(len(test_ds)))[:lim]

    obs, pred, *_ = train_ds[0]
    core, init = build_models(obs.shape[-1], pred.shape[-1], args.k)
    core, init = core.to(device), init.to(device)
    diff = Diffusion(device, steps=args.diffusion_steps)
    rng = np.random.default_rng(args.seed)
    Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)

    # ---- stage 1: core denoiser -------------------------------------------
    opt = torch.optim.AdamW(core.parameters(), lr=args.core_lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(args.core_epochs // 4, 1), gamma=0.5)
    best_core, best_core_loss, best_core_epoch = None, float("inf"), -1
    for epoch in range(1, args.core_epochs + 1):
        core.train()
        order = list(train_idx); rng.shuffle(order)
        losses = []
        for items, _ in batches(train_ds, order, args.batch_seqs):
            b = build_batch(items, args.traj_scale, device)
            loss = diff.noise_estimation_loss(core, b["past"], b["fut"], b["fut_mask"], b["mask"])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0); opt.step()
            losses.append(float(loss))
        sched.step()
        vl = val_noise_loss(core, diff, val_ds, args, device, val_idx)
        if vl < best_core_loss - 1e-6:
            best_core_loss, best_core_epoch = vl, epoch
            best_core = {k: v.detach().clone() for k, v in core.state_dict().items()}
        print(f"core epoch {epoch:3d} | train_eps={np.mean(losses):.4f} | val_eps={vl:.4f}"
              + ("  (best)" if epoch == best_core_epoch else ""), flush=True)
    if best_core is not None:
        core.load_state_dict(best_core)
    for p in core.parameters():
        p.requires_grad_(False)

    # ---- stage 2: leapfrog initializer -------------------------------------
    opt = torch.optim.AdamW(init.parameters(), lr=args.init_lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.decay_step, gamma=args.decay_gamma)
    best_val_ade, best_epoch = float("inf"), -1
    for epoch in range(1, args.init_epochs + 1):
        core.eval(); init.train()
        order = list(train_idx); rng.shuffle(order)
        losses = []
        for items, _ in batches(train_ds, order, args.init_batch_seqs):
            b = build_batch(items, args.traj_scale, device)
            loc, _, logvar = leapfrog_loc(init, b["past"], b["mask"])
            gen = diff.denoise(core, b["past"], b["mask"], loc, args.n_tau, checkpoint=True)
            loss, _, _ = initializer_loss(gen, b["fut"], b["fut_mask"], logvar)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(init.parameters(), 1.0); opt.step()
            losses.append(float(loss))
        sched.step()
        val = run_split(core, init, diff, val_ds, args, device, val_idx)
        if device.type == "cuda":
            torch.cuda.empty_cache()   # release the K-sample eval cache (shared GPU)
        v = val["most_likely"]["ade"]
        improved = v < best_val_ade - 1e-6
        if improved:
            best_val_ade, best_epoch = v, epoch
            torch.save({"core": core.state_dict(), "init": init.state_dict(), "epoch": epoch,
                        "core_epoch": best_core_epoch, "val_ml_ade": v, "args": vars(args)},
                       args.save_ckpt)
        print(f"init epoch {epoch:3d} | train_loss={np.mean(losses):.3f} | val_ML_ADE={v:.4f} "
              f"| val_bestof{args.k}_ADE={val['best_of_k']['ade']:.4f}"
              + ("  (best)" if improved else ""), flush=True)

    # ---- final: reload best, full val + test (population contract) ---------
    ck = torch.load(args.save_ckpt, map_location=device, weights_only=False)
    core.load_state_dict(ck["core"]); init.load_state_dict(ck["init"])
    if args.eval_only:
        best_epoch, best_core_epoch = ck["epoch"], ck["core_epoch"]
    torch.manual_seed(args.seed)
    val_summary, val_dump = run_split(core, init, diff, val_ds, args, device, val_idx, collect=True)
    test_summary, test_dump = run_split(core, init, diff, test_ds, args, device, test_idx, collect=True)

    metrics = {
        "model": "LED_v2",
        "upstream": "MediaBrain-SJTU/LED (Mao et al., CVPR 2023), modules reused, IDD-3D shapes",
        "data": args.data, "splits_file": args.splits_file,
        "obs_len": int(obs.shape[-1]), "pred_len": int(pred.shape[-1]), "seed": args.seed,
        "best_epoch": best_epoch, "core_best_epoch": best_core_epoch, "eval_k": args.k,
        "val": val_summary, "test": test_summary,
        "hyper": {k: v for k, v in vars(args).items()
                  if k not in {"data", "splits_file", "save_metrics_json",
                               "save_ckpt", "save_pred_npz"}},
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": "Benchmark v2: LED leapfrog diffusion at 10 Hz. Stage 1 trains the core "
                "denoiser (DDPM eps loss; upstream ships only an NBA core), stage 2 the "
                "leapfrog initializer with the core frozen. most_likely = initializer mean "
                "denoised with zero noise (deterministic); best_of_k = oracle min over "
                f"K={args.k} leapfrog samples. Masked per-timestep metrics, FDE at last "
                "valid step, model selection on val most-likely ADE. "
                + ("SMOKE run (seq-limited) — plumbing only, not a benchmark number."
                   if args.smoke else ""),
    }
    out = Path(args.save_metrics_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for split, s in (("val", val_summary), ("test", test_summary)):
        print(f"{split}: most-likely ADE={s['most_likely']['ade']:.4f} "
              f"FDE={s['most_likely']['fde']:.4f} (n={s['most_likely']['n_agents']}) | "
              f"best-of-{args.k} ADE={s['best_of_k']['ade']:.4f} FDE={s['best_of_k']['fde']:.4f}")
    print(f"saved metrics: {out}")

    if args.save_pred_npz:
        Path(args.save_pred_npz).parent.mkdir(parents=True, exist_ok=True)
        d = {k: np.concatenate([val_dump[k], test_dump[k]]) for k in val_dump}
        np.savez_compressed(
            args.save_pred_npz, pred=d["pred"], gt=d["gt"], fut_mask=d["fut_mask"],
            agent_type=d["agent_type"].astype(object), scene_id=d["scene_id"].astype(object),
            split=np.asarray(["val"] * len(val_dump["pred"]) + ["test"] * len(test_dump["pred"]),
                             dtype=object),
            model="LED (most-likely)")
        print(f"saved most-likely predictions: {args.save_pred_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
