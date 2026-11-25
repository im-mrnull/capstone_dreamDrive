"""Train/evaluate AutoBot-Joint (map-free) on the canonical v2 dataset.

AutoBot (Girgis et al., ICLR 2022; https://github.com/roggirg/AutoBots, BSD-3,
cloned to third_party/autobots at 0ceb8f7): the upstream AutoBotJoint module is
used unchanged, in its map-free TrajNet++ configuration (no map lanes, one agent
type). It is the chaos gap's map-free state-of-the-art model: IDD-3D has no HD
map, and every UniTraj model needs one.

Adapted here, not upstream:
- variable scene sizes: a batch holds sequences of one size, and the model's agent
  count (_M, used only in reshapes and repeats) is set per batch. No padding
  reaches the model: upstream sums the mode-probability logits over every agent
  slot, so padded agents would change a scene's mode probabilities and make its
  predictions depend on the rest of the batch;
- masked loss: upstream's joint loss (utils/train_helpers.nll_loss_multimodes_joint)
  treats agent 0's future as always present. Here every agent's NLL, entropy and
  ADE/FDE terms are masked by its own fut_mask, and FDE is at its last valid step;
- scene frame: positions relative to the mean last-observed position of the
  sequence's agents, in metres / --pos-scale (upstream centres TrajNet++ scenes on
  the primary agent, which canonical sequences do not have), with random scene
  rotation on train (dd_data).

The loss is otherwise upstream's: Laplace NLL weighted by the mode posterior,
entropy_weight x the bivariate-Gaussian entropy of the most uncertain mode,
kl_weight x KL(posterior || prior), and 100 x the best mode's ADE + FDE. Defaults
are upstream's TrajNet++ AutoBot-Joint command (README): K=6, d_k 128, 2 encoder
and 2 decoder layers, 16 heads, feed-forward 384, dropout 0.1, entropy weight 40,
KL weight 20, Adam lr 7.5e-4 (eps 1e-4) halved at epochs 10-50, grad clip 5,
64 sequences per batch.

Benchmark v2 interface: canonical NPZ + frozen splits; masked per-timestep ADE/FDE
(dd_data, FDE at the last valid step); most_likely = the scene's most probable mode
(the modes are joint), best_of_k = per-agent oracle min over the K modes; selection
on val most-likely ADE; JSON schema {"val": {"most_likely", "best_of_k"}, "test"}.
--init-ckpt starts from another run's weights (fine-tuning).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Laplace

from dd_data import (Batch, CanonicalSeqDataset, ErrorAccumulator, collate_pad,
                     per_agent_errors, per_agent_errors_bestofk)

REPO = Path(__file__).resolve().parent.parent
AB_REPO = REPO / "third_party" / "autobots"


def load_autobot_joint(ab_repo: str | Path = AB_REPO):
    """The upstream AutoBotJoint class. Its modules import `models.*` from the repo
    root, and LED's repo has a `models` package too, so the import leaves neither
    the path entry nor a `models` package behind in sys.modules."""
    p = Path(ab_repo)
    if not (p / "models" / "autobot_joint.py").exists():
        raise FileNotFoundError(
            f"AutoBots repo not found at {p}. Clone it: git clone --depth 1 "
            f"https://github.com/roggirg/AutoBots.git third_party/autobots")
    is_models = lambda k: k == "models" or k.startswith("models.")
    saved = {k: sys.modules.pop(k) for k in [k for k in sys.modules if is_models(k)]}
    sys.path.insert(0, str(p))
    try:
        from models.autobot_joint import AutoBotJoint
    finally:
        sys.path.remove(str(p))
        for k in [k for k in sys.modules if is_models(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
    return AutoBotJoint


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def seq_sizes(ds: CanonicalSeqDataset) -> np.ndarray:
    se = ds.seq_start_end[ds.seq_indices]
    return se[:, 1] - se[:, 0]


def batches(sizes: np.ndarray, indices, batch_seqs: int, max_slots: int):
    """Groups of sequences of one size (no padding), at most batch_seqs sequences and
    max_slots agents each; within a size, the order of `indices` is kept."""
    by_size: dict[int, list[int]] = {}
    for i in indices:
        by_size.setdefault(int(sizes[int(i)]), []).append(int(i))
    for m, group in sorted(by_size.items()):
        step = max(1, min(batch_seqs, max_slots // m))
        for b in range(0, len(group), step):
            yield group[b:b + step]


def train_batches(sizes, order, batch_seqs, max_slots, rng):
    """Shuffled order -> one-size batches, in random batch order."""
    out = list(batches(sizes, order, batch_seqs, max_slots))
    rng.shuffle(out)
    return out


def split_seqs(chunk, m: int, max_pairs: int):
    """[chunk], unless its agent pairs (sequences x m x m, what the social attention holds) exceed max_pairs
    (0: never split); then runs of consecutive whole sequences of at most max_pairs pairs each (at least one)."""
    if max_pairs <= 0 or len(chunk) * m * m <= max_pairs:
        return [chunk]
    step = max(1, max_pairs // (m * m))
    return [chunk[i:i + step] for i in range(0, len(chunk), step)]


class Diverged(RuntimeError):
    """A non-finite prediction, loss or gradient norm. The step would turn the weights NaN, and a NaN model never
    recovers (every later epoch's val ADE is NaN, never selected); raised before opt.step(), weights untouched."""


def train_step(model, opt, ds, chunk, m, args, device):
    """One optimizer step on one one-size batch. Every loss term is a mean over the batch's sequences, so a batch
    split by split_seqs runs as parts whose losses are weighted by their share of the sequences, with gradients
    accumulated: the same loss and the same step (only dropout draws differ). Returns (loss parts, agents seen).
    Raises Diverged (before any weight changes) on a non-finite prediction, loss or gradient norm."""
    pieces = split_seqs(chunk, m, args.max_batch_pairs)
    opt.zero_grad(set_to_none=True)
    agg, seen = {}, 0
    for sub in pieces:
        b = collate_pad([ds[i] for i in sub]).to(device)
        pred_obs, mode_probs, fut, fut_m, _ = forward(model, b, args.pos_scale)
        if not (torch.isfinite(pred_obs).all() and torch.isfinite(mode_probs).all()):
            raise Diverged("non-finite prediction")
        loss, parts = autobot_joint_loss(pred_obs, mode_probs, fut, fut_m, b.agent_mask,
                                         args.entropy_weight, args.kl_weight)
        if not torch.isfinite(loss):
            raise Diverged("non-finite loss")
        if len(pieces) == 1:
            loss.backward()
            agg = parts
        else:
            w = len(sub) / len(chunk)
            (loss * w).backward()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + w * v
        seen += int(b.agent_mask.sum())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    if not torch.isfinite(norm):
        raise Diverged("non-finite gradient norm")
    opt.step()
    return agg, seen


def scene_inputs(b: Batch, scale: float):
    """Batch -> AutoBot-Joint inputs in the scene frame.

    Returns ego_in (B, T_obs, 3), agents_in (B, T_obs, M-1, 3) (x, y, presence), the
    future fut (B, T, M, 2), its mask fut_m (B, T, M) and the frame origin (B, 2).
    Every real agent's observation is complete (canonical presence rule)."""
    am = b.agent_mask
    last = b.obs_abs[:, :, :, -1]                                          # (B, M, 2)
    origin = (last * am[..., None]).sum(1) / am.sum(1, keepdim=True)       # (B, 2)
    obs = (b.obs_abs.permute(0, 3, 1, 2) - origin[:, None, None]) / scale  # (B, T_obs, M, 2)
    fut = (b.pred_abs.permute(0, 3, 1, 2) - origin[:, None, None]) / scale
    m_obs = am[:, None, :, None].expand(-1, obs.shape[1], -1, -1).float()
    x = torch.cat([obs * m_obs, m_obs], dim=-1)                            # (B, T_obs, M, 3)
    fut_m = (b.fut_mask.permute(0, 2, 1) & am[:, None, :]).float()         # (B, T, M)
    return x[:, :, 0], x[:, :, 1:], fut, fut_m, origin


def forward(model, b: Batch, scale: float):
    ego_in, agents_in, fut, fut_m, origin = scene_inputs(b, scale)
    B, M = b.agent_mask.shape
    model._M = M - 1
    types = torch.ones((B, M, 1), device=ego_in.device)
    roads = torch.ones((B, 1, 1), device=ego_in.device)
    pred, mode_probs = model(ego_in, agents_in, roads, types)              # (K, T, B, M, 5), (B, K)
    return pred, mode_probs, fut, fut_m, origin


def autobot_joint_loss(pred, mode_probs, fut, fut_m, agent_mask, entropy_weight, kl_weight):
    """Upstream nll_loss_multimodes_joint with every agent masked by its own future
    presence. pred (K, T, B, M, 5), mode_probs (B, K), fut (B, T, M, 2), fut_m (B, T, M)."""
    m = fut_m.permute(1, 0, 2).unsqueeze(0)                                # (1, T, B, M)
    gt = fut.permute(1, 0, 2, 3).unsqueeze(0)                              # (1, T, B, M, 2)
    nll_steps = -Laplace(pred[..., :2], pred[..., 2:4]).log_prob(gt).sum(-1)
    per_step = (nll_steps * m).sum(-1) / m.sum(-1).clamp_min(1.0)          # (K, T, B)
    nll = per_step.sum(1).transpose(0, 1)                                  # (B, K)

    log_prior = torch.log(mode_probs)
    post = torch.softmax(-nll.detach() + log_prior.detach(), dim=1)        # mode posterior
    nll_loss = (nll * post).sum(1).mean()

    # entropy of each mode's bivariate Gaussian (upstream's term), summed over time,
    # averaged over real agents; the most uncertain mode is penalised
    sx, sy, rho = pred[..., 2], pred[..., 3], pred[..., 4]
    ent = 1.0 + math.log(2 * math.pi) + torch.log(sx) + torch.log(sy) + 0.5 * torch.log(1 - rho ** 2)
    am = agent_mask.float().unsqueeze(0)                                   # (1, B, M)
    ent = (ent.sum(1) * am).sum(-1) / am.sum(-1)                           # (K, B)
    entropy_loss = ent.max(0).values.mean()

    kl_loss = (post * (torch.log(post.clamp_min(1e-12)) - log_prior)).sum(1).mean()

    # 100 x (ADE + FDE) of the best mode, FDE at each agent's last valid step
    err = torch.linalg.vector_norm(pred[..., :2] - gt, dim=-1)             # (K, T, B, M)
    ade = (err * m).sum((1, 3)) / m.sum((1, 3)).clamp_min(1.0)             # (K, B)
    T = err.shape[1]
    last = (torch.arange(T, device=err.device).view(T, 1, 1) * fut_m.permute(1, 0, 2)).argmax(0)  # (B, M)
    fde_all = err.gather(1, last.unsqueeze(0).unsqueeze(1).expand(err.shape[0], 1, -1, -1)).squeeze(1)
    fde = (fde_all * am).sum(-1) / am.sum(-1)                              # (K, B)
    aux = 100.0 * (ade + fde).min(0).values.mean()

    loss = nll_loss + entropy_weight * entropy_loss + kl_weight * kl_loss + aux
    return loss, {"nll": float(nll_loss), "entropy": float(entropy_loss), "kl": float(kl_loss), "adefde": float(aux)}


def world_modes(pred, origin, scale):
    """(K, T, B, M, 5) scene-frame modes -> (B, M, K, 2, T) world positions."""
    pos = pred[..., :2] * scale + origin[None, None, :, None, :]
    return pos.permute(2, 3, 0, 4, 1)


@torch.no_grad()
def run_split(model, ds, sizes, args, device, indices=None, collect=False):
    """Most-likely + oracle best-of-K summaries over the given sequences (all by
    default). collect=True also returns the flat per-agent dump, in sequence order."""
    model.eval()
    idx = list(range(len(ds))) if indices is None else list(indices)
    acc_ml, acc_k = ErrorAccumulator(), ErrorAccumulator()
    rows = {}
    for chunk in batches(sizes, idx, args.batch_seqs, args.max_slots):
        b = collate_pad([ds[i] for i in chunk]).to(device)
        pred, mode_probs, *_, origin = forward(model, b, args.pos_scale)
        modes = world_modes(pred, origin, args.pos_scale)                  # (B, M, K, 2, T)
        best = mode_probs.argmax(1)                                        # (B,)
        ml = modes[torch.arange(len(chunk), device=device), :, best]       # (B, M, 2, T)
        acc_ml.add(*per_agent_errors(ml, b.pred_abs, b.agent_mask, b.fut_mask), b.agent_type, b.agent_mask)
        acc_k.add(*per_agent_errors_bestofk(modes, b.pred_abs, b.agent_mask, b.fut_mask), b.agent_type, b.agent_mask)
        if collect:                                                        # one-size batch: every slot is real
            M = b.agent_mask.shape[1]
            per = {"pred": ml.permute(0, 1, 3, 2), "samples": modes.permute(0, 1, 2, 4, 3),
                   "probs": mode_probs[:, None, :].expand(-1, M, -1), "gt": b.pred_abs.permute(0, 1, 3, 2),
                   "fut_mask": b.fut_mask}
            per = {k: v.cpu().numpy() for k, v in per.items()}
            for r, i in enumerate(chunk):
                rows[i] = {**{k: v[r] for k, v in per.items()},
                           "agent_type": np.asarray(b.agent_type[r], dtype=object),
                           "scene_id": np.full(M, str(ds.scene_id[int(ds.seq_indices[i])]), dtype=object)}
    out = {"most_likely": acc_ml.summary(), "best_of_k": acc_k.summary()}
    if not collect:
        return out
    return out, {k: np.concatenate([rows[i][k] for i in sorted(rows)]) for k in rows[idx[0]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/canonical_world_obs20_pred60.npz")
    ap.add_argument("--splits-file", default="splits/scene_splits.json")
    ap.add_argument("--save-metrics-json", default="outputs/chaos_gap/autobot_metrics.json")
    ap.add_argument("--save-ckpt", default="outputs/chaos_gap/autobot_ckpt.pt")
    ap.add_argument("--save-pred-npz", default=None,
                    help="optional: dump val+test predictions, the K modes and their probabilities")
    ap.add_argument("--init-ckpt", default=None, help="start from these weights (fine-tuning)")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150, help="upstream default")
    ap.add_argument("--batch-seqs", type=int, default=64, help="sequences per batch (upstream 64)")
    ap.add_argument("--max-slots", type=int, default=512,
                    help="memory cap: sequences x sequence size per batch. Training peaks at ~26 MB "
                         "per slot at 60 future steps, so 512 is ~14 GB; 2048 exceeds a 24 GB card, "
                         "and Windows then spills to system RAM (~20x slower)")
    ap.add_argument("--max-batch-pairs", type=int, default=26244,
                    help="a training batch of B sequences of m agents with B*m*m above this runs as parts of whole "
                         "sequences with accumulated gradients: the same loss and one optimizer step (train_step). "
                         "Default 9 x 54^2, the largest batch any Argoverse 2 run trained with; IDD-3D's 55-61-agent "
                         "scenes exceed it (the 61-agent scene ran out of memory under the 0.8 cap on 2026-11-11). "
                         "0: never split")
    ap.add_argument("--lr", type=float, default=7.5e-4)
    ap.add_argument("--adam-eps", type=float, default=1e-4)
    ap.add_argument("--lr-sched", type=int, nargs="*", default=[10, 20, 30, 40, 50],
                    help="epochs halving the lr (none given: a constant lr)")
    ap.add_argument("--select", choices=["best", "last"], default="best",
                    help="best: the checkpoint with the lowest val most-likely ADE; last: the final epoch (no val-based choice)")
    ap.add_argument("--select-include-init", action="store_true",
                    help="with --select best: the starting weights (epoch 0) are also a candidate")
    ap.add_argument("--patience", type=int, default=0,
                    help="with --select best: stop once the val most-likely ADE has not improved for this many "
                         "epochs (0: every epoch runs). The kept checkpoint is still the best epoch seen")
    ap.add_argument("--train-repeat", type=int, default=1,
                    help="repeat the training set this many times per epoch (fine-tuning: equal optimizer steps across training-set sizes; schedules stay in epochs)")
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--num-modes", type=int, default=6)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--num-encoder-layers", type=int, default=2)
    ap.add_argument("--num-decoder-layers", type=int, default=2)
    ap.add_argument("--tx-hidden-size", type=int, default=384)
    ap.add_argument("--tx-num-heads", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--entropy-weight", type=float, default=40.0)
    ap.add_argument("--kl-weight", type=float, default=20.0)
    ap.add_argument("--pos-scale", type=float, default=1.0, help="scene-frame unit in metres (upstream: 1)")
    ap.add_argument("--no-rot-aug", dest="rot_aug", action="store_false", default=True)

    ap.add_argument("--smoke", action="store_true", help="few seqs/epochs: plumbing only")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training: load --save-ckpt and evaluate val + test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-mem-frac", type=float, default=None,
                    help="cap the CUDA cache at this fraction of the card (Windows spills an "
                         "uncapped cache into shared system memory instead of raising OOM)")
    args = ap.parse_args()

    if args.eval_only:
        args.epochs = 0
    if args.smoke:
        args.epochs = min(args.epochs, 2)

    AutoBotJoint = load_autobot_joint()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if args.gpu_mem_frac and device.type == "cuda":   # free cached blocks at the cap, never spill
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)   # current device
    set_seed(args.seed)

    def make_ds(split, rot=False):
        return CanonicalSeqDataset(args.data, split, args.splits_file, rot_aug=rot, seed=args.seed)
    train_ds = make_ds("train", rot=bool(args.rot_aug))
    val_ds, test_ds = make_ds("val"), make_ds("test")
    sizes = {s: seq_sizes(d) for s, d in (("train", train_ds), ("val", val_ds), ("test", test_ds))}
    lim = 6 if args.smoke else None
    train_idx = list(range(len(train_ds)))[:lim]
    val_idx = list(range(len(val_ds)))[:lim]
    test_idx = list(range(len(test_ds)))[:lim]

    obs, pred, *_ = train_ds[0]
    model = AutoBotJoint(k_attr=2, d_k=args.hidden_size, _M=1, c=args.num_modes, T=pred.shape[-1],
                         L_enc=args.num_encoder_layers, dropout=args.dropout, num_heads=args.tx_num_heads,
                         L_dec=args.num_decoder_layers, tx_hidden_size=args.tx_hidden_size,
                         use_map_lanes=False, map_attr=0, num_agent_types=1, predict_yaw=False).to(device)
    if args.init_ckpt:
        model.load_state_dict(torch.load(args.init_ckpt, map_location=device, weights_only=False)["model"])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.adam_eps)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_sched, gamma=0.5)
    rng = np.random.default_rng(args.seed)
    Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)

    best_val_ade, best_epoch = float("inf"), -1
    history, steps, seen = [], 0, 0             # per epoch: the lr it trained at, val ADE, cumulative steps / agent-windows
    diverged = None                             # {"epoch", "step", "why"} if training stopped on Diverged
    if args.select_include_init and not args.eval_only:   # epoch 0, the starting weights, is a candidate
        best_val_ade, best_epoch = run_split(model, val_ds, sizes["val"], args, device, val_idx)["most_likely"]["ade"], 0
        torch.save({"model": model.state_dict(), "epoch": 0, "val_ml_ade": best_val_ade, "args": vars(args)},
                   args.save_ckpt)
        history.append({"epoch": 0, "lr": None, "val_ade": best_val_ade, "steps": 0, "agent_windows": 0})
    for epoch in range(1, args.epochs + 1):
        model.train()
        lr_now = opt.param_groups[0]["lr"]
        order = list(train_idx) * args.train_repeat; rng.shuffle(order)
        logs = []
        try:
            for chunk in train_batches(sizes["train"], order, args.batch_seqs, args.max_slots, rng):
                parts, n_seen = train_step(model, opt, train_ds, chunk, int(sizes["train"][chunk[0]]), args, device)
                logs.append(parts)
                steps += 1
                seen += n_seen
        except Diverged as e:
            # --select best: every later epoch would be NaN and never selected, so stopping here keeps the same
            # checkpoint (the best epoch so far). --select last has no such epoch to keep: the run fails, as before.
            if args.select != "best" or best_epoch < 0:
                raise
            diverged = {"epoch": epoch, "step": steps + 1, "why": str(e)}
            print(f"diverged in epoch {epoch} (step {steps + 1}, {e}); training stops, the best epoch so far "
                  f"({best_epoch}) is kept", flush=True)
            break
        sched.step()
        val = run_split(model, val_ds, sizes["val"], args, device, val_idx)
        v = val["most_likely"]["ade"]
        history.append({"epoch": epoch, "lr": lr_now, "val_ade": v, "steps": steps, "agent_windows": seen})
        improved = v < best_val_ade - 1e-6 if args.select == "best" else epoch == args.epochs
        if improved:
            best_val_ade, best_epoch = v, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_ml_ade": v, "args": vars(args)},
                       args.save_ckpt)
        mean = {k: np.mean([p[k] for p in logs]) for k in logs[0]}
        print(f"epoch {epoch:3d} | nll={mean['nll']:.2f} ent={mean['entropy']:.2f} kl={mean['kl']:.3f} "
              f"adefde={mean['adefde']:.1f} | val_ML_ADE={v:.4f} "
              f"| val_bestof{args.num_modes}_ADE={val['best_of_k']['ade']:.4f}" + ("  (best)" if improved else ""),
              flush=True)
        if args.patience > 0 and args.select == "best" and epoch - best_epoch >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs since epoch {best_epoch}", flush=True)
            break

    # ---- final: reload best, full val + test (population contract) ---------
    ck = torch.load(args.save_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    if args.eval_only:
        best_epoch = ck["epoch"]
    val_summary, val_dump = run_split(model, val_ds, sizes["val"], args, device, val_idx, collect=True)
    test_summary, test_dump = run_split(model, test_ds, sizes["test"], args, device, test_idx, collect=True)

    metrics = {
        "model": "AutoBot-Joint_v2",
        "upstream": "roggirg/AutoBots (Girgis et al., ICLR 2022) @ 0ceb8f7, AutoBotJoint unchanged, map-free",
        "data": args.data, "splits_file": args.splits_file, "init_ckpt": args.init_ckpt,
        "obs_len": int(obs.shape[-1]), "pred_len": int(pred.shape[-1]), "seed": args.seed,
        "best_epoch": best_epoch, "eval_k": args.num_modes, "history": history, "diverged": diverged,
        "val": val_summary, "test": test_summary,
        "hyper": {k: v for k, v in vars(args).items()
                  if k not in {"data", "splits_file", "save_metrics_json", "save_ckpt", "save_pred_npz",
                               "init_ckpt"}},
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": "AutoBot-Joint at 10 Hz, map-free, class-agnostic. most_likely = the scene's most "
                f"probable of K={args.num_modes} joint modes; best_of_k = per-agent oracle min over the "
                "modes. Masked per-timestep metrics, FDE at last valid step, model selection on val "
                "most-likely ADE. "
                + ("SMOKE run (seq-limited) — plumbing only, not a benchmark number." if args.smoke else ""),
    }
    out = Path(args.save_metrics_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for split, s in (("val", val_summary), ("test", test_summary)):
        print(f"{split}: most-likely ADE={s['most_likely']['ade']:.4f} "
              f"FDE={s['most_likely']['fde']:.4f} (n={s['most_likely']['n_agents']}) | "
              f"best-of-{args.num_modes} ADE={s['best_of_k']['ade']:.4f} FDE={s['best_of_k']['fde']:.4f}")
    print(f"saved metrics: {out}")

    if args.save_pred_npz:
        Path(args.save_pred_npz).parent.mkdir(parents=True, exist_ok=True)
        d = {k: np.concatenate([val_dump[k], test_dump[k]]) for k in val_dump}
        np.savez_compressed(
            args.save_pred_npz, pred=d["pred"].astype(np.float32), samples=d["samples"].astype(np.float32),
            probs=d["probs"].astype(np.float32), gt=d["gt"].astype(np.float32), fut_mask=d["fut_mask"],
            agent_type=d["agent_type"].astype(object), scene_id=d["scene_id"].astype(object),
            split=np.asarray(["val"] * len(val_dump["pred"]) + ["test"] * len(test_dump["pred"]), dtype=object),
            model="AutoBot-Joint (most-likely)")
        print(f"saved predictions: {args.save_pred_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
