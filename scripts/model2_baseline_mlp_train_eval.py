"""Single-agent MLP baseline on the canonical per-agent windows (v2).

NumPy-only 2-layer MLP mapping flattened observed positions (obs_len*2) to
future positions (pred_len*2). Benchmark v2 interface:
- data: outputs/agent_windows_canonical_obs20_pred60.npz (same population as
  every other model; heading-normalized, centered windows)
- split: frozen splits/scene_splits.json; seeds affect init/shuffling only
- masked MSE loss and masked metrics via fut_mask (FDE at last valid step);
  y-standardization statistics are computed over VALID entries only
- model selection on val ADE; final metrics reported for val AND test with
  per-class breakdown.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from dd_split import load_splits
from dd_data import masked_ade_fde_np, summarize_np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(x.dtype)


class MLP:
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 depth: int, rng: np.random.Generator) -> None:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = int(depth)
        self.params: dict[str, np.ndarray] = {}
        prev = int(input_dim)
        for i in range(depth):
            self.params[f"W{i+1}"] = rng.standard_normal(
                (prev, hidden_dim), dtype=np.float32) * np.float32(math.sqrt(2.0 / prev))
            self.params[f"b{i+1}"] = np.zeros((hidden_dim,), dtype=np.float32)
            prev = hidden_dim
        self.params["W_out"] = rng.standard_normal(
            (prev, output_dim), dtype=np.float32) * np.float32(math.sqrt(2.0 / prev))
        self.params["b_out"] = np.zeros((output_dim,), dtype=np.float32)

    def forward(self, x: np.ndarray):
        cache = {"x0": x}
        h = x
        for i in range(self.depth):
            z = h @ self.params[f"W{i+1}"] + self.params[f"b{i+1}"]
            h = relu(z)
            cache[f"z{i+1}"] = z
            cache[f"h{i+1}"] = h
        y = h @ self.params["W_out"] + self.params["b_out"]
        cache["y"] = y
        return y, cache

    def backward(self, cache, dy: np.ndarray, weight_decay: float):
        grads: dict[str, np.ndarray] = {}
        h_last = cache[f"h{self.depth}"]
        grads["W_out"] = h_last.T @ dy
        grads["b_out"] = dy.sum(axis=0)
        if weight_decay > 0:
            grads["W_out"] += np.float32(weight_decay) * self.params["W_out"]
        dh = dy @ self.params["W_out"].T
        for i in range(self.depth, 0, -1):
            z = cache[f"z{i}"]
            h_prev = cache["x0"] if i == 1 else cache[f"h{i-1}"]
            dz = dh * relu_grad(z)
            grads[f"W{i}"] = h_prev.T @ dz
            grads[f"b{i}"] = dz.sum(axis=0)
            if weight_decay > 0:
                grads[f"W{i}"] += np.float32(weight_decay) * self.params[f"W{i}"]
            dh = dz @ self.params[f"W{i}"].T
        return grads


class Adam:
    def __init__(self, params, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = float(lr), beta1, beta2, eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k, p in params.items():
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g * g
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] = p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def fit_masked_standardizer(y: np.ndarray, m: np.ndarray, eps: float = 1e-6):
    """Per-dimension mean/std over VALID entries only. y (N,D), m (N,D)."""
    n = np.maximum(m.sum(axis=0, keepdims=True), 1.0)
    mean = (y * m).sum(axis=0, keepdims=True) / n
    var = ((y - mean) ** 2 * m).sum(axis=0, keepdims=True) / n
    std = np.maximum(np.sqrt(var), eps)
    return mean.astype(np.float32), std.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/agent_windows_canonical_obs20_pred60.npz")
    ap.add_argument("--splits-file", default="splits/scene_splits.json")
    ap.add_argument("--save-metrics-json", default="outputs/benchmark_v2/mlp_metrics.json")
    ap.add_argument("--save-best", default="outputs/benchmark_v2/mlp_best.npz")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))

    d = np.load(args.input, allow_pickle=True)
    obs = d["obs"].astype(np.float32)          # (N, obs, 2)
    fut = d["fut"].astype(np.float32)          # (N, pred, 2)
    fut_mask = d["fut_mask"].astype(bool)      # (N, pred)
    scene_id = d["scene_id"].astype(object)
    agent_type = d["agent_type"].astype(object)

    n, obs_len, _ = obs.shape
    pred_len = fut.shape[1]
    x = obs.reshape(n, obs_len * 2)
    y = fut.reshape(n, pred_len * 2)
    m = np.repeat(fut_mask, 2, axis=1).astype(np.float32)  # (N, pred*2)

    mapping = load_splits(args.splits_file)
    split_of = np.array([mapping.get(str(s), "?") for s in scene_id])
    idx = {s: np.where(split_of == s)[0] for s in ("train", "val", "test")}

    x_mean = x[idx["train"]].mean(axis=0, keepdims=True).astype(np.float32)
    x_std = np.maximum(x[idx["train"]].std(axis=0, keepdims=True), 1e-6).astype(np.float32)
    y_mean, y_std = fit_masked_standardizer(y[idx["train"]], m[idx["train"]])

    def xs(a): return (a - x_mean) / x_std
    def ys(a): return (a - y_mean) / y_std
    def y_inv(a): return a * y_std + y_mean

    model = MLP(x.shape[1], y.shape[1], int(args.hidden_dim), int(args.depth), rng)
    opt = Adam(model.params, lr=float(args.lr))

    def eval_split(split: str) -> dict:
        sel = idx[split]
        yhat, _ = model.forward(xs(x[sel]))
        pred = y_inv(yhat).reshape(len(sel), pred_len, 2)
        ade, fde, path = masked_ade_fde_np(pred, fut[sel], fut_mask[sel])
        return summarize_np(ade, fde, path, agent_type[sel])

    best_val_ade = float("inf")
    best_epoch = -1
    best_params = None
    patience_left = int(args.patience)

    for epoch in range(1, int(args.epochs) + 1):
        order = idx["train"].copy()
        rng.shuffle(order)
        epoch_loss, n_seen = 0.0, 0
        for start in range(0, len(order), int(args.batch_size)):
            mb = order[start : start + int(args.batch_size)]
            xb, yb, mb_mask = xs(x[mb]), ys(y[mb]), m[mb]
            yhat, cache = model.forward(xb)
            diff = (yhat - yb) * mb_mask
            denom = max(float(mb_mask.sum()), 1.0)
            loss = float((diff * diff).sum() / denom)
            dy = (2.0 / denom) * diff
            grads = model.backward(cache, dy.astype(np.float32),
                                   weight_decay=float(args.weight_decay))
            opt.step(model.params, grads)
            epoch_loss += loss * len(mb)
            n_seen += len(mb)
        epoch_loss /= max(1, n_seen)

        val = eval_split("val")
        improved = val["ade"] < best_val_ade - 1e-6
        if improved:
            best_val_ade = val["ade"]
            best_epoch = epoch
            best_params = {k: v.copy() for k, v in model.params.items()}
            patience_left = int(args.patience)
        else:
            patience_left -= 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(f"epoch {epoch:4d} | train_mse={epoch_loss:.6f} "
                  f"| val_ADE={val['ade']:.4f} | val_FDE={val['fde']:.4f}"
                  + ("  (best)" if improved else ""))
        if patience_left <= 0:
            print(f"early stopping at epoch {epoch} (best_epoch={best_epoch})")
            break

    assert best_params is not None
    model.params = best_params
    val_summary = eval_split("val")
    test_summary = eval_split("test")

    Path(args.save_best).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.save_best, **model.params,
                        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    metrics = {
        "model": "MLP",
        "input": str(args.input),
        "splits_file": str(args.splits_file),
        "obs_len": int(obs_len),
        "pred_len": int(pred_len),
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "val": val_summary,
        "test": test_summary,
        "hyper": {"hidden_dim": int(args.hidden_dim), "depth": int(args.depth),
                  "lr": float(args.lr), "batch_size": int(args.batch_size),
                  "weight_decay": float(args.weight_decay), "epochs": int(args.epochs),
                  "patience": int(args.patience)},
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": "Benchmark v2: canonical per-agent windows, frozen splits, masked "
                "loss/metrics, y-standardization over valid entries only.",
    }
    Path(args.save_metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_metrics_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"val  ADE={val_summary['ade']:.4f} FDE={val_summary['fde']:.4f}")
    print(f"test ADE={test_summary['ade']:.4f} FDE={test_summary['fde']:.4f}")
    print(f"saved metrics: {args.save_metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
