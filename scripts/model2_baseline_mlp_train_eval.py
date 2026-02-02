"""Train/evaluate a simple single-agent neural baseline (MLP) on centered IDD-3D windows.

This script avoids heavyweight DL dependencies by implementing a small MLP and Adam
optimizer in NumPy.

Input NPZ must contain:
- obs: (N, obs_len, 2)
- fut: (N, pred_len, 2)
- scene_id: (N,)

Outputs:
- metrics JSON (ADE/FDE)
- optional predictions NPZ
- optional best checkpoint NPZ
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def invert(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(x.dtype)


class MLP:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        depth: int,
        rng: np.random.Generator,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)

        # He init for ReLU layers
        self.params: dict[str, np.ndarray] = {}

        prev = self.input_dim
        for i in range(depth):
            w = rng.standard_normal((prev, self.hidden_dim), dtype=np.float32) * np.float32(
                math.sqrt(2.0 / prev)
            )
            b = np.zeros((self.hidden_dim,), dtype=np.float32)
            self.params[f"W{i+1}"] = w
            self.params[f"b{i+1}"] = b
            prev = self.hidden_dim

        w_out = rng.standard_normal((prev, self.output_dim), dtype=np.float32) * np.float32(
            math.sqrt(2.0 / prev)
        )
        b_out = np.zeros((self.output_dim,), dtype=np.float32)
        self.params["W_out"] = w_out
        self.params["b_out"] = b_out

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cache: dict[str, np.ndarray] = {"x0": x}
        h = x
        for i in range(self.depth):
            w = self.params[f"W{i+1}"]
            b = self.params[f"b{i+1}"]
            z = h @ w + b
            h = relu(z)
            cache[f"z{i+1}"] = z
            cache[f"h{i+1}"] = h

        y = h @ self.params["W_out"] + self.params["b_out"]
        cache["y"] = y
        return y, cache

    def backward(
        self,
        cache: dict[str, np.ndarray],
        dy: np.ndarray,
        weight_decay: float,
    ) -> dict[str, np.ndarray]:
        grads: dict[str, np.ndarray] = {}

        h_last = cache[f"h{self.depth}"] if self.depth > 0 else cache["x0"]
        grads["W_out"] = h_last.T @ dy / dy.shape[0]
        grads["b_out"] = dy.mean(axis=0)

        if weight_decay > 0:
            grads["W_out"] = grads["W_out"] + np.float32(weight_decay) * self.params["W_out"]

        dh = dy @ self.params["W_out"].T

        for i in range(self.depth, 0, -1):
            z = cache[f"z{i}"]
            h_prev = cache["x0"] if i == 1 else cache[f"h{i-1}"]

            dz = dh * relu_grad(z)
            grads[f"W{i}"] = h_prev.T @ dz / dz.shape[0]
            grads[f"b{i}"] = dz.mean(axis=0)

            if weight_decay > 0:
                grads[f"W{i}"] = grads[f"W{i}"] + np.float32(weight_decay) * self.params[f"W{i}"]

            dh = dz @ self.params[f"W{i}"].T

        return grads


class Adam:
    def __init__(
        self,
        params: dict[str, np.ndarray],
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        lr = self.lr

        for k, p in params.items():
            g = grads[k]
            m = self.m[k] = b1 * self.m[k] + (1.0 - b1) * g
            v = self.v[k] = b2 * self.v[k] + (1.0 - b2) * (g * g)

            m_hat = m / (1.0 - b1**self.t)
            v_hat = v / (1.0 - b2**self.t)
            params[k] = p - (lr * m_hat) / (np.sqrt(v_hat) + self.eps)


def compute_ade_fde(pred_fut: np.ndarray, gt_fut: np.ndarray) -> tuple[float, float]:
    # pred/gt: (N, T, 2)
    err = np.linalg.norm(pred_fut - gt_fut, axis=-1)  # (N, T)
    ade = float(err.mean())
    fde = float(err[:, -1].mean())
    return ade, fde


def make_scene_split(
    scene_ids: np.ndarray,
    val_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(scene_ids)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * float(val_frac))))
    val_scenes = set(unique[:n_val].tolist())
    is_val = np.array([sid in val_scenes for sid in scene_ids], dtype=bool)
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]
    return train_idx, val_idx


def make_random_split(n: int, val_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * float(val_frac))))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def fit_standardizer(x: np.ndarray, eps: float = 1e-6) -> Standardizer:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.maximum(std, np.float32(eps))
    return Standardizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


def iter_minibatches(rng: np.random.Generator, indices: np.ndarray, batch_size: int):
    order = indices.copy()
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def save_npz_checkpoint(path: Path, model: MLP, meta: dict) -> None:
    arrays = {k: v for k, v in model.params.items()}
    arrays["__meta_json__"] = np.array([json.dumps(meta)], dtype=object)
    np.savez_compressed(path, **arrays)


def load_npz_checkpoint(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    ckpt = np.load(path, allow_pickle=True)
    meta = json.loads(str(ckpt["__meta_json__"][0])) if "__meta_json__" in ckpt else {}
    params = {k: ckpt[k] for k in ckpt.files if not k.startswith("__")}
    return params, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/windows_obs20_pred60.npz")
    ap.add_argument("--out-dir", default="outputs/baselines")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split-by-scene", action="store_true", default=True)
    ap.add_argument("--no-split-by-scene", dest="split_by_scene", action="store_false")

    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--patience", type=int, default=30)

    ap.add_argument("--standardize", action="store_true", default=True)
    ap.add_argument("--no-standardize", dest="standardize", action="store_false")

    ap.add_argument("--save-best", default="outputs/baselines/mlp_best_obs20_pred60.npz")
    ap.add_argument("--save-metrics-json", default="outputs/baselines/mlp_metrics_obs20_pred60.json")
    ap.add_argument("--save-pred-npz", default="outputs/baselines/mlp_pred_val_obs20_pred60.npz")

    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = set_seed(args.seed)

    data = np.load(in_path, allow_pickle=True)
    obs = np.asarray(data["obs"], dtype=np.float32)
    fut = np.asarray(data["fut"], dtype=np.float32)
    scene_id = data["scene_id"] if "scene_id" in data else None

    n, obs_len, _ = obs.shape
    _, pred_len, _ = fut.shape

    x = obs.reshape(n, obs_len * 2).astype(np.float32)
    y = fut.reshape(n, pred_len * 2).astype(np.float32)

    if args.split_by_scene and scene_id is not None:
        train_idx, val_idx = make_scene_split(scene_id, args.val_frac, rng)
        split_kind = "scene"
    else:
        train_idx, val_idx = make_random_split(n, args.val_frac, rng)
        split_kind = "random"

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    x_scaler = fit_standardizer(x_train) if args.standardize else None
    y_scaler = fit_standardizer(y_train) if args.standardize else None

    if x_scaler is not None:
        x_train_s = x_scaler.apply(x_train)
        x_val_s = x_scaler.apply(x_val)
    else:
        x_train_s, x_val_s = x_train, x_val

    if y_scaler is not None:
        y_train_s = y_scaler.apply(y_train)
        y_val_s = y_scaler.apply(y_val)
    else:
        y_train_s, y_val_s = y_train, y_val

    model = MLP(
        input_dim=x_train_s.shape[1],
        output_dim=y_train_s.shape[1],
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        rng=rng,
    )

    opt = Adam(model.params, lr=args.lr)

    best_val_ade = float("inf")
    best_epoch = -1
    patience_left = int(args.patience)

    for epoch in range(1, int(args.epochs) + 1):
        # Train
        epoch_loss = 0.0
        n_seen = 0
        for mb in iter_minibatches(rng, train_idx, int(args.batch_size)):
            xb = x[mb]
            yb = y[mb]
            if x_scaler is not None:
                xb = x_scaler.apply(xb)
            if y_scaler is not None:
                yb = y_scaler.apply(yb)

            yhat, cache = model.forward(xb)
            diff = (yhat - yb).astype(np.float32)
            loss = float((diff * diff).mean())

            dy = (2.0 / diff.size) * diff  # d/dyhat mean(diff^2)
            grads = model.backward(cache, dy.astype(np.float32), weight_decay=float(args.weight_decay))
            opt.step(model.params, grads)

            epoch_loss += loss * len(mb)
            n_seen += len(mb)

        epoch_loss /= max(1, n_seen)

        # Val
        yhat_val_s, _ = model.forward(x_val_s)
        yhat_val = y_scaler.invert(yhat_val_s) if y_scaler is not None else yhat_val_s

        pred_val = yhat_val.reshape(len(val_idx), pred_len, 2)
        gt_val = y_val.reshape(len(val_idx), pred_len, 2)
        val_ade, val_fde = compute_ade_fde(pred_val, gt_val)

        improved = val_ade < best_val_ade - 1e-6
        if improved:
            best_val_ade = val_ade
            best_epoch = epoch
            patience_left = int(args.patience)

            meta = {
                "when": datetime.now().isoformat(timespec="seconds"),
                "input": str(in_path).replace("\\", "/"),
                "split": split_kind,
                "seed": int(args.seed),
                "val_frac": float(args.val_frac),
                "obs_len": int(obs_len),
                "pred_len": int(pred_len),
                "hidden_dim": int(args.hidden_dim),
                "depth": int(args.depth),
                "lr": float(args.lr),
                "batch_size": int(args.batch_size),
                "weight_decay": float(args.weight_decay),
                "standardize": bool(args.standardize),
                "best_epoch": int(best_epoch),
                "best_val_ade": float(best_val_ade),
                "val_fde_at_best": float(val_fde),
                "train_size": int(len(train_idx)),
                "val_size": int(len(val_idx)),
            }

            save_npz_checkpoint(Path(args.save_best), model, meta)
        else:
            patience_left -= 1

        if epoch == 1 or epoch % 10 == 0 or improved or patience_left <= 0:
            print(
                f"epoch {epoch:4d} | train_mse={epoch_loss:.6f} | val_ADE={val_ade:.4f} | val_FDE={val_fde:.4f}"
                + ("  (best)" if improved else "")
            )

        if patience_left <= 0:
            print(f"early stopping at epoch {epoch} (best_epoch={best_epoch}, best_val_ADE={best_val_ade:.4f})")
            break

    # Load best checkpoint for final reporting + preds
    best_params, best_meta = load_npz_checkpoint(Path(args.save_best))
    for k in model.params.keys():
        model.params[k] = best_params[k].astype(np.float32)

    # Train metrics
    yhat_train_s, _ = model.forward(x_train_s)
    yhat_train = y_scaler.invert(yhat_train_s) if y_scaler is not None else yhat_train_s
    pred_train = yhat_train.reshape(len(train_idx), pred_len, 2)
    gt_train = y_train.reshape(len(train_idx), pred_len, 2)
    train_ade, train_fde = compute_ade_fde(pred_train, gt_train)

    # Val metrics
    yhat_val_s, _ = model.forward(x_val_s)
    yhat_val = y_scaler.invert(yhat_val_s) if y_scaler is not None else yhat_val_s
    pred_val = yhat_val.reshape(len(val_idx), pred_len, 2)
    gt_val = y_val.reshape(len(val_idx), pred_len, 2)
    val_ade, val_fde = compute_ade_fde(pred_val, gt_val)

    metrics = {
        "model": "MLP",
        "input": str(in_path).replace("\\", "/"),
        "obs_len": int(obs_len),
        "pred_len": int(pred_len),
        "count_total": int(n),
        "split": split_kind,
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "epochs_requested": int(args.epochs),
        "best_epoch": int(best_meta.get("best_epoch", best_epoch)),
        "standardize": bool(args.standardize),
        "weight_decay": float(args.weight_decay),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "train": {"ade": float(train_ade), "fde": float(train_fde)},
        "val": {"ade": float(val_ade), "fde": float(val_fde)},
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": "Single-agent MLP mapping flattened obs->future positions (centered windows).",
    }

    Path(args.save_metrics_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save val predictions
    np.savez_compressed(
        args.save_pred_npz,
        idx_val=val_idx.astype(np.int32),
        pred_fut=pred_val.astype(np.float32),
        gt_fut=gt_val.astype(np.float32),
        obs=obs[val_idx].astype(np.float32),
        scene_id=scene_id[val_idx] if scene_id is not None else np.array([], dtype=object),
    )

    print(
        f"MLP results | split={split_kind} | train ADE={train_ade:.4f} FDE={train_fde:.4f}"
        f" | val ADE={val_ade:.4f} FDE={val_fde:.4f}"
    )
    print(f"saved metrics: {args.save_metrics_json}")
    print(f"saved best:    {args.save_best}")
    print(f"saved preds:   {args.save_pred_npz}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
