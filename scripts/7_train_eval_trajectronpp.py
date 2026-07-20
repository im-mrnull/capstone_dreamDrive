"""Train and evaluate Trajectron++ on world-frame IDD-3D — end to end (v2).

Wraps the (patched) upstream trainer, then evaluates the final checkpoint on
the frozen val AND test splits, writing an artifact-grade metrics JSON.

Benchmark v2 evaluation protocol (consistent with the other models):
- nodes need the full 2.0 s history (19 past steps + current) but only >= 1
  future step (partial futures are masked, not dropped);
- ADE over valid future steps; FDE at the last valid step;
- per-IDD-3D-class breakdown via node_type_map.json (all nodes are modeled as
  point-mass PEDESTRIAN, but reported under their true class);
- oracle best-of-K (min over samples) and deployable most-likely mode.

Prerequisites:
    git clone --depth 1 https://github.com/StanfordASL/Trajectron-plus-plus.git third_party/trajectron-plus-plus
    python scripts/patch_trajectronpp.py
    python scripts/6_export_trajectronpp_world.py --dt 0.1
    pip install dill orjson pathos seaborn matplotlib scipy tensorboard tqdm

Usage:
    python scripts/7_train_eval_trajectronpp.py --epochs 100 --seed 0
    python scripts/7_train_eval_trajectronpp.py --eval-only --model-dir <run_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TRAJECTRON = REPO / "third_party" / "trajectron-plus-plus" / "trajectron"
# Set from --data-dir in main(); defaults preserve the benchmark v2 baseline.
DATA_DIR = REPO / "outputs" / "trajectronpp_world"
LOG_DIR = DATA_DIR / "logs"

OBS_LEN = 19   # past timesteps required (2.0 s minus the current frame)
PRED_LEN = 60  # 6.0 s at 10 fps
MIN_FUT = 1    # >= 1 future step; shorter futures are masked, not dropped

sys.path.insert(0, str(REPO / "scripts"))
from dd_data import per_class_summary  # noqa: E402


def train(epochs: int, device: str, log_tag: str, seed: int,
          augment: bool = False) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "train.py",
        "--conf", str(DATA_DIR / "config.json"),
        "--data_dir", str(DATA_DIR),
        "--train_data_dict", "train.pkl",
        "--eval_data_dict", "val.pkl",
        "--log_dir", str(LOG_DIR),
        "--log_tag", log_tag,
        "--train_epochs", str(epochs),
        "--device", device,
        "--seed", str(seed),
        "--eval_every", "5",
        "--vis_every", "100000",
        "--save_every", "1",
        "--preprocess_workers", "0",
        "--offline_scene_graph", "yes",
    ]
    if augment:
        cmd.append("--augment")
    print("launching:", " ".join(cmd))
    subprocess.run(cmd, cwd=TRAJECTRON, check=True)
    runs = sorted(LOG_DIR.glob(f"*{log_tag}"), key=os.path.getmtime)
    if not runs:
        raise SystemExit(f"no run directory matching *{log_tag} under {LOG_DIR}")
    return runs[-1]


def latest_checkpoint_iter(model_dir: Path) -> int:
    its = []
    for p in model_dir.glob("model_registrar-*.pt"):
        try:
            its.append(int(p.stem.split("-")[-1]))
        except ValueError:
            pass
    if not its:
        raise SystemExit(f"no model_registrar-*.pt in {model_dir}")
    return max(its)


# Sample counts for the oracle best-of-K decomposition curve (largest must be
# <= --k-best-of, the number of samples actually drawn).
K_CURVE = (1, 2, 5, 10, 20)


def _node_sample_errors(pred: np.ndarray, fut: np.ndarray):
    """Per-sample masked ADE/FDE + GT path length for one node@timestep.

    pred: (S, ph, 2) samples; fut: (ph, 2) with NaN beyond availability.
    Returns (ade_s (S,), fde_s (S,), path) or None if no valid future step."""
    valid = np.isfinite(fut).all(axis=1)          # (ph,)
    if not valid.any():
        return None
    fut_f = np.where(valid[:, None], fut, 0.0)
    err = np.linalg.norm(pred - fut_f[None], axis=-1)   # (S, ph)
    n_valid = float(valid.sum())
    ade_s = (err * valid[None]).sum(axis=1) / n_valid   # (S,)
    last = int(np.max(np.arange(len(valid)) * valid))
    fde_s = err[:, last]                                 # (S,)
    both = valid[1:] & valid[:-1]
    # fut is NaN-padded; NaN * 0 == NaN, so mask with where() before summing.
    step_d = np.linalg.norm(fut_f[1:] - fut_f[:-1], axis=-1)
    path = float(np.where(both, step_d, 0.0).sum()) if len(valid) > 1 else 0.0
    return ade_s, fde_s, path


def _node_metrics(pred: np.ndarray, fut: np.ndarray) -> tuple[float, float, float] | None:
    """Masked min-over-samples ADE/FDE + GT path length for one node@timestep."""
    r = _node_sample_errors(pred, fut)
    if r is None:
        return None
    ade_s, fde_s, path = r
    return float(ade_s.min()), float(fde_s.min()), path


def best_of_curve(ade_s: np.ndarray, k_list=K_CURVE) -> list[float]:
    """Prefix best-of-k ADE from per-sample ADEs (samples are i.i.d. draws).

    best-of-k uses the first k samples; monotone non-increasing in k."""
    return [float(ade_s[:min(k, len(ade_s))].min()) for k in k_list]


def _curve_per_class(curve: np.ndarray, mean_s: np.ndarray, types: np.ndarray,
                     k_list=K_CURVE, min_class_n: int = 50) -> dict:
    """Per-class best-of-1, best-of-max, and mean-sample ADE (curve endpoints)."""
    out = {}
    for c in np.unique(types):
        if str(c) == "":
            continue
        mask = types == c
        out[str(c)] = {
            "n": int(mask.sum()),
            "best_of_1": float(curve[mask, 0].mean()),
            "best_of_max": float(curve[mask, -1].mean()),
            "mean_sample": float(mean_s[mask].mean()),
        }
    return {"classes": out, "min_class_n": min_class_n, "k_max": int(k_list[-1])}


def evaluate(model_dir: Path, device: str, k_best_of: int, split: str,
             type_map: dict, seed: int = 0, dump_preds: bool = False) -> dict:
    sys.path.insert(0, str(TRAJECTRON))
    import dill
    import torch
    from model.model_registrar import ModelRegistrar
    from model.trajectron import Trajectron

    # Sampling below is stochastic; seed so the best-of-K curve is reproducible.
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    it = latest_checkpoint_iter(model_dir)
    hyperparams = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    registrar = ModelRegistrar(str(model_dir), device)
    registrar.load_models(it)
    stg = Trajectron(registrar, hyperparams, None, device)

    with open(DATA_DIR / f"{split}.pkl", "rb") as f:
        env = dill.load(f, encoding="latin1")
    for scene in env.scenes:
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams["edge_addition_filter"],
                                    hyperparams["edge_removal_filter"])
    stg.set_environment(env)
    stg.set_annealing_params()

    def run_mode(kwargs: dict, n_samples: int, collect: bool = False):
        """Predict over all scenes; return per-node (ade_s, fde_s, path, class).

        With collect=True also gathers the deployable single trajectory, GT and
        validity mask per node-timestep (for the conformal predictions dump)."""
        ade_s_all, fde_s_all, paths, classes = [], [], [], []
        dpred, dgt, dmask = [], [], []
        for scene in env.scenes:
            timesteps = np.arange(scene.timesteps)
            predictions = stg.predict(
                scene, timesteps, PRED_LEN, num_samples=n_samples,
                min_history_timesteps=OBS_LEN, min_future_timesteps=MIN_FUT, **kwargs)
            scene_types = type_map.get(scene.name, {})
            for ts, node_preds in predictions.items():
                for node, pred in node_preds.items():
                    fut = node.get(np.array([ts + 1, ts + PRED_LEN]),
                                   {"position": ["x", "y"]}, padding=np.nan)
                    fut64 = np.asarray(fut, dtype=np.float64)
                    arr = np.asarray(pred).reshape(-1, PRED_LEN, 2)
                    r = _node_sample_errors(arr, fut64)
                    if r is None:
                        continue
                    ade_s_all.append(r[0]); fde_s_all.append(r[1]); paths.append(r[2])
                    classes.append(scene_types.get(str(node.id), ""))
                    if collect:
                        valid = np.isfinite(fut64).all(axis=1)
                        dpred.append(arr[0].astype(np.float32))
                        dgt.append(np.where(valid[:, None], fut64, 0.0).astype(np.float32))
                        dmask.append(valid)
        base = (ade_s_all, fde_s_all, np.asarray(paths), np.asarray(classes, dtype=object))
        if collect:
            return base + (np.asarray(dpred), np.asarray(dgt), np.asarray(dmask))
        return base

    results: dict = {"checkpoint_iter": it}
    with torch.no_grad():
        # Sampling readout: draw k_best_of samples ONCE; oracle best-of-K is the
        # min, and the best-of-{1..K} curve + mean-sample come from the same draw.
        ade_s_all, fde_s_all, path, types = run_mode(
            dict(z_mode=False, gmm_mode=False, full_dist=False), k_best_of)
        ade = np.asarray([a.min() for a in ade_s_all])
        fde = np.asarray([f.min() for f in fde_s_all])
        results["best_of_k"] = {
            "ade": float(ade.mean()) if len(ade) else None,
            "fde": float(fde.mean()) if len(fde) else None,
            "n_node_timesteps": int(len(ade)),
            "num_samples": k_best_of,
            "per_class": per_class_summary(ade, fde, path, types) if len(ade) else None,
        }
        if len(ade_s_all):
            curve = np.asarray([best_of_curve(a) for a in ade_s_all])   # (N, |K_CURVE|)
            mean_s = np.asarray([float(a.mean()) for a in ade_s_all])   # (N,)
            results["sampling_curve"] = {
                "k_list": list(K_CURVE),
                "ade": [float(curve[:, i].mean()) for i in range(len(K_CURVE))],
                "mean_sample_ade": float(mean_s.mean()),
                "n_node_timesteps": int(len(ade)),
                "per_class": _curve_per_class(curve, mean_s, types),
            }
            print(f"{split}/curve: " + " ".join(
                f"K{k}={v:.3f}" for k, v in zip(K_CURVE, results['sampling_curve']['ade']))
                + f" mean_sample={results['sampling_curve']['mean_sample_ade']:.3f}")

        # Most-likely readout: single argmax latent + GMM mode (deployable).
        ml_out = run_mode(dict(z_mode=True, gmm_mode=True, full_dist=False), 1,
                          collect=dump_preds)
        ade_s_all, fde_s_all, path, types = ml_out[:4]
        ade = np.asarray([a.min() for a in ade_s_all])
        fde = np.asarray([f.min() for f in fde_s_all])
        results["most_likely"] = {
            "ade": float(ade.mean()) if len(ade) else None,
            "fde": float(fde.mean()) if len(fde) else None,
            "n_node_timesteps": int(len(ade)),
            "num_samples": 1,
            "per_class": per_class_summary(ade, fde, path, types) if len(ade) else None,
        }
        if dump_preds:
            results["_pred_dump"] = {
                "pred": ml_out[4], "gt": ml_out[5], "fut_mask": ml_out[6],
                "agent_type": types}
        print(f"{split}/best_of_k: ADE={results['best_of_k']['ade']:.4f} | "
              f"most_likely: ADE={results['most_likely']['ade']:.4f}")
    return results


def main() -> int:
    global DATA_DIR, LOG_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log-tag", default="_idd3d_world_v2")
    ap.add_argument("--k-best-of", type=int, default=20)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--model-dir", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                    help="export dir from script 6 (use the _hetero dir for "
                         "VEHICLE/unicycle runs)")
    ap.add_argument("--augment", action="store_true",
                    help="pass --augment to upstream train.py (needs an export "
                         "made with --augment-copies > 0)")
    ap.add_argument("--dump-preds", default=None,
                    help="also save most-likely predictions (val+test) to this NPZ "
                         "for the conformal wrapper")
    ap.add_argument("--save-metrics-json",
                    default=str(REPO / "outputs" / "benchmark_v2" / "trajectronpp_metrics.json"))
    args = ap.parse_args()

    DATA_DIR = Path(args.data_dir)
    if not DATA_DIR.is_absolute():
        DATA_DIR = REPO / DATA_DIR  # upstream train.py runs with cwd=TRAJECTRON
    LOG_DIR = DATA_DIR / "logs"

    if not (DATA_DIR / "train.pkl").exists():
        raise SystemExit("run scripts/6_export_trajectronpp_world.py first")

    type_map = json.loads((DATA_DIR / "node_type_map.json").read_text(encoding="utf-8"))

    if args.eval_only:
        if args.model_dir:
            model_dir = Path(args.model_dir)
        else:
            runs = sorted(LOG_DIR.glob(f"*{args.log_tag}"), key=os.path.getmtime)
            if not runs:
                raise SystemExit("no run dir found; pass --model-dir")
            model_dir = runs[-1]
    else:
        model_dir = train(args.epochs, args.device, args.log_tag, int(args.seed),
                          augment=bool(args.augment))

    dp = bool(args.dump_preds)
    val_results = evaluate(model_dir, args.device, args.k_best_of, "val", type_map,
                           seed=int(args.seed), dump_preds=dp)
    test_results = evaluate(model_dir, args.device, args.k_best_of, "test", type_map,
                            seed=int(args.seed), dump_preds=dp)

    if dp:
        vd = val_results.pop("_pred_dump"); td = test_results.pop("_pred_dump")
        np.savez_compressed(
            args.dump_preds,
            pred=np.concatenate([vd["pred"], td["pred"]]).astype(np.float32),
            gt=np.concatenate([vd["gt"], td["gt"]]).astype(np.float32),
            fut_mask=np.concatenate([vd["fut_mask"], td["fut_mask"]]).astype(bool),
            agent_type=np.concatenate([vd["agent_type"], td["agent_type"]]).astype(object),
            split=np.asarray(["val"] * len(vd["pred"]) + ["test"] * len(td["pred"]),
                             dtype=object),
            model="Trajectron++ (most-likely)")
        print(f"saved predictions dump -> {args.dump_preds}")

    export_meta = json.loads((DATA_DIR / "export_meta.json").read_text(encoding="utf-8"))
    hetero = bool(export_meta.get("hetero", False))
    payload = {
        "model": ("Trajectron++ (upstream, patched; VEHICLE=Unicycle for wheeled classes, "
                  "PEDESTRIAN=point-mass otherwise)" if hetero else
                  "Trajectron++ (upstream, patched; all agents as point-mass PEDESTRIAN nodes)"),
        "data": f"{DATA_DIR.name}/{{train,val,test}}.pkl (world-frame, ego-compensated)",
        "augment": bool(args.augment),
        "obs_len_required": OBS_LEN,
        "pred_len": PRED_LEN,
        "min_future_steps": MIN_FUT,
        "dt": export_meta["dt"],
        "splits_file": export_meta.get("splits_file", "splits/scene_splits.json"),
        "seed": int(args.seed),
        "epochs_trained": None if args.eval_only else args.epochs,
        "model_dir": str(model_dir),
        "val": val_results,
        "test": test_results,
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": "Benchmark v2: frozen splits; partial futures masked (>=1 step); "
                "FDE at last valid step; per-class via node_type_map.json. "
                "best_of_k is an oracle upper bound; most_likely is deployable.",
    }
    out = Path(args.save_metrics_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
