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


def _node_metrics(pred: np.ndarray, fut: np.ndarray) -> tuple[float, float, float] | None:
    """Masked min-over-samples ADE/FDE + GT path length for one node@timestep.

    pred: (S, ph, 2) samples; fut: (ph, 2) with NaN beyond availability."""
    valid = np.isfinite(fut).all(axis=1)          # (ph,)
    if not valid.any():
        return None
    fut_f = np.where(valid[:, None], fut, 0.0)
    err = np.linalg.norm(pred - fut_f[None], axis=-1)   # (S, ph)
    n_valid = float(valid.sum())
    ade_s = (err * valid[None]).sum(axis=1) / n_valid   # (S,)
    last = int(np.max(np.arange(len(valid)) * valid))
    fde_s = err[:, last]
    both = valid[1:] & valid[:-1]
    # fut is NaN-padded; NaN * 0 == NaN, so mask with where() before summing.
    step_d = np.linalg.norm(fut_f[1:] - fut_f[:-1], axis=-1)
    path = float(np.where(both, step_d, 0.0).sum()) if len(valid) > 1 else 0.0
    return float(ade_s.min()), float(fde_s.min()), path


def evaluate(model_dir: Path, device: str, k_best_of: int, split: str,
             type_map: dict) -> dict:
    sys.path.insert(0, str(TRAJECTRON))
    import dill
    import torch
    from model.model_registrar import ModelRegistrar
    from model.trajectron import Trajectron

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

    results: dict = {"checkpoint_iter": it}
    with torch.no_grad():
        for mode, kwargs, n_samples in (
            ("best_of_k", dict(z_mode=False, gmm_mode=False, full_dist=False), k_best_of),
            ("most_likely", dict(z_mode=True, gmm_mode=True, full_dist=False), 1),
        ):
            ades, fdes, paths, classes = [], [], [], []
            for scene in env.scenes:
                timesteps = np.arange(scene.timesteps)
                predictions = stg.predict(
                    scene, timesteps, PRED_LEN,
                    num_samples=n_samples,
                    min_history_timesteps=OBS_LEN,
                    min_future_timesteps=MIN_FUT,
                    **kwargs,
                )
                scene_types = type_map.get(scene.name, {})
                for ts, node_preds in predictions.items():
                    for node, pred in node_preds.items():
                        fut = node.get(np.array([ts + 1, ts + PRED_LEN]),
                                       {"position": ["x", "y"]}, padding=np.nan)
                        arr = np.asarray(pred).reshape(-1, PRED_LEN, 2)
                        m = _node_metrics(arr, np.asarray(fut, dtype=np.float64))
                        if m is None:
                            continue
                        ades.append(m[0]); fdes.append(m[1]); paths.append(m[2])
                        classes.append(scene_types.get(str(node.id), ""))
            ade = np.asarray(ades); fde = np.asarray(fdes)
            path = np.asarray(paths); types = np.asarray(classes, dtype=object)
            results[mode] = {
                "ade": float(ade.mean()) if len(ade) else None,
                "fde": float(fde.mean()) if len(fde) else None,
                "n_node_timesteps": int(len(ade)),
                "num_samples": n_samples,
                "per_class": per_class_summary(ade, fde, path, types) if len(ade) else None,
            }
            print(f"{split}/{mode}: ADE={results[mode]['ade']} FDE={results[mode]['fde']} "
                  f"(n={len(ade)})")
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

    val_results = evaluate(model_dir, args.device, args.k_best_of, "val", type_map)
    test_results = evaluate(model_dir, args.device, args.k_best_of, "test", type_map)

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
