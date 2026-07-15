"""Train and evaluate Trajectron++ on world-frame IDD-3D — end to end.

Wraps the (patched) upstream trainer, then evaluates the final checkpoint on
the val split and writes an artifact-grade metrics JSON that
scripts/make_tables.py picks up. No number is reported anywhere else.

Prerequisites:
    git clone --depth 1 https://github.com/StanfordASL/Trajectron-plus-plus.git third_party/trajectron-plus-plus
    python scripts/patch_trajectronpp.py
    python scripts/6_export_trajectronpp_world.py --dt 0.1
    pip install dill orjson ncls pathos seaborn matplotlib scipy tensorboard tqdm

Usage:
    python scripts/7_train_eval_trajectronpp.py --epochs 100
    python scripts/7_train_eval_trajectronpp.py --eval-only --model-dir <log_dir/model_dir>
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
DATA_DIR = REPO / "outputs" / "trajectronpp_world"
LOG_DIR = REPO / "outputs" / "trajectronpp_world" / "logs"
METRICS_JSON = REPO / "outputs" / "baselines_world" / "trajectronpp_metrics_world_obs20_pred60.json"

OBS_LEN = 19   # past timesteps required (2.0 s minus the current frame)
PRED_LEN = 60  # 6.0 s at 10 fps


def train(epochs: int, device: str, log_tag: str) -> Path:
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
        "--eval_every", "5",
        "--vis_every", "100000",
        "--save_every", "1",
        "--preprocess_workers", "0",
        "--offline_scene_graph", "yes",
    ]
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


def evaluate(model_dir: Path, device: str, k_best_of: int) -> dict:
    sys.path.insert(0, str(TRAJECTRON))
    import dill
    import torch
    import evaluation  # noqa: F401  (upstream package)
    from model.model_registrar import ModelRegistrar
    from model.trajectron import Trajectron

    it = latest_checkpoint_iter(model_dir)
    hyperparams = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    registrar = ModelRegistrar(str(model_dir), device)
    registrar.load_models(it)
    stg = Trajectron(registrar, hyperparams, None, device)

    with open(DATA_DIR / "val.pkl", "rb") as f:
        env = dill.load(f, encoding="latin1")
    if env.robot_type is None and hyperparams.get("incl_robot_node"):
        env.robot_type = env.NodeType[0]
    for scene in env.scenes:
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams["edge_addition_filter"],
                                    hyperparams["edge_removal_filter"])
    stg.set_environment(env)
    stg.set_annealing_params()

    max_hl = hyperparams["maximum_history_length"]
    results = {}
    with torch.no_grad():
        for mode, kwargs, n_samples in (
            ("best_of_k", dict(z_mode=False, gmm_mode=False, full_dist=False), k_best_of),
            ("most_likely", dict(z_mode=True, gmm_mode=True, full_dist=False), 1),
        ):
            ades, fdes = [], []
            for scene in env.scenes:
                timesteps = np.arange(scene.timesteps)
                predictions = stg.predict(
                    scene, timesteps, PRED_LEN,
                    num_samples=n_samples,
                    min_history_timesteps=OBS_LEN,
                    min_future_timesteps=PRED_LEN,
                    **kwargs,
                )
                if not predictions:
                    continue
                batch_error = evaluation.compute_batch_statistics(
                    predictions, scene.dt,
                    max_hl=max_hl, ph=PRED_LEN,
                    node_type_enum=env.NodeType,
                    map=None, kde=False,
                    best_of=True, prune_ph_to_future=False,
                )
                for node_type in batch_error:
                    ades.extend(batch_error[node_type]["ade"])
                    fdes.extend(batch_error[node_type]["fde"])
            results[mode] = {
                "ade": float(np.mean(ades)) if ades else None,
                "fde": float(np.mean(fdes)) if fdes else None,
                "num_node_timesteps": len(ades),
                "num_samples": n_samples,
            }
            print(f"{mode}: ADE={results[mode]['ade']} FDE={results[mode]['fde']} "
                  f"(n={len(ades)})")
    results["checkpoint_iter"] = it
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log-tag", default="_idd3d_world")
    ap.add_argument("--k-best-of", type=int, default=20)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--model-dir", type=str, default=None,
                    help="existing run dir (with model_registrar-*.pt) for --eval-only")
    args = ap.parse_args()

    if not (DATA_DIR / "train.pkl").exists():
        raise SystemExit("run scripts/6_export_trajectronpp_world.py first")

    if args.eval_only:
        if not args.model_dir:
            runs = sorted(LOG_DIR.glob(f"*{args.log_tag}"), key=os.path.getmtime)
            if not runs:
                raise SystemExit("no run dir found; pass --model-dir")
            model_dir = runs[-1]
        else:
            model_dir = Path(args.model_dir)
    else:
        model_dir = train(args.epochs, args.device, args.log_tag)

    results = evaluate(model_dir, args.device, args.k_best_of)

    export_meta = json.loads((DATA_DIR / "export_meta.json").read_text(encoding="utf-8"))
    payload = {
        "model": "Trajectron++ (upstream, patched; all agents as point-mass PEDESTRIAN nodes)",
        "data": "outputs/trajectronpp_world/{train,val}.pkl (world-frame, ego-compensated)",
        "obs_len_required": OBS_LEN,
        "pred_len": PRED_LEN,
        "dt": export_meta["dt"],
        "split": {"kind": "scene", "seed": export_meta["seed"], "val_frac": export_meta["val_frac"]},
        "epochs_trained": None if args.eval_only else args.epochs,
        "model_dir": str(model_dir),
        "best_of_k": results["best_of_k"],
        "most_likely": results["most_likely"],
        "checkpoint_iter": results["checkpoint_iter"],
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": "Real end-to-end run. best_of_k uses min-over-samples (oracle); "
                "most_likely is the deployable single output (z_mode+gmm_mode).",
    }
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {METRICS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
