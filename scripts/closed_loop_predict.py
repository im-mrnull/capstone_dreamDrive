"""Every predictor's output on the closed-loop dense windows (all seeds).

Writes outputs/closed_loop/preds/seed{s}/{key}.npz, rows aligned with
outputs/closed_loop/dense_windows_test.npz (sequence order, agents in order):
  pred     (N,60,2) float32  the deployable single trajectory
  samples  (N,10,60,2) float16  multi-future models only (K-mode: its 10 modes;
           LED: first 10 of its 20 leapfrog samples; AgentFormer / Trajectron++:
           10 prior samples), as in the safety scorecard
  probs    (N,10) float32  K-mode / multi-future transformer only: each mode's
           predicted probability (softmax of the mode scorer), same order as samples
  scene_id, t0, agent_id  for alignment checks
Trajectron++ is run in its own interpreter (`--models trajectronpp`): it and
AgentFormer both ship a top-level `model` package.

Usage:
    python scripts/closed_loop_predict.py --models cv mlp stgat transformer mfp led agentformer
    python scripts/closed_loop_predict.py --models trajectronpp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from conformal_wrapper import cv_predict
from dd_data import CanonicalSeqDataset, collate_pad
from safety_scorecard import OUT as BENCH, REPO, _load_script, mlp_world

# run configuration (overridable from the command line; see main)
DENSE = REPO / "outputs" / "closed_loop" / "dense_windows_test.npz"
PRED_DIR = REPO / "outputs" / "closed_loop" / "preds"
SPLITS = str(REPO / "splits" / "scene_splits.json")
SPLIT = "test"
CKPT_DIR = BENCH
K_SET = 10
MODELS = ("cv", "mlp", "stgat", "transformer", "mfp", "mftf", "led", "agentformer", "trajectronpp")
TPP_DATA = None        # --tpp-data / --tpp-metrics-dir: a fold's Trajectron++ export and metrics
TPP_METRICS = None


def rows_meta() -> dict:
    d = np.load(DENSE, allow_pickle=True)
    sse = d["seq_start_end"]
    sid = np.concatenate([[str(d["scene_id"][q])] * int(b - a) for q, (a, b) in enumerate(sse)])
    t0 = np.concatenate([[int(d["t0"][q])] * int(b - a) for q, (a, b) in enumerate(sse)])
    return {"scene_id": sid.astype(object), "t0": t0.astype(np.int32), "agent_id": d["agent_id"],
            "obs": d["obs_traj"].transpose(0, 2, 1).astype(np.float64)}


def save(key: str, seed: int, pred, samples, meta, probs=None) -> None:
    out = PRED_DIR / f"seed{seed}" / f"{key}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = {} if samples is None else {"samples": np.asarray(samples, dtype=np.float16)}
    if probs is not None:
        extra["probs"] = np.asarray(probs, dtype=np.float32)
    np.savez_compressed(out, pred=np.asarray(pred, dtype=np.float32), **extra,
                        scene_id=meta["scene_id"], t0=meta["t0"], agent_id=meta["agent_id"])
    print(f"seed {seed} {key}: {len(pred)} rows" + ("" if samples is None else f", samples {np.shape(samples)[1]}")
          + f" -> {out}", flush=True)


def dataset():
    return CanonicalSeqDataset(str(DENSE), SPLIT, SPLITS, rot_aug=False, seed=0)


def run_seq_models(keys, seed, device, meta):
    dp = _load_script("dump_predictions.py")
    loaders = {"stgat": dp.load_stgat, "transformer": dp.load_transformer, "mfp": dp.load_mfp,
               "mftf": dp.load_mftf}
    loader = DataLoader(dataset(), batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_pad)
    for key in keys:
        fn = loaders[key](device, seed, CKPT_DIR)
        pred = dp.flatten(loader, fn, device, "test")[0]
        samples = dp.flatten_samples(loader, fn.samples_fn, device) if hasattr(fn, "samples_fn") else None
        probs = flatten_probs(loader, fn.probs_fn, device) if hasattr(fn, "probs_fn") else None
        save(key, seed, pred, samples, meta, probs)


def flatten_probs(loader, probs_fn, device) -> list:
    """Per real agent (flatten order): mode probabilities (K,) from probs_fn(b) -> (B,M,K)."""
    out = []
    for batch in loader:
        b = batch.to(device)
        with torch.no_grad():
            p = probs_fn(b).cpu().numpy()
        am = b.agent_mask.cpu().numpy()
        out += [p[i, j] for i in range(am.shape[0]) for j in range(am.shape[1]) if am[i, j]]
    return out


def run_led(seed, device, meta):
    from dd_led import Diffusion, build_batch, build_models, predict_batch
    ck = torch.load(CKPT_DIR / f"led_ckpt_seed{seed}.pt", map_location=device, weights_only=False)
    a = ck["args"]
    core, init = build_models(20, 60, a["k"])
    core.load_state_dict(ck["core"]); init.load_state_dict(ck["init"])
    core, init = core.to(device).eval(), init.to(device).eval()
    diff = Diffusion(device, steps=a["diffusion_steps"])
    torch.manual_seed(seed)
    ds, P, S = dataset(), [], []
    with torch.no_grad():
        for b0 in range(0, len(ds), 8):
            items = [ds[i] for i in range(b0, min(b0 + 8, len(ds)))]
            b = build_batch([(o, p, f, t) for o, p, _, _, f, t in items], a["traj_scale"], device)
            ml, samp = predict_batch(core, init, diff, b, a["traj_scale"], a["n_tau"])
            P.append(ml.cpu().numpy()); S.append(samp[:, :K_SET].cpu().numpy())
    save("led", seed, np.concatenate(P), np.concatenate(S), meta)


def run_agentformer(seed, device, meta):
    lat = _load_script("safety_latency.py")
    from dd_agentformer import seq_to_af_data, predict_seq
    ck = torch.load(CKPT_DIR / f"agentformer_ckpt_seed{seed}.pt", map_location=device, weights_only=False)
    model, cfg = lat.build_agentformer(ck, device)
    torch.manual_seed(seed)
    ds, P, S = dataset(), [], []
    with torch.no_grad():
        for i in range(len(ds)):
            obs, pred, _, _, fm, _ = ds[i]
            data = seq_to_af_data(obs, pred, fm, cfg.traj_scale, stride=cfg.stride)
            ml, samp = predict_seq(model, data, nz=cfg.nz, sample_k=K_SET, traj_scale=cfg.traj_scale,
                                   stride=cfg.stride)
            P.append(ml.cpu().numpy()); S.append(samp.cpu().numpy())
    save("agentformer", seed, np.concatenate(P), np.concatenate(S), meta)


def run_trajectronpp(seed, device_str, meta):
    import dill
    ev = _load_script("eval_trajectronpp_canonical.py")
    if TPP_DATA is not None:
        ev.TPP_DATA = Path(TPP_DATA)
    if TPP_METRICS is not None:
        ev.OUT = Path(TPP_METRICS)
    sys.path.insert(0, str(ev.TRAJECTRON))
    from model.model_registrar import ModelRegistrar
    from model.trajectron import Trajectron
    exp7 = ev._load_module("exp7", ev.SCRIPTS / "7_train_eval_trajectronpp.py")
    model_dir = ev.model_dir_for(seed)
    hyper = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    reg = ModelRegistrar(str(model_dir), device_str)
    reg.load_models(exp7.latest_checkpoint_iter(model_dir))
    stg = Trajectron(reg, hyper, None, device_str)
    torch.manual_seed(seed); np.random.seed(seed)
    with open(ev.TPP_DATA / f"{SPLIT}.pkl", "rb") as f:
        env = dill.load(f, encoding="latin1")
    stg.set_environment(env)
    stg.set_annealing_params()
    want = {}
    for i, (s, t0) in enumerate(zip(meta["scene_id"], meta["t0"])):
        want.setdefault(str(s), set()).add(int(t0) + 19)
    ml, samp = {}, {}
    kw = dict(min_history_timesteps=1, min_future_timesteps=0, full_dist=False)
    with torch.no_grad():
        for scene in env.scenes:
            if scene.name not in want:
                continue
            scene.calculate_scene_graph(env.attention_radius, hyper["edge_addition_filter"],
                                        hyper["edge_removal_filter"])
            ts = np.array(sorted(want[scene.name]))
            for t, nodes in stg.predict(scene, ts, 60, num_samples=1, z_mode=True, gmm_mode=True, **kw).items():
                for node, arr in nodes.items():
                    ml[(scene.name, int(t), str(node.id))] = np.asarray(arr).reshape(-1, 60, 2)[0]
            for t, nodes in stg.predict(scene, ts, 60, num_samples=K_SET, z_mode=False, gmm_mode=False, **kw).items():
                for node, arr in nodes.items():
                    samp[(scene.name, int(t), str(node.id))] = np.asarray(arr).reshape(-1, 60, 2)
    keys = [(str(s), int(t0) + 19, str(a)) for s, t0, a in zip(meta["scene_id"], meta["t0"], meta["agent_id"])]
    cv = meta["obs"][:, -1:, :] + cv_predict(meta["obs"], 5)       # no T++ node (track gap): CV
    has = np.array([k in ml for k in keys])
    pred = np.stack([ml[k] if h else cv[j] for j, (k, h) in enumerate(zip(keys, has))])
    sp = np.stack([samp[k] if h else np.repeat(cv[j][None], K_SET, 0) for j, (k, h) in enumerate(zip(keys, has))])
    print(f"  trajectronpp seed {seed}: {(~has).sum()} rows without a T++ node -> CV", flush=True)
    save("trajectronpp", seed, pred, sp, meta)


def main() -> int:
    global DENSE, PRED_DIR, SPLITS, SPLIT, CKPT_DIR, TPP_DATA, TPP_METRICS
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=MODELS, default=[m for m in MODELS if m != "trajectronpp"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dense", default=str(DENSE), help="dense windows NPZ (closed_loop_windows.py)")
    ap.add_argument("--split", default=SPLIT, help="split of the dense windows' scenes")
    ap.add_argument("--splits-file", default=SPLITS)
    ap.add_argument("--ckpt-dir", default=str(BENCH), help="model checkpoints (+ mlp_best_seed<k>.npz)")
    ap.add_argument("--out-dir", default=str(PRED_DIR))
    ap.add_argument("--tpp-data", default=None, help="Trajectron++ export dir (fold)")
    ap.add_argument("--tpp-metrics-dir", default=None, help="dir with trajectronpp_seed<k>.json (fold)")
    args = ap.parse_args()
    DENSE, PRED_DIR, SPLITS, SPLIT, CKPT_DIR = (Path(args.dense), Path(args.out_dir), args.splits_file,
                                                args.split, Path(args.ckpt_dir))
    TPP_DATA, TPP_METRICS = args.tpp_data, args.tpp_metrics_dir
    if "trajectronpp" in args.models and len(args.models) > 1:
        raise SystemExit("run trajectronpp on its own (module clash with AgentFormer)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = rows_meta()
    for seed in args.seeds:
        if "cv" in args.models:
            save("cv", seed, meta["obs"][:, -1:, :] + cv_predict(meta["obs"], 5), None, meta)
        if "mlp" in args.models:
            w = np.load(CKPT_DIR / f"mlp_best_seed{seed}.npz")
            save("mlp", seed, mlp_world(meta["obs"], {k: w[k] for k in w.files}), None, meta)
        seq = [k for k in ("stgat", "transformer", "mfp", "mftf") if k in args.models]
        if seq:
            run_seq_models(seq, seed, device, meta)
        if "led" in args.models:
            run_led(seed, device, meta)
        if "agentformer" in args.models:
            run_agentformer(seed, device, meta)
        if "trajectronpp" in args.models:
            run_trajectronpp(seed, "cuda:0" if device.type == "cuda" else "cpu", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
