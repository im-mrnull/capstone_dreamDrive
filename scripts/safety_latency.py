"""Inference latency of every safety-scorecard package (tie-break 2 of the rule in
docs/wiki/safety-scorecard.md).

Unit: wall-clock time to predict every agent of one canonical test sequence (one
scene snapshot), including all samples for set packages and the host->GPU copy,
after 5 warm-up sequences; median and p95 over the test sequences. Neural models
use their seed-0 checkpoint on the GPU; CV and the MLP run in numpy on the CPU.
Trajectron++ predicts every node present at that timestep (its scene graph is
built beforehand and not timed), so its unit can cover a few more agents.

Usage:
    python scripts/safety_latency.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch

import dump_predictions as dp
from conformal_wrapper import cv_predict
from dd_data import CanonicalSeqDataset, collate_pad
from safety_scorecard import OUT, PACKAGES, REPO, mlp_world, _load_script

SPLITS = str(REPO / "splits" / "scene_splits.json")
DATA = str(REPO / "outputs" / "canonical_world_obs20_pred60.npz")
K_SET = 10
WARMUP = 5


def timed(fn, items, device) -> dict:
    ms = []
    for i, it in enumerate(items):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            fn(it)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= WARMUP:
            ms.append(1000.0 * (time.perf_counter() - t))
    return {"median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)), "n": len(ms)}


def numpy_fns() -> dict:
    w = np.load(OUT / "mlp_best_seed0.npz")
    weights = {k: w[k] for k in w.files}
    obs = lambda it: it[0].transpose(0, 2, 1).astype(np.float64)   # noqa: E731
    return {"CV (vel_k=5)": lambda it: obs(it)[:, -1:, :] + cv_predict(obs(it), 5),
            "MLP": lambda it: mlp_world(obs(it), weights)}


def seq_model_fns(device) -> dict:
    stgat, tf, mfp = dp.load_stgat(device, 0), dp.load_transformer(device, 0), dp.load_mfp(device, 0)
    batch = lambda it: collate_pad([it]).to(device)                   # noqa: E731
    return {"ST-GAT (MHA)": lambda it: stgat(batch(it)),
            "Map-free transformer": lambda it: tf(batch(it)),
            "K-mode (top-1)": lambda it: mfp(batch(it)),
            "K-mode set (10 modes)": lambda it: mfp.samples_fn(batch(it))}


def led_fns(device) -> dict:
    from dd_led import Diffusion, build_batch, build_models, leapfrog_loc
    ck = torch.load(OUT / "led_ckpt_seed0.pt", map_location=device, weights_only=False)
    a = ck["args"]
    core, init = build_models(20, 60, a["k"])
    core.load_state_dict(ck["core"]); init.load_state_dict(ck["init"])
    core, init = core.to(device).eval(), init.to(device).eval()
    diff = Diffusion(device, steps=a["diffusion_steps"])

    def run(it, single: bool):
        obs, pred, _, _, fm, at = it
        b = build_batch([(obs, pred, fm, at)], a["traj_scale"], device)
        loc, mean, _ = leapfrog_loc(init, b["past"], b["mask"])
        start = mean[:, None] if single else loc[:, :K_SET]
        y = diff.denoise(core, b["past"], b["mask"], start, a["n_tau"],
                         noise_scale=0.0 if single else 1e-5)
        return y * a["traj_scale"] + b["last"][:, None, None]          # (N,K,T,2)
    return {"LED (mean-denoised)": lambda it: run(it, True),
            "LED set (10 samples)": lambda it: run(it, False)}


def agentformer_fns(device) -> dict:
    m6 = _load_script("model6_agentformer_train_eval.py")
    from dd_agentformer import build_map_free_cfg, keyframe_indices, predict_seq, seq_to_af_data, upsample_keyframes
    ck = torch.load(OUT / "agentformer_ckpt_seed0.pt", map_location=device, weights_only=False)
    a = ck["args"]
    obs_idx, fut_idx = keyframe_indices(20, 60, a["stride"])
    cfg = build_map_free_cfg(
        len(obs_idx), len(fut_idx), nz=a["nz"], sample_k=a["sample_loss_k"],
        tf_model_dim=a["tf_model_dim"], tf_ff_dim=a["tf_ff_dim"], tf_nhead=a["tf_nhead"],
        tf_dropout=a["tf_dropout"], nlayer=a["nlayer"],
        out_mlp_dim=None if a["no_out_mlp"] else (512, 256),
        rand_rot_scene=bool(a["rot_aug"]), traj_scale=a["traj_scale"])
    cfg.stride = int(a["stride"])
    model = m6.build_model(cfg, device)
    model.load_state_dict(ck["model"]); model.eval()

    def most_likely(it):   # predict_seq's z=0 rollout without the extra sample decode
        obs, pred, _, _, fm, _ = it
        data = seq_to_af_data(obs, pred, fm, cfg.traj_scale, stride=cfg.stride)
        model.set_data(data)
        model.context_encoder(model.data)
        z0 = torch.zeros(model.data["agent_num"], cfg.nz, device=model.device)
        model.future_decoder(model.data, mode="infer", sample_num=1, z=z0)
        ml = model.data["infer_dec_motion"][:, 0] * cfg.traj_scale
        last = torch.stack([p[-1] for p in data["pre_motion_3D"]]).to(ml) * cfg.traj_scale
        return upsample_keyframes(ml, last, cfg.stride)

    def sample_set(it):
        obs, pred, _, _, fm, _ = it
        data = seq_to_af_data(obs, pred, fm, cfg.traj_scale, stride=cfg.stride)
        return predict_seq(model, data, nz=cfg.nz, sample_k=K_SET, traj_scale=cfg.traj_scale,
                           stride=cfg.stride)
    return {"AgentFormer (most-likely)": most_likely, "AgentFormer set (10 samples)": sample_set}


def trajectronpp_timings(device_str: str) -> dict:
    import dill
    ev = _load_script("eval_trajectronpp_canonical.py")
    sys.path.insert(0, str(ev.TRAJECTRON))
    from model.model_registrar import ModelRegistrar
    from model.trajectron import Trajectron
    exp7 = ev._load_module("exp7", ev.SCRIPTS / "7_train_eval_trajectronpp.py")
    model_dir = ev.model_dir_for(0)
    hyper = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    reg = ModelRegistrar(str(model_dir), device_str)
    reg.load_models(exp7.latest_checkpoint_iter(model_dir))
    stg = Trajectron(reg, hyper, None, device_str)
    with open(ev.TPP_DATA / "test.pkl", "rb") as f:
        env = dill.load(f, encoding="latin1")
    stg.set_environment(env)
    stg.set_annealing_params()
    scenes = {s.name: s for s in env.scenes}
    jobs = sorted({(sid, t0 + ev.OBS_LEN - 1) for sid, t0, _ in ev.windows("test")})
    for name in {sid for sid, _ in jobs}:
        scenes[name].calculate_scene_graph(env.attention_radius, hyper["edge_addition_filter"],
                                           hyper["edge_removal_filter"])
    kw = dict(min_history_timesteps=1, min_future_timesteps=0, full_dist=False)
    device = torch.device(device_str)
    items = [(scenes[s], np.array([t])) for s, t in jobs]
    single = lambda it: stg.predict(it[0], it[1], 60, num_samples=1, z_mode=True, gmm_mode=True, **kw)  # noqa: E731
    sample = lambda it: stg.predict(it[0], it[1], 60, num_samples=K_SET, z_mode=False, gmm_mode=False, **kw)  # noqa: E731
    return {"Trajectron++ (most-likely)": timed(single, items, device),
            "Trajectron++ set (10 samples)": timed(sample, items, device)}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "--tpp-only" in sys.argv:
        # AgentFormer and Trajectron++ both ship a top-level `model` package, so
        # Trajectron++ is timed in its own interpreter.
        print(json.dumps(trajectronpp_timings("cuda:0" if device.type == "cuda" else "cpu")))
        return 0
    torch.manual_seed(0)
    ds = CanonicalSeqDataset(DATA, "test", SPLITS, rot_aug=False, seed=0)
    items = [ds[i] for i in range(len(ds))]
    out = {}
    for name, fn in numpy_fns().items():
        out[name] = timed(fn, items, torch.device("cpu")) | {"device": "cpu (numpy)"}
    for builder in (seq_model_fns, led_fns, agentformer_fns):
        for name, fn in builder(device).items():
            out[name] = timed(fn, items, device) | {"device": str(device)}
    tpp = subprocess.run([sys.executable, __file__, "--tpp-only"], capture_output=True, text=True,
                         check=True).stdout.strip().splitlines()[-1]
    for name, r in json.loads(tpp).items():
        out[name] = r | {"device": str(device)}
    missing = {n for n, _, _ in PACKAGES} - set(out)
    assert not missing, f"no latency for {missing}"
    for name, r in out.items():
        print(f"{name:30s} median {r['median_ms']:7.2f} ms  p95 {r['p95_ms']:7.2f} ms  ({r['device']})")
    payload = {"unit": "ms to predict every agent of one canonical test sequence (one scene "
                       "snapshot), incl. all samples for set packages; seed-0 models",
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               "n_sequences": len(items), "warmup": WARMUP, "packages": out,
               "when": datetime.now().isoformat(timespec="seconds")}
    path = OUT / "safety_latency.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
