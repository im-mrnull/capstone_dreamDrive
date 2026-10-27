"""Trajectron++ on the canonical windows (matched evaluation population).

scripts/7_train_eval_trajectronpp.py scores Trajectron++ at every valid
node-timestep (23,570 test instances), the upstream protocol. Every other model
in benchmark v2 is scored on the canonical windows (2,177 test agent-windows in
outputs/canonical_world_obs20_pred60.npz). This script re-scores the trained
T++ checkpoints on exactly those windows: for canonical window (scene, t0,
agent) it takes T++'s prediction for that node at timestep t0+19 (the last
observed frame) and scores it against the canonical future and mask with the
same metric code as the other models (dd_data.masked_ade_fde_np).

T++ nodes are the longest contiguous run of each track (script 6), so a window
whose track has a gap inside the observation gets T++'s prediction from the
history it does have (min_history_timesteps=1 here; the count is reported as
`n_short_history`). Prediction does not depend on the future, so
min_future_timesteps=0.

Usage:
    python scripts/eval_trajectronpp_canonical.py                 # seeds 0, 1, 2
    python scripts/eval_trajectronpp_canonical.py --seeds 0 --device cpu
    python scripts/eval_trajectronpp_canonical.py --hetero        # unicycle variant
    # every seed's dump incl. the K samples (safety scorecard / seed-pooled bootstrap):
    python scripts/eval_trajectronpp_canonical.py --dump-dir outputs/benchmark_v2/seed_dumps
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
OUT = REPO / "outputs" / "benchmark_v2"
TPP_DATA = REPO / "outputs" / "trajectronpp_world"
TAG = "trajectronpp"   # metrics-file prefix; --hetero switches both
TRAJECTRON = REPO / "third_party" / "trajectron-plus-plus" / "trajectron"
OBS_LEN, PRED_LEN = 20, 60
SPLITS_FILE = REPO / "splits" / "scene_splits.json"   # --splits-file (cross-validation folds)

sys.path.insert(0, str(SCRIPTS))
from dd_data import masked_ade_fde_np, summarize_np  # noqa: E402
from dd_split import load_splits  # noqa: E402
from model1_baseline_cv_eval import compute_cv_predictions  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model_dir_for(seed: int) -> Path:
    """The seed's run dir, by basename (metrics JSONs carry another machine's path)."""
    m = json.loads((OUT / f"{TAG}_seed{seed}.json").read_text(encoding="utf-8"))
    d = TPP_DATA / "logs" / Path(m["model_dir"].replace("\\", "/")).name
    if not d.is_dir():
        raise SystemExit(f"model dir for seed {seed} not found: {d}")
    return d


def windows(split: str) -> list[tuple[str, int, int]]:
    """(scene_id, t0, row) for every canonical agent-window of `split`, in NPZ order."""
    c = np.load(REPO / "outputs" / "canonical_world_obs20_pred60.npz", allow_pickle=True)
    mapping = load_splits(SPLITS_FILE)
    out = []
    for q, (a, b) in enumerate(c["seq_start_end"]):
        sid = str(c["scene_id"][q])
        if mapping.get(sid) == split:
            out += [(sid, int(c["t0"][q]), i) for i in range(a, b)]
    return out


def evaluate(seed: int, device: str, k: int) -> tuple[dict, dict]:
    sys.path.insert(0, str(TRAJECTRON))
    import dill
    import torch
    from model.model_registrar import ModelRegistrar
    from model.trajectron import Trajectron

    exp7 = _load_module("exp7", SCRIPTS / "7_train_eval_trajectronpp.py")
    model_dir = model_dir_for(seed)
    hyper = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    reg = ModelRegistrar(str(model_dir), device)
    reg.load_models(exp7.latest_checkpoint_iter(model_dir))
    stg = Trajectron(reg, hyper, None, device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    c = np.load(REPO / "outputs" / "canonical_world_obs20_pred60.npz", allow_pickle=True)
    results, dump = {"seed": seed, "model_dir": model_dir.name}, {}
    for split in ("val", "test"):
        with open(TPP_DATA / f"{split}.pkl", "rb") as f:
            env = dill.load(f, encoding="latin1")
        stg.set_environment(env)
        stg.set_annealing_params()
        wins = windows(split)
        by_scene: dict[str, set[int]] = {}
        for sid, t0, _ in wins:
            by_scene.setdefault(sid, set()).add(t0 + OBS_LEN - 1)
        ml, samp = {}, {}
        with torch.no_grad():
            for scene in env.scenes:
                if scene.name not in by_scene:
                    continue
                scene.calculate_scene_graph(env.attention_radius, hyper["edge_addition_filter"],
                                            hyper["edge_removal_filter"])
                ts = np.array(sorted(by_scene[scene.name]))
                kw = dict(min_history_timesteps=1, min_future_timesteps=0)
                p_ml = stg.predict(scene, ts, PRED_LEN, num_samples=1, z_mode=True,
                                   gmm_mode=True, full_dist=False, **kw)
                p_s = stg.predict(scene, ts, PRED_LEN, num_samples=k, z_mode=False,
                                  gmm_mode=False, full_dist=False, **kw)
                for t, nodes in p_ml.items():
                    for node, arr in nodes.items():
                        ml[(scene.name, int(t), str(node.id))] = np.asarray(arr).reshape(-1, PRED_LEN, 2)[0]
                for t, nodes in p_s.items():
                    for node, arr in nodes.items():
                        samp[(scene.name, int(t), str(node.id))] = np.asarray(arr).reshape(-1, PRED_LEN, 2)

        rows = [i for _, _, i in wins]
        keys = [(sid, t0 + OBS_LEN - 1, str(c["agent_id"][i])) for sid, t0, i in wins]
        # A track with a gap keeps only its longest run in the T++ export, so some
        # windows have no T++ node at their prediction frame: constant velocity
        # (the benchmark's CV, vel_k=5) stands in, for every sample.
        obs = c["obs_traj"][rows].transpose(0, 2, 1).astype(np.float64)    # (N, 20, 2)
        cv = obs[:, -1:, :] + compute_cv_predictions(obs, PRED_LEN, 5)
        has = np.array([kk in ml for kk in keys])
        pred = np.stack([ml[kk] if h else cv[j] for j, (kk, h) in enumerate(zip(keys, has))]
                        ).astype(np.float64)
        sp = np.stack([samp[kk] if h else np.repeat(cv[j][None], k, 0)
                       for j, (kk, h) in enumerate(zip(keys, has))]).astype(np.float64)  # (N, K, 60, 2)
        gt = c["pred_traj"][rows].transpose(0, 2, 1).astype(np.float64)    # (N, 60, 2)
        fm = c["fut_mask"][rows].astype(bool)
        types = c["agent_type"][rows].astype(object)

        a, f, path = masked_ade_fde_np(pred, gt, fm)
        ak = np.stack([masked_ade_fde_np(sp[:, j], gt, fm)[0] for j in range(sp.shape[1])], 1)
        fk = np.stack([masked_ade_fde_np(sp[:, j], gt, fm)[1] for j in range(sp.shape[1])], 1)
        k_list = [kk for kk in exp7.K_CURVE if kk <= k]
        results[split] = {"most_likely": summarize_np(a, f, path, types),
                          "best_of_k": summarize_np(ak.min(1), fk.min(1), path, types),
                          "sampling_curve": {  # same readout as script 7, on these windows
                              "k_list": k_list,
                              "ade": [float(ak[:, :kk].min(1).mean()) for kk in k_list],
                              "mean_sample_ade": float(ak.mean()),
                              "n_agents": int(len(a))},
                          "n_short_history": n_short_history(keys, env),
                          "n_cv_fallback": int((~has).sum())}
        dump[split] = {"pred": pred.astype(np.float32), "samples": sp.astype(np.float32),
                       "gt": gt.astype(np.float32),
                       "fut_mask": fm, "agent_type": types,
                       "scene_id": np.asarray([sid for sid, _, _ in wins], dtype=object)}
        print(f"seed {seed} {split}: most-likely ADE={a.mean():.4f} FDE={f.mean():.4f} "
              f"(n={len(a)}) | best-of-{k} ADE={ak.min(1).mean():.4f} FDE={fk.min(1).mean():.4f}")
    return results, dump


def n_short_history(keys: list[tuple[str, int, str]], env) -> int:
    """Windows whose T++ node starts inside the window's observation (but by its last frame)."""
    first = {(s.name, str(n.id)): n.first_timestep for s in env.scenes for n in s.nodes}
    return int(sum(first[(sid, nid)] <= t < first[(sid, nid)] + OBS_LEN - 1
                   for sid, t, nid in keys))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--dump-seed", type=int, default=0,
                    help="seed whose most-likely predictions are saved for the conformal wrapper")
    ap.add_argument("--dump-dir", default=None,
                    help="also write every seed's dump, with the K samples, to "
                         "<dump-dir>/seed<s>/preds_trajectronpp_canonical.npz")
    ap.add_argument("--data-dir", default=None, help="T++ export dir (default outputs/trajectronpp_world)")
    ap.add_argument("--metrics-dir", default=None,
                    help="where <tag>_seed<k>.json are read and <tag>_canonical_seed<k>.json written")
    ap.add_argument("--splits-file", default=None, help="split file of the T++ export")
    ap.add_argument("--hetero", action="store_true",
                    help="the VEHICLE=Unicycle variant (outputs/trajectronpp_world_hetero)")
    args = ap.parse_args()

    global TPP_DATA, TAG, OUT, SPLITS_FILE
    if args.hetero:
        TPP_DATA, TAG = REPO / "outputs" / "trajectronpp_world_hetero", "trajectronpp_hetero"
    if args.data_dir:
        TPP_DATA = Path(args.data_dir)
    if args.metrics_dir:
        OUT = Path(args.metrics_dir)
    if args.splits_file:
        SPLITS_FILE = Path(args.splits_file)
    model = ("Trajectron++ (upstream, patched; VEHICLE=Unicycle for wheeled classes, "
             "PEDESTRIAN=point-mass otherwise)" if args.hetero else
             "Trajectron++ (upstream, patched; all agents as point-mass PEDESTRIAN nodes)")

    for seed in args.seeds:
        res, dump = evaluate(seed, args.device, args.k)
        payload = {
            "model": model,
            "protocol": "canonical windows: prediction at t0+19 for each canonical agent-window, "
                        "scored on the canonical future and mask (same population as every "
                        "other benchmark-v2 model)",
            "data": "outputs/canonical_world_obs20_pred60.npz",
            "splits_file": str(SPLITS_FILE),
            "eval_k": args.k,
            "device": args.device,
            **res,
        }
        out = OUT / f"{TAG}_canonical_seed{seed}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved metrics: {out}")
        targets = []
        if seed == args.dump_seed and not args.hetero:
            targets.append((OUT / "preds_trajectronpp_canonical.npz", ("pred", "gt", "fut_mask")))
        if args.dump_dir and not args.hetero:
            targets.append((Path(args.dump_dir) / f"seed{seed}" / "preds_trajectronpp_canonical.npz",
                            ("pred", "samples", "gt", "fut_mask")))
        for npz, keys in targets:
            v, t = dump["val"], dump["test"]
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz,
                **{k: np.concatenate([v[k], t[k]]) for k in keys},
                agent_type=np.concatenate([v["agent_type"], t["agent_type"]]).astype(object),
                split=np.asarray(["val"] * len(v["pred"]) + ["test"] * len(t["pred"]), dtype=object),
                scene_id=np.concatenate([v["scene_id"], t["scene_id"]]).astype(object),
                model="Trajectron++ (most-likely, canonical windows)")
            print(f"saved most-likely predictions: {npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
