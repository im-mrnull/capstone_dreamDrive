"""Export ego-compensated (world-frame) IDD-3D scenes to Trajectron++ format.

Input:  outputs/multiagent_world_v1/scenes/*.npz  (from script 4 on world CSVs)
Output: outputs/trajectronpp_world/{train.pkl, val.pkl, config.json}

Protocol notes (recorded in the pickles' companion config):
- dt = 0.1 s (10 fps IDD-3D). This was the root cause of the earlier scale
  bug: the upstream default of 0.4 s must never be used for this data.
- All agents are exported as PEDESTRIAN nodes (point-mass SingleIntegrator
  dynamics) exactly like the upstream ETH/UCY processing; per-class dynamics
  (e.g. Unicycle for vehicles) is future work and would need heading.
- Per agent, only the longest contiguous observed run is kept (Trajectron++
  node data must be gap-free).
- Scene-level 80/20 split with numpy default_rng(seed=0) over sorted scene
  names — the same protocol as the other world-frame baselines.

Requires the patched upstream repo on disk (see scripts/patch_trajectronpp.py).

Usage:
    python scripts/6_export_trajectronpp_world.py --dt 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TRAJECTRON = REPO / "third_party" / "trajectron-plus-plus" / "trajectron"

STANDARDIZATION = {
    "PEDESTRIAN": {
        "position": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
        "velocity": {"x": {"mean": 0, "std": 2}, "y": {"mean": 0, "std": 2}},
        "acceleration": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
    }
}


def longest_contiguous_run(valid: np.ndarray) -> tuple[int, int]:
    """Return (start, end_exclusive) of the longest True run; (0, 0) if none."""
    best = (0, 0)
    start = None
    for i, v in enumerate(valid.tolist() + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-dir", default="outputs/multiagent_world_v1/scenes")
    ap.add_argument("--out-dir", default="outputs/trajectronpp_world")
    ap.add_argument("--dt", type=float, default=0.1,
                    help="frame period in seconds; 0.1 for 10 fps IDD-3D (DO NOT use 0.4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--attention-radius", type=float, default=20.0)
    ap.add_argument("--min-node-len", type=int, default=2)
    args = ap.parse_args()

    if not TRAJECTRON.is_dir():
        raise SystemExit("Patched Trajectron++ repo required. Run:\n"
                         "  git clone --depth 1 https://github.com/StanfordASL/Trajectron-plus-plus.git third_party/trajectron-plus-plus\n"
                         "  python scripts/patch_trajectronpp.py")
    sys.path.insert(0, str(TRAJECTRON))

    import dill
    import pandas as pd
    from environment import Environment, Scene, Node, derivative_of

    scene_paths = sorted(Path(args.scenes_dir).glob("*.npz"))
    if not scene_paths:
        raise SystemExit(f"no scene NPZs in {args.scenes_dir}")

    rng = np.random.default_rng(int(args.seed))
    names = np.array([p.stem for p in scene_paths])
    shuffled = names.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(names) * float(args.val_frac))))
    val_set = set(shuffled[:n_val].tolist())

    data_columns = pd.MultiIndex.from_product([["position", "velocity", "acceleration"], ["x", "y"]])

    def build_env(paths: list[Path]) -> tuple[object, dict]:
        env = Environment(node_type_list=["PEDESTRIAN"], standardization=STANDARDIZATION)
        env.attention_radius = {
            (env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): float(args.attention_radius)
        }
        scenes = []
        n_nodes = 0
        for p in paths:
            d = np.load(p, allow_pickle=True)
            pos = np.asarray(d["pos"], dtype=np.float64)   # (T, A, 2)
            mask = np.asarray(d["mask"], dtype=bool)        # (T, A)
            agent_ids = d["agent_id"]
            T, A, _ = pos.shape
            scene = Scene(timesteps=T, dt=float(args.dt), name=p.stem, aug_func=None)
            for a in range(A):
                s, e = longest_contiguous_run(mask[:, a])
                if e - s < int(args.min_node_len):
                    continue
                xy = pos[s:e, a, :]
                vx = derivative_of(xy[:, 0], scene.dt)
                vy = derivative_of(xy[:, 1], scene.dt)
                ax = derivative_of(vx, scene.dt)
                ay = derivative_of(vy, scene.dt)
                node_df = pd.DataFrame(
                    {("position", "x"): xy[:, 0], ("position", "y"): xy[:, 1],
                     ("velocity", "x"): vx, ("velocity", "y"): vy,
                     ("acceleration", "x"): ax, ("acceleration", "y"): ay},
                    columns=data_columns,
                )
                node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=str(agent_ids[a]),
                            data=node_df, first_timestep=s)
                scene.nodes.append(node)
                n_nodes += 1
            if scene.nodes:
                scenes.append(scene)
        env.scenes = scenes
        return env, {"scenes": len(scenes), "nodes": n_nodes}

    train_paths = [p for p in scene_paths if p.stem not in val_set]
    val_paths = [p for p in scene_paths if p.stem in val_set]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, paths in (("train", train_paths), ("val", val_paths)):
        env, st = build_env(paths)
        with open(out_dir / f"{split}.pkl", "wb") as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
        stats[split] = st
        print(f"{split}: {st['scenes']} scenes, {st['nodes']} nodes -> {out_dir / (split + '.pkl')}")

    # Derive the training config from upstream defaults; only the horizon,
    # history length and bookkeeping differ.
    base_cfg = json.loads((REPO / "third_party" / "trajectron-plus-plus" / "config" / "config.json")
                          .read_text(encoding="utf-8"))
    base_cfg["prediction_horizon"] = 60   # 6.0 s at 10 fps
    base_cfg["maximum_history_length"] = 19  # 2.0 s at 10 fps (19 past + current)
    base_cfg["batch_size"] = 128
    (out_dir / "config.json").write_text(json.dumps(base_cfg, indent=2), encoding="utf-8")

    meta = {
        "dt": float(args.dt),
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "attention_radius_m": float(args.attention_radius),
        "node_type_treatment": "all agents as PEDESTRIAN (SingleIntegrator point-mass)",
        "source_scenes_dir": str(args.scenes_dir),
        "splits": stats,
        "val_scenes": sorted(val_set),
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote config + export_meta -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
