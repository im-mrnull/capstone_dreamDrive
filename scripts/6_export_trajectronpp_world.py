"""Export ego-compensated (world-frame) IDD-3D scenes to Trajectron++ format.

Input:  outputs/multiagent_world_v1/scenes/*.npz  (from script 4 on world CSVs)
Output: <out-dir>/{train,val,test}.pkl + config.json + node_type_map.json

Two export modes:

1. Homogeneous (default, benchmark v2 baseline): all agents as PEDESTRIAN
   nodes (point-mass SingleIntegrator), exactly like upstream ETH/UCY.
   Output dir: outputs/trajectronpp_world

2. Heterogeneous (--hetero): wheeled classes become VEHICLE nodes with
   Unicycle dynamics (state gains heading °/d° and velocity/acceleration
   norms, mirroring upstream nuScenes int_ee); everything else stays
   PEDESTRIAN. IDD-3D has no usable yaw in this repo's parsed CSVs (raw
   dataset absent), so heading is derived from the smoothed velocity
   direction — held through static periods — which is upstream's own
   fallback and the defining no-slip assumption of a unicycle.
   Output dir: outputs/trajectronpp_world_hetero

Optional --augment-copies N (train split only): attaches N rotated copies
(360/N° steps) to each train scene via scene.augmented + a pickled aug_func,
enabling upstream train.py --augment (nuScenes-style rotation augmentation).

Protocol notes:
- dt = 0.1 s (10 fps IDD-3D). The upstream default of 0.4 s must never be
  used for this data (root cause of the earlier scale bug).
- Per agent, only the longest contiguous observed run is kept (Trajectron++
  node data must be gap-free).
- Scene split comes from the frozen splits/scene_splits.json; nothing is
  re-derived from an RNG.
- node_type_map.json records each node's true IDD-3D class; evaluation
  reports per-class metrics via this map regardless of modeled node type.

Requires the patched upstream repo on disk (see scripts/patch_trajectronpp.py).

Usage:
    python scripts/6_export_trajectronpp_world.py --dt 0.1
    python scripts/6_export_trajectronpp_world.py --hetero --augment-copies 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TRAJECTRON = REPO / "third_party" / "trajectron-plus-plus" / "trajectron"

# Wheeled classes modeled as VEHICLE (Unicycle) in --hetero mode. Ambiguous
# classes (Unknown*, Misc, DontCare, BicycleGroup) stay point-mass PEDESTRIAN:
# an unconstrained model is the safer prior when the class is unclear.
VEHICLE_CLASSES = {
    "Car", "TourCar", "PoliceCar", "Van", "Truck", "Bus", "Crane",
    "ConcreteTruck", "MotorcyleRider", "ScooterRider", "BicycleRider",
    "Motorcycle", "Scooter", "Bicycle",
}

STANDARDIZATION_PED_ONLY = {
    "PEDESTRIAN": {
        "position": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
        "velocity": {"x": {"mean": 0, "std": 2}, "y": {"mean": 0, "std": 2}},
        "acceleration": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
    }
}

# Mirrors upstream experiments/nuScenes/process_data.py (position std is
# overridden by the attention radius inside upstream preprocessing anyway).
STANDARDIZATION_HETERO = {
    "PEDESTRIAN": STANDARDIZATION_PED_ONLY["PEDESTRIAN"],
    "VEHICLE": {
        "position": {"x": {"mean": 0, "std": 80}, "y": {"mean": 0, "std": 80}},
        "velocity": {"x": {"mean": 0, "std": 15}, "y": {"mean": 0, "std": 15},
                     "norm": {"mean": 0, "std": 15}},
        "acceleration": {"x": {"mean": 0, "std": 4}, "y": {"mean": 0, "std": 4},
                         "norm": {"mean": 0, "std": 4}},
        "heading": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1},
                    "°": {"mean": 0, "std": np.pi}, "d°": {"mean": 0, "std": 1}},
    },
}

MOVING_SPEED = 0.5  # m/s; below this the last moving heading is held


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


def motion_heading(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Heading angle from velocity direction, held (ffill/bfill) while static."""
    speed = np.hypot(vx, vy)
    ang = np.arctan2(vy, vx)
    moving = speed > MOVING_SPEED
    if not moving.any():
        return np.zeros_like(ang)
    # forward-fill the index of the last moving step; back-fill the static prefix
    idx = np.where(moving, np.arange(len(ang)), -1)
    idx = np.maximum.accumulate(idx)
    idx = np.where(idx < 0, int(np.argmax(moving)), idx)
    return ang[idx]


def vehicle_frame(xy: np.ndarray, dt: float, derivative_of) -> dict:
    """nuScenes-style VEHICLE node data dict from world positions."""
    import pandas as pd  # noqa: F401  (kept local: caller builds the DataFrame)
    vx = derivative_of(xy[:, 0], dt)
    vy = derivative_of(xy[:, 1], dt)
    ax = derivative_of(vx, dt)
    ay = derivative_of(vy, dt)
    heading = motion_heading(vx, vy)
    heading = (heading + np.pi) % (2.0 * np.pi) - np.pi
    v = np.stack((vx, vy), axis=-1)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.0))
    return {
        ("position", "x"): xy[:, 0], ("position", "y"): xy[:, 1],
        ("velocity", "x"): vx, ("velocity", "y"): vy,
        ("velocity", "norm"): np.linalg.norm(v, axis=-1),
        ("acceleration", "x"): ax, ("acceleration", "y"): ay,
        ("acceleration", "norm"): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
        ("heading", "x"): heading_v[:, 0], ("heading", "y"): heading_v[:, 1],
        ("heading", "°"): heading,
        ("heading", "d°"): derivative_of(heading, dt, radian=True),
    }


def pedestrian_frame(xy: np.ndarray, dt: float, derivative_of) -> dict:
    vx = derivative_of(xy[:, 0], dt)
    vy = derivative_of(xy[:, 1], dt)
    return {
        ("position", "x"): xy[:, 0], ("position", "y"): xy[:, 1],
        ("velocity", "x"): vx, ("velocity", "y"): vy,
        ("acceleration", "x"): derivative_of(vx, dt),
        ("acceleration", "y"): derivative_of(vy, dt),
    }


def augment(scene):
    """aug_func pickled into the env; upstream train.py calls scene.augment()."""
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    scene_aug.map = scene.map
    return scene_aug


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-dir", default="outputs/multiagent_world_v1/scenes")
    ap.add_argument("--out-dir", default=None,
                    help="default: outputs/trajectronpp_world[_hetero]")
    ap.add_argument("--dt", type=float, default=0.1,
                    help="frame period in seconds; 0.1 for 10 fps IDD-3D (DO NOT use 0.4)")
    ap.add_argument("--splits-file", default="splits/scene_splits.json")
    ap.add_argument("--attention-radius", type=float, default=20.0,
                    help="homogeneous mode only; hetero uses nuScenes per-pair radii")
    ap.add_argument("--min-node-len", type=int, default=2)
    ap.add_argument("--hetero", action="store_true",
                    help="VEHICLE (Unicycle) nodes for wheeled classes")
    ap.add_argument("--augment-copies", type=int, default=0,
                    help="rotated scene copies for the train split (0 = off)")
    args = ap.parse_args()

    if not TRAJECTRON.is_dir():
        raise SystemExit("Patched Trajectron++ repo required. Run:\n"
                         "  git clone --depth 1 https://github.com/StanfordASL/Trajectron-plus-plus.git third_party/trajectron-plus-plus\n"
                         "  python scripts/patch_trajectronpp.py")
    sys.path.insert(0, str(TRAJECTRON))

    import dill
    import pandas as pd
    from environment import Environment, Scene, Node, derivative_of

    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO / "outputs" / ("trajectronpp_world_hetero" if args.hetero else "trajectronpp_world"))

    # Frozen split (shared with every other model in benchmark v2).
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("dd_split", Path(__file__).resolve().parent / "dd_split.py")
    dd_split = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(dd_split)
    mapping = dd_split.load_splits(args.splits_file)

    scene_paths = sorted(Path(args.scenes_dir).glob("*.npz"))
    if not scene_paths:
        raise SystemExit(f"no scene NPZs in {args.scenes_dir}")

    cols_ped = pd.MultiIndex.from_product(
        [["position", "velocity", "acceleration"], ["x", "y"]])
    cols_veh = pd.MultiIndex.from_product(
        [["position", "velocity", "acceleration", "heading"], ["x", "y"]])
    cols_veh = cols_veh.append(pd.MultiIndex.from_tuples([("heading", "°"), ("heading", "d°")]))
    cols_veh = cols_veh.append(pd.MultiIndex.from_product([["velocity", "acceleration"], ["norm"]]))

    node_type_map: dict[str, dict[str, str]] = {}

    def make_env() -> object:
        if args.hetero:
            env = Environment(node_type_list=["VEHICLE", "PEDESTRIAN"],
                              standardization=STANDARDIZATION_HETERO)
            env.attention_radius = {
                (env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 10.0,
                (env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE): 20.0,
                (env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN): 20.0,
                (env.NodeType.VEHICLE, env.NodeType.VEHICLE): 30.0,
            }
        else:
            env = Environment(node_type_list=["PEDESTRIAN"],
                              standardization=STANDARDIZATION_PED_ONLY)
            env.attention_radius = {
                (env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): float(args.attention_radius)
            }
        return env

    def build_node(env, node_type_name: str, node_id: str, xy: np.ndarray,
                   first_ts: int, dt: float, non_aug_node=None):
        if node_type_name == "VEHICLE":
            df = pd.DataFrame(vehicle_frame(xy, dt, derivative_of), columns=cols_veh)
            ntype = env.NodeType.VEHICLE
        else:
            df = pd.DataFrame(pedestrian_frame(xy, dt, derivative_of), columns=cols_ped)
            ntype = env.NodeType.PEDESTRIAN
        kwargs = {"non_aug_node": non_aug_node} if non_aug_node is not None else {}
        return Node(node_type=ntype, node_id=node_id, data=df,
                    first_timestep=first_ts, **kwargs)

    def rotate(xy: np.ndarray, alpha: float) -> np.ndarray:
        m = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return xy @ m.T

    def build_env(paths: list[Path], with_aug: bool) -> tuple[object, dict]:
        env = make_env()
        scenes = []
        n_nodes = {"VEHICLE": 0, "PEDESTRIAN": 0}
        angles = ([360.0 * i / args.augment_copies for i in range(args.augment_copies)]
                  if with_aug and args.augment_copies > 0 else [])
        for p in paths:
            d = np.load(p, allow_pickle=True)
            pos = np.asarray(d["pos"], dtype=np.float64)   # (T, A, 2)
            mask = np.asarray(d["mask"], dtype=bool)        # (T, A)
            agent_ids = d["agent_id"]
            agent_types = (
                np.asarray(d["agent_type"], dtype=object)
                if "agent_type" in d else np.asarray([""] * len(agent_ids), dtype=object)
            )
            node_type_map.setdefault(p.stem, {})
            T, A, _ = pos.shape
            scene = Scene(timesteps=T, dt=float(args.dt), name=p.stem,
                          aug_func=augment if angles else None)
            # (node_type_name, node_id, xy, first_ts) for augmentation reuse
            raw_nodes: list[tuple[str, str, np.ndarray, int]] = []
            for a in range(A):
                s, e = longest_contiguous_run(mask[:, a])
                if e - s < int(args.min_node_len):
                    continue
                cls = str(agent_types[a])
                tname = ("VEHICLE" if args.hetero and cls in VEHICLE_CLASSES
                         else "PEDESTRIAN")
                xy = pos[s:e, a, :]
                node = build_node(env, tname, str(agent_ids[a]), xy, s, scene.dt)
                scene.nodes.append(node)
                raw_nodes.append((tname, str(agent_ids[a]), xy, s))
                node_type_map[p.stem][str(agent_ids[a])] = cls
                n_nodes[tname] += 1
            if not scene.nodes:
                continue
            if angles:
                scene.augmented = []
                base_by_id = {n.id: n for n in scene.nodes}
                for ang in angles:
                    alpha = ang * np.pi / 180.0
                    sc = Scene(timesteps=T, dt=float(args.dt), name=p.stem,
                               non_aug_scene=scene)
                    for tname, nid, xy, s in raw_nodes:
                        sc.nodes.append(build_node(env, tname, nid, rotate(xy, alpha),
                                                   s, scene.dt,
                                                   non_aug_node=base_by_id[nid]))
                    scene.augmented.append(sc)
            scenes.append(scene)
        env.scenes = scenes
        return env, {"scenes": len(scenes),
                     "nodes": int(sum(n_nodes.values())),
                     "nodes_by_modeled_type": n_nodes,
                     "augment_copies": len(angles)}

    by_split: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for p in scene_paths:
        split = mapping.get(p.stem)
        if split in by_split:
            by_split[split].append(p)

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, paths in by_split.items():
        env, st = build_env(paths, with_aug=(split == "train"))
        with open(out_dir / f"{split}.pkl", "wb") as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
        stats[split] = st
        print(f"{split}: {st['scenes']} scenes, {st['nodes']} nodes "
              f"({st['nodes_by_modeled_type']}, aug x{st['augment_copies']}) "
              f"-> {out_dir / (split + '.pkl')}")

    (out_dir / "node_type_map.json").write_text(
        json.dumps(node_type_map, indent=2), encoding="utf-8")

    # Config: hetero derives from the nuScenes int_ee config (per-type dynamic/
    # state incl. Unicycle for VEHICLE); homogeneous keeps the upstream default.
    if args.hetero:
        cfg_src = (REPO / "third_party" / "trajectron-plus-plus" / "experiments"
                   / "nuScenes" / "models" / "int_ee" / "config.json")
    else:
        cfg_src = REPO / "third_party" / "trajectron-plus-plus" / "config" / "config.json"
    base_cfg = json.loads(cfg_src.read_text(encoding="utf-8"))
    base_cfg["prediction_horizon"] = 60   # 6.0 s at 10 fps
    base_cfg["maximum_history_length"] = 19  # 2.0 s at 10 fps (19 past + current)
    base_cfg["batch_size"] = 128
    (out_dir / "config.json").write_text(json.dumps(base_cfg, indent=2), encoding="utf-8")

    meta = {
        "dt": float(args.dt),
        "splits_file": str(args.splits_file),
        "hetero": bool(args.hetero),
        "augment_copies_train": int(args.augment_copies),
        "attention_radius_m": ("nuScenes per-pair (10/20/30)" if args.hetero
                               else float(args.attention_radius)),
        "node_type_treatment": (
            "VEHICLE (Unicycle, heading from motion direction) for wheeled classes; "
            "PEDESTRIAN (SingleIntegrator) otherwise; true classes in node_type_map.json"
            if args.hetero else
            "all agents as PEDESTRIAN (SingleIntegrator point-mass); "
            "true classes in node_type_map.json"),
        "vehicle_classes": sorted(VEHICLE_CLASSES) if args.hetero else [],
        "config_derived_from": str(cfg_src.relative_to(REPO)),
        "source_scenes_dir": str(args.scenes_dir),
        "splits": stats,
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote config + export_meta -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
