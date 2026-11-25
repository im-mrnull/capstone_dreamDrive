"""Trajectron++ export of a chaos-gap dataset, so that one export dir serves training
(scripts/7_train_eval_trajectronpp.py --data-dir), the AV2 in-domain check and the IDD zero-shot
score (scripts/eval_trajectronpp_canonical.py --data-dir --canonical).

    train, val  one Trajectron++ scene per AV2 sequence (one scenario), built from the chaos-gap
                windows themselves, so Trajectron++ trains on the same agents as every other
                model. Each agent's node is its 20 observed frames plus its future up to the first
                missing frame (AV2 futures end early but have no holes), placed at frame t0 of a
                (t0 + 80)-frame scene: the canonical window's last observed frame, t0 + 19, is
                the scene timestep the canonical evaluator predicts from. A masked future step
                (stored as 0 in the canonical file) never enters a node.
    test        the IDD-3D scenes of the existing IDD export, copied byte for byte: its 'val'
                pickle during development, its 'test' pickle (--idd-test-split test) only for the
                pre-registered runs. The zero-shot score therefore uses exactly the scenes, nodes
                and CV fallback of the IDD-trained Trajectron++ rows.

Everything else matches the homogeneous IDD export (scripts/6_export_trajectronpp_world.py), which
the IDD-trained rows used: every agent a point-mass PEDESTRIAN node (no model in the study takes
the class as input; true classes go to node_type_map.json), dt = 0.1 s, its standardization, a
20 m attention radius, and its config.json, copied, so the hyperparameters are identical.

Usage:
    python scripts/cg_av2_trajectronpp.py --name av2_matched_dev
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from dd_split import load_splits

REPO = Path(__file__).resolve().parent.parent
TRAJECTRON = REPO / "third_party" / "trajectron-plus-plus" / "trajectron"
CG = REPO / "outputs" / "chaos_gap"
IDD_EXPORT = REPO / "outputs" / "trajectronpp_world"
OBS_LEN, PRED_LEN = 20, 60
DT, ATTENTION_RADIUS = 0.1, 20.0                                        # as the IDD export


def export6():
    """scripts/6_export_trajectronpp_world.py (its file name is not importable)."""
    spec = importlib.util.spec_from_file_location("export6", REPO / "scripts" / "6_export_trajectronpp_world.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def node_tracks(d, q: int) -> tuple[list[tuple[str, np.ndarray, str]], int]:
    """(node_id, xy (L, 2), class) per agent of canonical sequence q: its observed frames plus its
    future up to the first missing step, so L = 20 + that many steps. Also returns how many
    futures were cut at a hole (a missing step followed by a valid one)."""
    a, b = d["seq_start_end"][q]
    out, holes = [], 0
    for i in range(a, b):
        fm = np.asarray(d["fut_mask"][i], bool)
        n_fut = int(np.argmin(fm)) if not fm.all() else len(fm)        # leading valid steps
        holes += int(fm[n_fut:].any())
        xy = np.concatenate([np.asarray(d["obs_traj"][i]).T, np.asarray(d["pred_traj"][i]).T[:n_fut]]).astype(np.float64)
        out.append((str(d["agent_id"][i]), xy, str(d["agent_type"][i])))
    return out, holes


def build_env(d, seqs, e6, Environment, Scene, Node, derivative_of, pd):
    """A homogeneous Trajectron++ Environment of the given canonical sequences."""
    env = Environment(node_type_list=["PEDESTRIAN"], standardization=e6.STANDARDIZATION_PED_ONLY)
    env.attention_radius = {(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): ATTENTION_RADIUS}
    cols = pd.MultiIndex.from_product([["position", "velocity", "acceleration"], ["x", "y"]])
    scenes, types, holes = [], {}, 0
    for q in seqs:
        t0, name = int(d["t0"][q]), str(d["scene_id"][q])
        scene = Scene(timesteps=t0 + OBS_LEN + PRED_LEN, dt=DT, name=name)
        tracks, h = node_tracks(d, q)
        holes += h
        for node_id, xy, cls in tracks:
            df = pd.DataFrame(e6.pedestrian_frame(xy, DT, derivative_of), columns=cols)
            scene.nodes.append(Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=df, first_timestep=t0))
            types.setdefault(name, {})[node_id] = cls
        scenes.append(scene)
    env.scenes = scenes
    return env, types, {"scenes": len(scenes), "nodes": int(sum(len(s.nodes) for s in scenes)),
                        "futures_cut_at_a_hole": holes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="chaos-gap dataset, e.g. av2_matched_dev")
    ap.add_argument("--idd-test-split", choices=("val", "test"), default="val",
                    help="IDD export pickle copied as test.pkl: 'val' during development, 'test' only "
                         "for the pre-registered runs")
    ap.add_argument("--data", default=None, help="default outputs/chaos_gap/<name>.npz")
    ap.add_argument("--splits-file", default=None, help="default splits/chaos_gap/<name>.json")
    ap.add_argument("--idd-export", default=str(IDD_EXPORT))
    ap.add_argument("--out-dir", default=None, help="default outputs/chaos_gap/tpp_<name>")
    args = ap.parse_args()
    data = Path(args.data or CG / f"{args.name}.npz")
    sp = Path(args.splits_file or REPO / "splits" / "chaos_gap" / f"{args.name}.json")
    idd, out = Path(args.idd_export), Path(args.out_dir or CG / f"tpp_{args.name}")

    sys.path.insert(0, str(TRAJECTRON))
    import dill
    import pandas as pd
    from environment import Environment, Node, Scene, derivative_of

    e6 = export6()
    d = np.load(data, allow_pickle=True)
    mapping = load_splits(sp)
    split_of = [mapping.get(str(s)) for s in d["scene_id"]]
    test_scenes = sorted({str(s) for s, p in zip(d["scene_id"], split_of) if p == "test"})
    src = idd / f"{args.idd_test_split}.pkl"
    with open(src, "rb") as f:
        idd_env = dill.load(f, encoding="latin1")
    if sorted({s.name for s in idd_env.scenes}) != test_scenes:
        raise SystemExit(f"{src} does not hold exactly the test scenes of {sp}")

    out.mkdir(parents=True, exist_ok=True)
    idd_types = json.loads((idd / "node_type_map.json").read_text(encoding="utf-8"))
    type_map, stats = {}, {}
    for split in ("train", "val"):
        seqs = [q for q, p in enumerate(split_of) if p == split]
        env, types, stats[split] = build_env(d, seqs, e6, Environment, Scene, Node, derivative_of, pd)
        with open(out / f"{split}.pkl", "wb") as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
        type_map.update(types)
        print(f"{split}: {stats[split]}")
    shutil.copyfile(src, out / "test.pkl")
    type_map.update({s: idd_types[s] for s in test_scenes})
    stats["test"] = {"scenes": len(idd_env.scenes), "nodes": int(sum(len(s.nodes) for s in idd_env.scenes)),
                     "copied_from": str(src)}
    print(f"test: {stats['test']}")
    shutil.copyfile(idd / "config.json", out / "config.json")
    (out / "node_type_map.json").write_text(json.dumps(type_map, indent=2), encoding="utf-8")
    meta = {
        "dt": DT,
        "splits_file": str(sp),
        "hetero": False,
        "augment_copies_train": 0,
        "attention_radius_m": ATTENTION_RADIUS,
        "node_type_treatment": "all agents as PEDESTRIAN (SingleIntegrator point-mass); "
                               "true classes in node_type_map.json",
        "config_derived_from": str(idd / "config.json"),
        "source_data": str(data),
        "note": ("DEV ONLY: test = IDD-3D val" if args.idd_test_split == "val"
                 else "PRE-REGISTERED: test = the frozen IDD-3D test"),
        "splits": stats,
    }
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
