"""Build a chaos-gap dataset: Argoverse 2 sequences chosen by scenario hash, merged with the
IDD-3D sequences of one split, plus its split file (the dd_split format).

Every benchmark script then runs on it unchanged, through --data / --input / --input-npz and
--splits-file. Membership depends only on each scenario id (sha256 of SALT:scene_id, ascending),
never on an RNG, so a subset is reproducible from the ids alone.

    train  AV2 train scenarios in hash order, filled to exactly --train-agents agents: a
           scenario that would overshoot is skipped, so the count is exact and the order fixed.
    val    the first --val-scenarios AV2 val scenarios in hash order (model selection; it also
           serves as the in-domain sanity check).
    test   the IDD-3D scenes of --idd-test-split. During development this is 'val' (10 scenes,
           993 agents). The frozen IDD test is excluded from the file unless the split is given
           explicitly as 'test', which is for the pre-registered runs only.

Writes <out>.npz (the canonical keys), <out>.stats.json and the split file.

Usage:
    python scripts/cg_build_splits.py --name av2_matched --train-agents 7406
    python scripts/cg_build_splits.py --name av2_large --train-agents 200000
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dd_split import load_splits

REPO = Path(__file__).resolve().parent.parent
SALT = "dreamdrive-chaosgap-v1"
KEYS = ("obs_traj", "pred_traj", "obs_traj_rel", "pred_traj_rel", "fut_mask", "agent_id", "agent_type")
AV2_TRAIN = REPO / "outputs" / "chaos_gap" / "av2_train.npz"
AV2_VAL = REPO / "outputs" / "chaos_gap" / "av2_val.npz"
IDD = REPO / "outputs" / "canonical_world_obs20_pred60.npz"
IDD_SPLITS = REPO / "splits" / "scene_splits.json"


def hash_order(scene_ids) -> np.ndarray:
    """Indices of scene_ids in ascending sha256(SALT:scene_id) order."""
    keys = [hashlib.sha256(f"{SALT}:{s}".encode()).hexdigest() for s in scene_ids]
    return np.argsort(np.asarray(keys), kind="stable")


def fill_exact(sizes: np.ndarray, order: np.ndarray, target: int) -> list[int]:
    """Sequences in `order` whose agent counts sum to exactly `target`; one that would
    overshoot is skipped."""
    out, total = [], 0
    for i in order:
        n = int(sizes[i])
        if total + n <= target:
            out.append(int(i))
            total += n
            if total == target:
                return out
    raise SystemExit(f"cannot fill exactly {target} agents (reached {total})")


def take(d, seqs) -> dict:
    """The arrays of the given sequences of a canonical file, re-indexed."""
    se = d["seq_start_end"]
    rows = np.concatenate([np.arange(se[i, 0], se[i, 1]) for i in seqs]) if len(seqs) else np.zeros(0, int)
    n = np.cumsum([0] + [int(se[i, 1] - se[i, 0]) for i in seqs])
    out = {k: d[k][rows] for k in KEYS}
    out["seq_start_end"] = np.stack([n[:-1], n[1:]], axis=1).astype(np.int64)
    out["scene_id"] = np.asarray(d["scene_id"][list(seqs)], dtype=object)
    out["t0"] = np.asarray(d["t0"][list(seqs)], dtype=np.int32)
    return out


def merge(parts: list[dict]) -> dict:
    out = {k: np.concatenate([p[k] for p in parts], axis=0) for k in KEYS + ("scene_id", "t0")}
    offset, se = 0, []
    for p in parts:
        se.append(p["seq_start_end"] + offset)
        offset += p["obs_traj"].shape[0]
    out["seq_start_end"] = np.concatenate(se, axis=0).astype(np.int64)
    out["scene_id"] = out["scene_id"].astype(object)
    out["agent_type"] = out["agent_type"].astype(object)
    return out


def build(av2_train, av2_val, idd, idd_splits: dict, train_agents: int, val_scenarios: int,
          idd_test_split: str) -> tuple[dict, dict, dict]:
    tr_sizes = av2_train["seq_start_end"][:, 1] - av2_train["seq_start_end"][:, 0]
    tr = fill_exact(tr_sizes, hash_order(av2_train["scene_id"]), train_agents)
    va = hash_order(av2_val["scene_id"])[:val_scenarios].tolist()
    te = [i for i, s in enumerate(idd["scene_id"]) if idd_splits.get(str(s)) == idd_test_split]
    parts = {"train": take(av2_train, tr), "val": take(av2_val, va), "test": take(idd, te)}
    data = merge([parts["train"], parts["val"], parts["test"]])
    splits = {k: sorted({str(s) for s in parts[k]["scene_id"]}) for k in parts}
    assert not (set(splits["train"]) & set(splits["val"])) and not (set(splits["train"]) & set(splits["test"]))
    stats = {}
    for k, p in parts.items():
        types, counts = np.unique(p["agent_type"], return_counts=True)
        stats[k] = {"sequences": len(p["scene_id"]), "scenes": len(splits[k]), "agents": int(p["obs_traj"].shape[0]),
                    "agents_by_type": {str(t): int(c) for t, c in zip(types, counts)}}
    return data, splits, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="dataset name, e.g. av2_matched")
    ap.add_argument("--train-agents", type=int, required=True)
    ap.add_argument("--val-scenarios", type=int, default=250,
                    help="AV2 val scenarios for selection (~18 agents each); every model validates each epoch")
    ap.add_argument("--idd-test-split", choices=("val", "test"), default="val",
                    help="'val' during development; 'test' only for the pre-registered runs")
    ap.add_argument("--av2-train", default=str(AV2_TRAIN))
    ap.add_argument("--av2-val", default=str(AV2_VAL))
    ap.add_argument("--idd", default=str(IDD))
    ap.add_argument("--idd-splits", default=str(IDD_SPLITS))
    ap.add_argument("--out-dir", default=str(REPO / "outputs" / "chaos_gap"))
    ap.add_argument("--splits-dir", default=str(REPO / "splits" / "chaos_gap"))
    args = ap.parse_args()
    tag = f"{args.name}_{'dev' if args.idd_test_split == 'val' else 'final'}"
    data, splits, stats = build(np.load(args.av2_train, allow_pickle=True), np.load(args.av2_val, allow_pickle=True),
                                np.load(args.idd, allow_pickle=True), load_splits(args.idd_splits), args.train_agents,
                                args.val_scenarios, args.idd_test_split)
    out, sp = Path(args.out_dir) / f"{tag}.npz", Path(args.splits_dir) / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    sp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    note = ("DEV ONLY: test = IDD-3D val; the frozen IDD-3D test is not in this file" if args.idd_test_split == "val"
            else "PRE-REGISTERED: test = the frozen IDD-3D test")
    sp.write_text(json.dumps({"method": f"sha256({SALT}:scene_id) ascending; train filled to exactly "
                                        f"{args.train_agents} agents; val = first {args.val_scenarios} AV2 val scenarios; "
                                        f"test = IDD-3D {args.idd_test_split}",
                              "salt": SALT, "note": note, "counts": {k: len(v) for k, v in splits.items()},
                              "splits": splits}, indent=1), encoding="utf-8")
    out.with_suffix(".stats.json").write_text(json.dumps({"name": tag, "note": note, **stats}, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=1))
    print("saved:", out, sp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
