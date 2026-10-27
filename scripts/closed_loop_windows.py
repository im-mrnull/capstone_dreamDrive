"""Dense prediction windows for the closed-loop test (canonical NPZ layout).

The closed-loop agents replay their recorded tracks and do not react to the
ego, so every predictor's output at a planning step depends only on the log and
can be computed once. This writes, for every test scene and every planning
time t in {19, 21, ..., 97} (one replan per 0.2 s), one sequence holding every
agent observed for the full 2 s window [t-19, t], in the canonical layout that
CanonicalSeqDataset and the model drivers read (same conventions as script 5:
zero-filled masked futures, rel = frame differences). Unlike the benchmark
windows, an agent needs no future step here, and the 60-step future may run past
the end of the scene (masked), because a planner must predict every agent it
sees.

Usage:
    python scripts/closed_loop_windows.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from dd_split import load_splits

REPO = Path(__file__).resolve().parent.parent
SCENES = REPO / "outputs" / "multiagent_world_v1" / "scenes"
OUT_DIR = REPO / "outputs" / "closed_loop"
OBS_LEN, PRED_LEN = 20, 60
PLAN_TIMES = range(OBS_LEN - 1, 98, 2)          # 19, 21, ..., 97


def _script5():
    spec = importlib.util.spec_from_file_location("s5", REPO / "scripts" / "5_export_stgat_format.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # its @dataclass looks the module up
    spec.loader.exec_module(mod)
    return mod


def build(split: str, splits_file) -> dict:
    s5 = _script5()
    mapping = load_splits(splits_file)
    cols = {k: [] for k in ("obs", "pred", "obs_rel", "pred_rel", "fm", "aid", "atype")}
    sse, scene_ids, t0s, cursor = [], [], [], 0
    for p in sorted(SCENES.glob("*.npz")):
        if mapping.get(p.stem) != split:
            continue
        d = np.load(p, allow_pickle=True)
        pos, mask = d["pos"].astype(np.float32), d["mask"].astype(bool)
        T, A, _ = pos.shape
        pad = OBS_LEN + PRED_LEN
        pos = np.concatenate([pos, np.zeros((pad, A, 2), np.float32)])
        mask = np.concatenate([mask, np.zeros((pad, A), bool)])
        for t in PLAN_TIMES:
            t0 = t - (OBS_LEN - 1)
            win_mask = mask[t0:t0 + pad]
            idx = np.where(win_mask[:OBS_LEN].all(axis=0))[0]
            if idx.size == 0:
                continue
            traj = np.transpose(pos[t0:t0 + pad, idx], (1, 2, 0))
            obs, pred = traj[:, :, :OBS_LEN], traj[:, :, OBS_LEN:]
            fm = win_mask[OBS_LEN:, idx].T
            pred = np.where(fm[:, None, :], pred, 0.0).astype(np.float32)
            obs_rel = np.zeros_like(obs)
            s5._traj_rel_inplace(obs_rel, obs)
            rel_valid = fm.copy()
            rel_valid[:, 1:] &= fm[:, :-1]
            pred_rel = np.where(rel_valid[:, None, :], s5._pred_rel(pred, obs), 0.0).astype(np.float32)
            for k, v in zip(cols, (obs, pred, obs_rel, pred_rel, fm, d["agent_id"][idx],
                                   np.asarray(d["agent_type"], dtype=object)[idx])):
                cols[k].append(v)
            sse.append((cursor, cursor + idx.size)); cursor += idx.size
            scene_ids.append(p.stem); t0s.append(t0)
    cat = np.concatenate
    return {"obs_traj": cat(cols["obs"]), "pred_traj": cat(cols["pred"]),
            "obs_traj_rel": cat(cols["obs_rel"]), "pred_traj_rel": cat(cols["pred_rel"]),
            "fut_mask": cat(cols["fm"]), "seq_start_end": np.asarray(sse, dtype=np.int64),
            "agent_id": cat(cols["aid"]).astype(np.int32), "agent_type": cat(cols["atype"]).astype(object),
            "scene_id": np.asarray(scene_ids, dtype=object), "t0": np.asarray(t0s, dtype=np.int32)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--splits-file", default=str(REPO / "splits" / "scene_splits.json"))
    args = ap.parse_args()
    data = build(args.split, args.splits_file)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"dense_windows_{args.split}.npz"
    np.savez_compressed(out, **data)
    stats = {"split": args.split, "sequences": int(len(data["t0"])), "agents": int(len(data["agent_id"])),
             "plan_times": [PLAN_TIMES.start, PLAN_TIMES.stop - 1, PLAN_TIMES.step],
             "agents_without_future": int((~data["fut_mask"].any(axis=1)).sum())}
    (OUT_DIR / f"dense_windows_{args.split}_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(stats, "->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
