"""Export multi-agent per-scene tensors from parsed IDD-3D positions.

Goal
----
Trajectron++ (and other multi-agent models) need per-scene, per-timestep data with
multiple agents present simultaneously. Our existing `windows_obs20_pred60.npz`
is single-agent windows, which is not enough.

This exporter takes `outputs/parsed_positions/<scene_id>.csv` (produced by
`scripts/parse_idd3d_positions.py`) and writes, for each scene, an NPZ containing:

- pos:      (T, A, 2) float32, NaN where missing
- mask:     (T, A) bool, True where position is present
- vel:      (T, A, 2) float32, NaN where unavailable
- vel_mask: (T, A) bool, True when vel is valid (both t and t-1 present)
- frame_id: (T,) int32 or object (original frame ids, ordered)
- agent_id: (A,) int32 (original obj_id, ordered)
- agent_type: (A,) object (mode/most-common obj_type per agent)

This is a lightweight intermediate format; later we can adapt it to whichever
Trajectron++ codebase/dataloader you choose.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class SceneStats:
    scene_id: str
    frames: int
    agents: int
    points: int


def _try_int(s: str) -> int | None:
    try:
        return int(s)
    except Exception:
        return None


def _read_scene_csv(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    required = {"scene_id", "frame_id", "obj_id", "obj_type", "x", "y"}
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    return rows


def _mode_nonempty(values: Iterable[str]) -> str:
    c = Counter(v for v in values if v is not None and str(v).strip() != "")
    if not c:
        return ""
    return c.most_common(1)[0][0]


def build_scene_tensors(rows: List[Dict[str, str]]) -> Tuple[Dict[str, np.ndarray], SceneStats]:
    if not rows:
        raise ValueError("empty scene")

    scene_id = rows[0]["scene_id"]

    # Collect frames + agents
    frame_vals = [r["frame_id"] for r in rows]
    frame_ints = [_try_int(v) for v in frame_vals]
    all_frames_int = all(v is not None for v in frame_ints)

    if all_frames_int:
        unique_frames = sorted(set(int(v) for v in frame_ints if v is not None))
        frame_id_arr: np.ndarray = np.asarray(unique_frames, dtype=np.int32)
        frame_to_idx = {fid: i for i, fid in enumerate(unique_frames)}
        get_frame_key = lambda r: int(r["frame_id"])  # noqa: E731
    else:
        unique_frames_str = sorted(set(frame_vals))
        frame_id_arr = np.asarray(unique_frames_str, dtype=object)
        frame_to_idx = {fid: i for i, fid in enumerate(unique_frames_str)}
        get_frame_key = lambda r: r["frame_id"]  # noqa: E731

    obj_ids = sorted(set(int(r["obj_id"]) for r in rows))
    agent_id_arr = np.asarray(obj_ids, dtype=np.int32)
    agent_to_idx = {oid: i for i, oid in enumerate(obj_ids)}

    # Determine a stable type per agent (mode of obj_type)
    types_by_agent: Dict[int, List[str]] = defaultdict(list)
    for r in rows:
        types_by_agent[int(r["obj_id"])].append(r.get("obj_type", "") or "")
    agent_type_arr = np.asarray([_mode_nonempty(types_by_agent[oid]) for oid in obj_ids], dtype=object)

    T = len(frame_id_arr)
    A = len(agent_id_arr)

    pos = np.full((T, A, 2), np.nan, dtype=np.float32)
    mask = np.zeros((T, A), dtype=bool)

    points = 0
    for r in rows:
        fid = get_frame_key(r)
        oid = int(r["obj_id"])
        t = frame_to_idx[fid]
        a = agent_to_idx[oid]

        # Last-write-wins if duplicates exist (rare)
        pos[t, a, 0] = np.float32(r["x"])
        pos[t, a, 1] = np.float32(r["y"])
        mask[t, a] = True
        points += 1

    # Velocity: finite difference in frame index space (dt=1 index step)
    vel = np.full((T, A, 2), np.nan, dtype=np.float32)
    vel_mask = np.zeros((T, A), dtype=bool)
    if T >= 2:
        valid_prev = mask[:-1]
        valid_curr = mask[1:]
        vmask = valid_prev & valid_curr
        dv = pos[1:] - pos[:-1]
        vel[1:][vmask] = dv[vmask]
        vel_mask[1:] = vmask

    tensors = {
        "pos": pos,
        "mask": mask,
        "vel": vel,
        "vel_mask": vel_mask,
        "frame_id": frame_id_arr,
        "agent_id": agent_id_arr,
        "agent_type": agent_type_arr,
    }

    stats = SceneStats(scene_id=scene_id, frames=T, agents=A, points=points)
    return tensors, stats


def save_scene_npz(out_path: Path, tensors: Dict[str, np.ndarray]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **tensors)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parsed-positions-dir",
        default="outputs/parsed_positions",
        help="Directory containing per-scene CSVs from parse_idd3d_positions.py",
    )
    ap.add_argument(
        "--out-dir",
        default="outputs/multiagent_trajpp_v1",
        help="Output directory for per-scene NPZs + index.json",
    )
    ap.add_argument("--limit", type=int, default=0, help="If >0, only export first N scenes")

    args = ap.parse_args()

    in_dir = Path(args.parsed_positions_dir)
    out_dir = Path(args.out_dir)
    scenes_dir = out_dir / "scenes"

    csv_paths = sorted(in_dir.glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"No CSVs found in: {in_dir}")

    if args.limit and args.limit > 0:
        csv_paths = csv_paths[: int(args.limit)]

    all_stats: List[Dict[str, Any]] = []

    for i, csv_path in enumerate(csv_paths, start=1):
        rows = _read_scene_csv(csv_path)
        tensors, stats = build_scene_tensors(rows)

        out_path = scenes_dir / f"{stats.scene_id}.npz"
        save_scene_npz(out_path, tensors)

        all_stats.append(
            {
                "scene_id": stats.scene_id,
                "frames": stats.frames,
                "agents": stats.agents,
                "points": stats.points,
                "file": str(out_path.relative_to(out_dir)).replace("\\", "/"),
            }
        )

        if i % 10 == 0 or i == len(csv_paths):
            print(f"exported {i}/{len(csv_paths)} scenes")

    index = {
        "format": "multiagent_trajpp_v1",
        "source": str(in_dir).replace("\\", "/"),
        "scenes_dir": "scenes",
        "num_scenes": len(all_stats),
        "total_points": int(sum(s["points"] for s in all_stats)),
        "generated_files": all_stats,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    schema = {
        "pos": {"shape": ["T", "A", 2], "dtype": "float32", "missing": "NaN"},
        "mask": {"shape": ["T", "A"], "dtype": "bool"},
        "vel": {"shape": ["T", "A", 2], "dtype": "float32", "missing": "NaN"},
        "vel_mask": {"shape": ["T", "A"], "dtype": "bool"},
        "frame_id": {"shape": ["T"], "dtype": "int32|object", "notes": "ordered unique frames"},
        "agent_id": {"shape": ["A"], "dtype": "int32", "notes": "ordered unique obj_id"},
        "agent_type": {"shape": ["A"], "dtype": "object", "notes": "mode of obj_type per agent"},
    }
    with open(out_dir / "schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"wrote: {out_dir / 'index.json'}")
    print(f"wrote: {out_dir / 'schema.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
