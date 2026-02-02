"""Create trajectory windows (obs/fut) from consolidated IDD-3D tracks.

Input CSV:
outputs/trajectories/trajectories.csv
Columns: scene_id, obj_id, obj_type, trajectory
Where `trajectory` is a JSON list of [frame_id, x, y]

Output NPZ:
outputs/windows_obs20_pred60.npz
Contains arrays:
- obs: (N, obs_len, 2)
- fut: (N, pred_len, 2)
- scene_id: (N,)

Windowing:
- sort each trajectory by frame_id
- sliding windows with stride
- discard if shorter than obs_len + pred_len
- center by subtracting last observed point (obs[-1]) from both obs and fut

Example:
python scripts/make_windows_obs20_pred60.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple


def _frame_sort_key(frame_id: Any) -> Tuple[int, str]:
    """Sort key for frame IDs: numeric when possible, else lexicographic."""
    s = str(frame_id)
    try:
        return int(s), s
    except ValueError:
        return (2**31 - 1), s


def _parse_trajectory_json(text: str, source: str) -> List[Tuple[str, float, float]]:
    """Parse trajectory JSON list of [frame_id, x, y]."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Bad trajectory JSON in {source}: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"Trajectory JSON is not a list in {source}")

    points: List[Tuple[str, float, float]] = []
    for item in data:
        if not (isinstance(item, (list, tuple)) and len(item) == 3):
            continue
        frame_id, x, y = item
        try:
            points.append((str(frame_id), float(x), float(y)))
        except (TypeError, ValueError):
            continue

    return points


def main() -> int:
    parser = argparse.ArgumentParser(description="Create obs/fut windows from consolidated trajectories")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(Path("outputs") / "trajectories" / "trajectories.csv"),
        help="Path to consolidated trajectories CSV",
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default=str(Path("outputs") / "windows_obs20_pred60.npz"),
        help="Output .npz path",
    )
    parser.add_argument("--obs-len", type=int, default=20)
    parser.add_argument("--pred-len", type=int, default=60)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument(
        "--limit-tracks",
        type=int,
        default=None,
        help="Optional limit for debugging (process first K tracks)",
    )
    args = parser.parse_args()

    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "numpy is required for this script. Install with: pip install numpy\n"
            f"Import error: {e}"
        )

    input_csv = Path(args.input_csv)
    output_npz = Path(args.output_npz)

    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    obs_len = int(args.obs_len)
    pred_len = int(args.pred_len)
    stride = int(args.stride)

    if obs_len <= 0 or pred_len <= 0 or stride <= 0:
        raise SystemExit("obs_len, pred_len, stride must be positive")

    window_len = obs_len + pred_len

    obs_list: List[Any] = []
    fut_list: List[Any] = []
    scene_list: List[str] = []

    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"scene_id", "obj_id", "obj_type", "trajectory"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(f"Unexpected columns in {input_csv}. Found: {reader.fieldnames}")

        for idx, row in enumerate(reader, start=1):
            if args.limit_tracks is not None and idx > args.limit_tracks:
                break

            scene_id = (row.get("scene_id") or "").strip()
            trajectory_text = row.get("trajectory")
            if not scene_id or not trajectory_text:
                continue

            points = _parse_trajectory_json(trajectory_text, source=f"{input_csv}:{idx}")
            if len(points) < window_len:
                continue

            points_sorted = sorted(points, key=lambda p: _frame_sort_key(p[0]))
            xy = np.asarray([[p[1], p[2]] for p in points_sorted], dtype=np.float32)

            # Sliding windows
            for start in range(0, len(xy) - window_len + 1, stride):
                window_xy = xy[start : start + window_len]
                obs = window_xy[:obs_len]
                fut = window_xy[obs_len:]

                if obs.shape[0] != obs_len or fut.shape[0] != pred_len:
                    continue

                center = obs[-1].copy()
                obs_centered = obs - center
                fut_centered = fut - center

                obs_list.append(obs_centered)
                fut_list.append(fut_centered)
                scene_list.append(scene_id)

            if idx % 500 == 0:
                print(f"Processed {idx} tracks... windows so far: {len(obs_list)}")

    if not obs_list:
        raise SystemExit("No windows were generated. Check input CSV and lengths.")

    obs_arr = np.stack(obs_list, axis=0).astype(np.float32, copy=False)
    fut_arr = np.stack(fut_list, axis=0).astype(np.float32, copy=False)
    scene_arr = np.asarray(scene_list, dtype=str)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, obs=obs_arr, fut=fut_arr, scene_id=scene_arr)

    print(f"Saved: {output_npz}")
    print(f"obs shape: {obs_arr.shape} | fut shape: {fut_arr.shape} | scene_id shape: {scene_arr.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
