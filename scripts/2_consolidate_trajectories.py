"""Consolidate per-scene position CSVs into per-agent trajectories.

Input: outputs/parsed_positions/*.csv
Columns: scene_id, frame_id, obj_id, obj_type, x, y

Output: one CSV with columns:
scene_id, obj_id, obj_type, trajectory

Where `trajectory` is a JSON list of [frame_id, x, y] entries sorted by frame_id.

Example:
python scripts/consolidate_trajectories.py --input-dir outputs/parsed_positions --output-csv outputs/trajectories/trajectories.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Track:
    obj_type: str
    points: List[Tuple[str, float, float]]  # (frame_id, x, y)


def _parse_frame_key(frame_id: str) -> Tuple[int, str]:
    """Sort key for frame IDs.

    Most IDD-3D frames are zero-padded numeric strings (e.g., '01000').
    We sort numerically when possible, otherwise fall back to lexicographic.
    """
    try:
        return int(frame_id), frame_id
    except ValueError:
        return (2**31 - 1), frame_id


def read_tracks_from_dir(input_dir: Path, limit_files: Optional[int] = None) -> Dict[Tuple[str, str], Track]:
    """Read all per-scene CSVs and return mapping (scene_id, obj_id) -> Track."""
    tracks: Dict[Tuple[str, str], Track] = {}

    csv_files = sorted(input_dir.glob("*.csv"))
    if limit_files is not None:
        csv_files = csv_files[:limit_files]

    if not csv_files:
        raise SystemExit(f"No CSV files found in: {input_dir}")

    for idx, csv_path in enumerate(csv_files, start=1):
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"scene_id", "frame_id", "obj_id", "obj_type", "x", "y"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise SystemExit(f"Unexpected columns in {csv_path}. Found: {reader.fieldnames}")

            row_count = 0
            for row in reader:
                row_count += 1
                scene_id = (row.get("scene_id") or "").strip()
                frame_id = (row.get("frame_id") or "").strip()
                obj_id = (row.get("obj_id") or "").strip()
                obj_type = (row.get("obj_type") or "").strip()

                if not scene_id or not frame_id or not obj_id:
                    continue

                try:
                    x = float(row["x"])
                    y = float(row["y"])
                except (KeyError, ValueError, TypeError):
                    continue

                key = (scene_id, obj_id)
                track = tracks.get(key)
                if track is None:
                    tracks[key] = Track(obj_type=obj_type, points=[(frame_id, x, y)])
                else:
                    # Keep the first non-empty type; warn on conflicts.
                    if (not track.obj_type) and obj_type:
                        track.obj_type = obj_type
                    elif obj_type and track.obj_type and obj_type != track.obj_type:
                        print(
                            f"Warning: obj_type changed for {scene_id}/{obj_id}: '{track.obj_type}' -> '{obj_type}'",
                            file=sys.stderr,
                        )
                    track.points.append((frame_id, x, y))

        print(f"[{idx}/{len(csv_files)}] Read {csv_path.name}: {row_count} rows")

    return tracks


def write_consolidated_csv(tracks: Dict[Tuple[str, str], Track], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "obj_id", "obj_type", "trajectory"])

        # Deterministic order
        for (scene_id, obj_id) in sorted(tracks.keys()):
            track = tracks[(scene_id, obj_id)]
            points_sorted = sorted(track.points, key=lambda p: _parse_frame_key(p[0]))
            trajectory_json = json.dumps([[fid, x, y] for (fid, x, y) in points_sorted], separators=(",", ":"))
            writer.writerow([scene_id, obj_id, track.obj_type, trajectory_json])


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolidate per-scene position CSVs into per-agent trajectories")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(Path("outputs") / "parsed_positions"),
        help="Directory containing per-scene parsed position CSVs",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(Path("outputs") / "trajectories" / "trajectories.csv"),
        help="Path to consolidated trajectories CSV",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Optional limit for quick debugging (process first K CSVs)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    tracks = read_tracks_from_dir(input_dir, limit_files=args.limit_files)
    write_consolidated_csv(tracks, output_csv)

    total_tracks = len(tracks)
    total_points = sum(len(t.points) for t in tracks.values())
    print(f"Done. Wrote {total_tracks} tracks with {total_points} total points -> {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
