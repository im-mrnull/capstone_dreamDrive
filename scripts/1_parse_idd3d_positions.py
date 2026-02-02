"""Parse IDD-3D per-frame 3D labels into per-object 2D ground-plane positions.

Outputs one CSV per scene with rows:
scene_id,frame_id,obj_id,obj_type,x,y

Notes on coordinates:
IDD-3D label JSONs store object pose under `psr.position` with keys (x, y, z).
In this dataset, (x, y) corresponds to ground-plane coordinates and z is vertical.
This script outputs (x, y) as requested.

Example:
python scripts/parse_idd3d_positions.py --dataset-root idd-3d_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


SCENE_NAME_RE = re.compile(r".*_seq_\d+$")


@dataclass(frozen=True)
class PositionRow:
    scene_id: str
    frame_id: str
    obj_id: str
    obj_type: str
    x: float
    y: float


def iter_scene_dirs(dataset_root: Path, splits: Optional[List[str]] = None) -> Iterator[Tuple[str, Path]]:
    """Yield (scene_id, scene_dir) for each scene under known split dirs."""
    if splits is None:
        # Prefer common split folders if they exist.
        candidate_splits = ["train_val", "train", "val", "test"]
        splits = [s for s in candidate_splits if (dataset_root / s).is_dir()]
        if not splits:
            # Fallback: treat dataset_root as a split-like folder.
            splits = ["."]

    for split in splits:
        split_dir = (dataset_root / split).resolve() if split != "." else dataset_root.resolve()
        if not split_dir.is_dir():
            continue

        for scene_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            label_dir = scene_dir / "label"
            if label_dir.is_dir():
                yield scene_dir.name, scene_dir


def _coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        # avoid scientific formatting surprises
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _loads_json_lenient(text: str, source: str) -> Any:
    """Load JSON but tolerate concatenated extra content.

    Some IDD-3D label files appear to contain valid JSON followed by additional
    bytes (e.g., concatenated JSON). In that case, we parse the first JSON value
    and ignore any trailing non-JSON content.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Common failure: JSON is valid but followed by extra data.
        decoder = json.JSONDecoder()
        try:
            value, end = decoder.raw_decode(text)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON in {source}: {e}") from e

        trailing = text[end:].strip()
        if trailing:
            print(
                f"Warning: trailing extra data after JSON in {source}; ignoring remainder.",
                file=sys.stderr,
            )
        return value


def _read_json_file_lenient(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")
    return _loads_json_lenient(text, str(path))


def parse_label_file(scene_id: str, label_path: Path) -> List[PositionRow]:
    """Parse one per-frame label JSON (list of objects)."""
    frame_id = label_path.stem

    data = _read_json_file_lenient(label_path)

    # IDD-3D per-frame labels are usually a list; handle a couple of common wrappers.
    if isinstance(data, dict):
        for key in ("objects", "labels", "annotations", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(f"Unexpected label JSON structure in {label_path}")

    rows: List[PositionRow] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue

        obj_id = (
            _coerce_str(obj.get("obj_id"))
            or _coerce_str(obj.get("track_id"))
            or _coerce_str(obj.get("id"))
            or _coerce_str(obj.get("instance_id"))
        )
        obj_type = _coerce_str(obj.get("obj_type") or obj.get("category") or obj.get("type") or obj.get("label"))

        psr = obj.get("psr") if isinstance(obj.get("psr"), dict) else None
        pos = psr.get("position") if psr and isinstance(psr.get("position"), dict) else None

        if obj_id is None or obj_type is None or pos is None:
            continue

        x = _coerce_float(pos.get("x"))
        y = _coerce_float(pos.get("y"))

        if x is None or y is None:
            continue

        rows.append(PositionRow(scene_id=scene_id, frame_id=frame_id, obj_id=obj_id, obj_type=obj_type, x=x, y=y))

    return rows


def parse_scene(scene_id: str, scene_dir: Path) -> List[PositionRow]:
    label_dir = scene_dir / "label"
    if not label_dir.is_dir():
        return []

    label_files = sorted(label_dir.glob("*.json"))
    rows: List[PositionRow] = []

    for label_path in label_files:
        # Most frame IDs are 5-digit numbers; keep generic.
        try:
            rows.extend(parse_label_file(scene_id, label_path))
        except Exception as e:
            print(f"Warning: skipping bad label file: {label_path} ({e})", file=sys.stderr)

    return rows


def write_scene_csv(rows: List[PositionRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "frame_id", "obj_id", "obj_type", "x", "y"])
        for r in rows:
            writer.writerow([r.scene_id, r.frame_id, r.obj_id, r.obj_type, f"{r.x:.6f}", f"{r.y:.6f}"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse IDD-3D labels into per-scene position CSVs")
    parser.add_argument("--dataset-root", type=str, default="idd-3d_dataset", help="Path to IDD-3D dataset root")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("outputs") / "parsed_positions"),
        help="Output directory for per-scene CSVs",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=None,
        help="Optional split folder names under dataset root (e.g., train_val train val test)",
    )
    parser.add_argument(
        "--limit-scenes",
        type=int,
        default=None,
        help="Optional limit for quick debugging (process first K scenes)",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.output_dir)

    if not dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    scene_iter = list(iter_scene_dirs(dataset_root, args.splits))
    if args.limit_scenes is not None:
        scene_iter = scene_iter[: args.limit_scenes]

    if not scene_iter:
        raise SystemExit(
            "No scenes found. Expected structure like: idd-3d_dataset/train_val/<scene_id>/label/*.json"
        )

    total_rows = 0
    for idx, (scene_id, scene_dir) in enumerate(scene_iter, start=1):
        rows = parse_scene(scene_id, scene_dir)
        out_path = out_dir / f"{scene_id}.csv"
        write_scene_csv(rows, out_path)

        total_rows += len(rows)
        print(f"[{idx}/{len(scene_iter)}] {scene_id}: {len(rows)} rows -> {out_path}")

    print(f"Done. Wrote {len(scene_iter)} scene CSVs with {total_rows} total rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
