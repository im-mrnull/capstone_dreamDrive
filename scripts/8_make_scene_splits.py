"""Generate the frozen train/val/test scene split (run once; commit the JSON).

Assignment is by md5 hash of the scene_id, so it is deterministic, order-free,
and stable if scenes are ever added: hash(scene_id) -> [0,1) bucket,
train < 0.70 <= val < 0.80 <= test. The committed file
splits/scene_splits.json is the single source of truth for every script;
tests/test_splits.py asserts the file matches regeneration.

Usage:
    python scripts/8_make_scene_splits.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TRAIN_UPPER = 0.70
VAL_UPPER = 0.80  # test is the remainder


def bucket(scene_id: str, salt: str) -> float:
    h = hashlib.md5(f"{salt}:{scene_id}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16**12)


def assign(scene_id: str, salt: str) -> str:
    b = bucket(scene_id, salt)
    if b < TRAIN_UPPER:
        return "train"
    if b < VAL_UPPER:
        return "val"
    return "test"


def make_splits(scene_ids: list[str], salt: str) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for sid in sorted(scene_ids):
        splits[assign(sid, salt)].append(sid)
    return splits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-dir", default="outputs/multiagent_world_v1/scenes",
                    help="directory whose *.npz stems define the scene universe")
    ap.add_argument("--out", default="splits/scene_splits.json")
    ap.add_argument("--salt", default="dreamdrive-v2.9",
                    help="hash salt; changing it reshuffles every assignment. "
                         "dreamdrive-v2.9 was chosen (from 40 candidates) as the salt whose "
                         "buckets land closest to the 70/10/20 target on the 105 scenes")
    args = ap.parse_args()

    scene_ids = sorted(p.stem for p in (REPO / args.scenes_dir).glob("*.npz"))
    if not scene_ids:
        raise SystemExit(f"no scenes found in {args.scenes_dir}")

    splits = make_splits(scene_ids, args.salt)
    payload = {
        "method": "md5(salt:scene_id) bucket; train<0.70<=val<0.80<=test",
        "salt": args.salt,
        "num_scenes": len(scene_ids),
        "counts": {k: len(v) for k, v in splits.items()},
        "splits": splits,
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}: {payload['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
