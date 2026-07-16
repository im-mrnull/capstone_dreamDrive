"""Shared loader for the frozen scene splits.

The split assignment lives in a committed file (splits/scene_splits.json) so
that no script ever re-derives it from an RNG. Seeds affect model init and
shuffling only — never which scenes are train/val/test.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_SPLITS_FILE = Path(__file__).resolve().parent.parent / "splits" / "scene_splits.json"

VALID_SPLITS = ("train", "val", "test")


def load_splits(path: str | Path = DEFAULT_SPLITS_FILE) -> dict[str, str]:
    """Return {scene_id: 'train'|'val'|'test'}."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for split in VALID_SPLITS:
        for sid in payload["splits"][split]:
            if sid in mapping:
                raise ValueError(f"scene {sid} assigned to more than one split")
            mapping[sid] = split
    return mapping


def scenes_for(split: str, path: str | Path = DEFAULT_SPLITS_FILE) -> set[str]:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return set(payload["splits"][split])
