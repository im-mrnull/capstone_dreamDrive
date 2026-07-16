"""Integrity of the frozen scene split (splits/scene_splits.json)."""

import json
from pathlib import Path

import pytest

from conftest import load_script

REPO = Path(__file__).resolve().parent.parent
SPLITS_FILE = REPO / "splits" / "scene_splits.json"

mk = load_script("8_make_scene_splits.py")

import dd_split  # noqa: E402  (scripts dir on sys.path via conftest)


def _payload():
    return json.loads(SPLITS_FILE.read_text(encoding="utf-8"))


def test_every_scene_in_exactly_one_split():
    p = _payload()
    all_ids = [sid for split in ("train", "val", "test") for sid in p["splits"][split]]
    assert len(all_ids) == len(set(all_ids)), "a scene appears in more than one split"
    assert len(all_ids) == p["num_scenes"]


def test_ratios_within_tolerance():
    p = _payload()
    n = p["num_scenes"]
    assert p["counts"]["train"] / n == pytest.approx(0.70, abs=0.06)
    assert p["counts"]["val"] / n == pytest.approx(0.10, abs=0.05)
    assert p["counts"]["test"] / n == pytest.approx(0.20, abs=0.06)
    assert p["counts"]["val"] >= 5, "val split too small for model selection"


def test_file_matches_regeneration():
    """Guards against hand-edits: the committed assignment must equal what the
    generator produces for the recorded salt."""
    p = _payload()
    scene_ids = [sid for split in ("train", "val", "test") for sid in p["splits"][split]]
    regenerated = mk.make_splits(scene_ids, p["salt"])
    assert regenerated == p["splits"]


def test_loader_roundtrip():
    mapping = dd_split.load_splits(SPLITS_FILE)
    p = _payload()
    assert len(mapping) == p["num_scenes"]
    for split in ("train", "val", "test"):
        assert dd_split.scenes_for(split, SPLITS_FILE) == set(p["splits"][split])


def test_splits_cover_canonical_dataset_scenes():
    import numpy as np
    npz = REPO / "outputs" / "canonical_world_obs20_pred60.npz"
    if not npz.exists():
        pytest.skip("canonical dataset not generated")
    d = np.load(npz, allow_pickle=True)
    mapping = dd_split.load_splits(SPLITS_FILE)
    scene_ids = set(str(s) for s in d["scene_id"])
    missing = [s for s in scene_ids if s not in mapping]
    assert not missing, f"scenes without a split assignment: {missing[:5]}"
