"""Data sanity checks on the committed world-frame trajectories.

These are the tests that would have caught the ego-frame contamination
automatically: in a world frame, pedestrians walk at ~1.4 m/s and cannot
sustain 5+ m/s medians. They run in CI against the committed CSVs.
"""

import csv
from pathlib import Path

import pytest

from conftest import load_script

ego = load_script("0_compensate_ego_motion.py")

WORLD_DIR = Path(__file__).resolve().parent.parent / "outputs" / "parsed_positions_world"
RAW_DIR = Path(__file__).resolve().parent.parent / "outputs" / "parsed_positions"

needs_world = pytest.mark.skipif(
    not WORLD_DIR.is_dir(), reason="world-frame CSVs not generated (run scripts/0_compensate_ego_motion.py)"
)


def _load_rows(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(directory.glob("*.csv")):
        with p.open("r", newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


@needs_world
def test_pedestrian_speeds_plausible_in_world_frame():
    sp = ego.speed_percentiles(_load_rows(WORLD_DIR), "Pedestrian")
    assert sp is not None, "no pedestrians found"
    assert sp["p50"] < 2.5, f"median pedestrian speed {sp['p50']:.2f} m/s is not walking pace"
    assert sp["p90"] < 4.5, f"p90 pedestrian speed {sp['p90']:.2f} m/s exceeds running pace"


@needs_world
def test_world_frame_actually_differs_from_raw_ego_frame():
    """Guards against the compensation stage silently becoming a no-op."""
    raw = ego.speed_percentiles(_load_rows(RAW_DIR), "Pedestrian")
    world = ego.speed_percentiles(_load_rows(WORLD_DIR), "Pedestrian")
    assert raw is not None and world is not None
    assert world["p50"] < 0.7 * raw["p50"], (
        f"compensation had little effect: raw p50 {raw['p50']:.2f} vs world p50 {world['p50']:.2f}"
    )


@needs_world
def test_car_speeds_within_urban_limits():
    sp = ego.speed_percentiles(_load_rows(WORLD_DIR), "Car")
    assert sp is not None
    assert sp["p90"] < 25.0, f"p90 car speed {sp['p90']:.2f} m/s (=90 km/h) implausible for urban scenes"
