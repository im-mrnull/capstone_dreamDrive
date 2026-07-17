"""Heterogeneous Trajectron++ export: heading math and export invariants."""

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import load_script

exp = load_script("6_export_trajectronpp_world.py")

REPO = Path(__file__).resolve().parent.parent
HETERO_META = REPO / "outputs" / "trajectronpp_world_hetero" / "export_meta.json"
BASE_META = REPO / "outputs" / "trajectronpp_world" / "export_meta.json"


def test_motion_heading_straight_line():
    vx = np.full(10, 3.0)
    vy = np.zeros(10)
    h = exp.motion_heading(vx, vy)
    assert np.allclose(h, 0.0)
    h = exp.motion_heading(vy, vx)  # moving +y
    assert np.allclose(h, np.pi / 2)


def test_motion_heading_holds_through_static_periods():
    # moving east, then stopped, then moving north
    vx = np.array([2.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    vy = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 2.0])
    h = exp.motion_heading(vx, vy)
    assert np.allclose(h[:4], 0.0)          # held while static
    assert np.allclose(h[4:], np.pi / 2)
    assert np.isfinite(h).all()


def test_motion_heading_static_prefix_backfilled():
    vx = np.array([0.0, 0.0, 0.0, 2.0, 2.0])
    vy = np.zeros(5)
    h = exp.motion_heading(vx, vy)
    assert np.allclose(h, 0.0)
    # fully static agent: heading defined (zeros), never NaN
    h0 = exp.motion_heading(np.zeros(5), np.zeros(5))
    assert np.allclose(h0, 0.0)


def test_vehicle_classes_exclude_walkers():
    assert exp.VEHICLE_CLASSES.isdisjoint(
        {"Pedestrian", "Child", "Animal", "BicycleGroup", "Unknown", "Misc", "DontCare"})
    assert {"Car", "Truck", "Bus", "MotorcyleRider", "ScooterRider"} <= exp.VEHICLE_CLASSES


@pytest.mark.skipif(not (HETERO_META.exists() and BASE_META.exists()),
                    reason="hetero export not generated")
def test_hetero_export_same_population_as_baseline():
    het = json.loads(HETERO_META.read_text(encoding="utf-8"))
    base = json.loads(BASE_META.read_text(encoding="utf-8"))
    assert het["hetero"] is True
    assert het["dt"] == base["dt"] == 0.1
    assert het["splits_file"] == base["splits_file"]
    for split in ("train", "val", "test"):
        # identical node population; only the modeled dynamics differ
        assert het["splits"][split]["nodes"] == base["splits"][split]["nodes"]
        by_type = het["splits"][split]["nodes_by_modeled_type"]
        assert by_type["VEHICLE"] > 0 and by_type["PEDESTRIAN"] > 0
    assert het["splits"]["train"]["augment_copies"] == het["augment_copies_train"]
    assert het["splits"]["val"]["augment_copies"] == 0
    assert het["splits"]["test"]["augment_copies"] == 0
