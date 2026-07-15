"""Unit tests for scripts/0_compensate_ego_motion.py on synthetic scenes with
known ego motion: the recovered world-frame positions must match ground truth.
"""

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from conftest import load_script

ego = load_script("0_compensate_ego_motion.py")


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def make_synthetic_scene(tmp_path: Path, n_frames: int = 30) -> tuple[Path, dict]:
    """Ego drives forward while turning; world contains static objects and one
    moving pedestrian. Positions are written in the per-frame ego frame,
    exactly like the raw IDD-3D parse."""
    rng = np.random.default_rng(7)
    static_world = rng.uniform(-30, 30, size=(8, 2))  # 8 parked objects
    ped_start = np.array([5.0, 5.0])
    # 2.2 m/s: clearly outside the 0.12 m/frame RANSAC inlier threshold, so the
    # moving agent is a proper outlier. (Agents moving slower than the threshold
    # get absorbed into the "static" fit and cause small ego-pose drift — a
    # documented limitation of annotation-based compensation.)
    ped_vel = np.array([0.18, 0.12])

    rows = []
    truth: dict[tuple[str, str], np.ndarray] = {}
    for t in range(n_frames):
        # Ego pose in world: gentle arc, 8 m/s forward
        yaw = 0.02 * t
        pos = np.array([0.8 * t, 0.5 * math.sin(0.05 * t)])
        R, o = _rot(yaw), pos
        frame_id = f"{1000 + t:05d}"
        world_pts = {f"{i}": static_world[i] for i in range(len(static_world))}
        world_pts["ped"] = ped_start + t * ped_vel
        for oid, w in world_pts.items():
            e = R.T @ (w - o)  # observe in ego frame
            otype = "Pedestrian" if oid == "ped" else "Car"
            rows.append({"scene_id": "synth", "frame_id": frame_id, "obj_id": oid,
                         "obj_type": otype, "x": f"{e[0]:.6f}", "y": f"{e[1]:.6f}"})
            truth[(frame_id, oid)] = w

    csv_path = tmp_path / "synth.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scene_id", "frame_id", "obj_id", "obj_type", "x", "y"])
        w.writeheader()
        w.writerows(rows)
    return csv_path, truth


def test_recovers_world_positions(tmp_path):
    csv_path, truth = make_synthetic_scene(tmp_path)
    world_rows, ego_rows, stats = ego.compensate_scene(
        csv_path, inlier_thresh=0.12, ransac_iters=200, min_inliers=3, seed=0
    )
    assert stats["fallback_frames"] == 0

    # World frame is anchored at the ego pose of frame 0; transform ground
    # truth into that same frame before comparing.
    first_frame = min(truth, key=lambda k: k[0])[0]
    # Ego pose at t=0 in the generator: yaw=0, pos=(0, 0) -> frames coincide.
    err = []
    for r in world_rows:
        w_true = truth[(r["frame_id"], r["obj_id"])]
        w_est = np.array([float(r["x"]), float(r["y"])])
        err.append(np.linalg.norm(w_true - w_est))
    assert max(err) < 0.10, f"max world-position error {max(err):.3f} m"


def test_pedestrian_speed_is_plausible_after_compensation(tmp_path):
    csv_path, _ = make_synthetic_scene(tmp_path)
    world_rows, _, _ = ego.compensate_scene(
        csv_path, inlier_thresh=0.12, ransac_iters=200, min_inliers=3, seed=0
    )
    sp = ego.speed_percentiles(world_rows, "Pedestrian")
    assert sp is not None
    assert 1.9 < sp["p50"] < 2.5  # generator moves at ~2.16 m/s


def test_rigid_from_pairs_exact():
    rng = np.random.default_rng(0)
    src = rng.uniform(-10, 10, size=(6, 2))
    A_true, b_true = _rot(0.3), np.array([1.5, -2.0])
    dst = src @ A_true.T + b_true
    A, b = ego.rigid_from_pairs(src, dst)
    assert np.allclose(A, A_true, atol=1e-9)
    assert np.allclose(b, b_true, atol=1e-9)
