"""Ego-motion compensation for IDD-3D parsed positions (annotation-based).

Problem
-------
IDD-3D label JSONs store `psr.position` in the per-frame LiDAR/ego frame.
The ego vehicle moves, so raw (x, y) tracks mix agent motion with ego motion
(e.g. pedestrians appear to travel ~7 m/s). Every downstream artifact built
from `outputs/parsed_positions/` inherits this contamination.

Method
------
The raw dataset (LiDAR .pcd + official ego poses) is not part of this repo,
so ego motion is estimated from the annotations themselves:

For each consecutive frame pair (t, t+1), objects observed in both frames
give point correspondences e_t -> e_{t+1} in the two ego frames. For a
STATIC object these are related by one rigid SE(2) transform
    e_{t+1} = A e_t + b,   A = R_{t+1}^T R_t,   b = R_{t+1}^T (o_t - o_{t+1}),
where (R_t, o_t) is the ego pose in the world frame. Moving agents violate
this transform. RANSAC over correspondence pairs therefore finds the static
cluster and its transform, from which the relative ego pose follows. Poses
are accumulated per scene, with the scene's first ego frame as world origin.

Failure modes handled:
- < 2 shared objects, or too few inliers: reuse the previous relative
  transform (constant-velocity ego assumption) and record the fallback.
- A rigid platoon of vehicles can outvote the static world in dense traffic.
  Candidate transforms are scored by inlier count with a smoothness
  tie-break (agreement with the previous relative transform).

This is an APPROXIMATION pending official ego poses; per-scene diagnostics
(inlier ratios, fallback counts, pedestrian speed percentiles before/after)
are written to a stats JSON so its quality is auditable.

Usage:
python scripts/0_compensate_ego_motion.py \
    --input-dir outputs/parsed_positions \
    --output-dir outputs/parsed_positions_world \
    --ego-csv outputs/ego_trajectories.csv \
    --stats-json outputs/ego_compensation_stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

FPS = 10.0  # IDD-3D capture rate; dt = 0.1 s


def _frame_key(frame_id: str) -> Tuple[int, str]:
    try:
        return int(frame_id), frame_id
    except ValueError:
        return (2**31 - 1), frame_id


def read_scene(csv_path: Path) -> Tuple[List[str], Dict[str, Dict[str, Tuple[float, float, str]]]]:
    """Return (ordered frame_ids, frames[frame_id][obj_id] = (x, y, obj_type))."""
    frames: Dict[str, Dict[str, Tuple[float, float, str]]] = defaultdict(dict)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                x, y = float(row["x"]), float(row["y"])
            except (KeyError, ValueError, TypeError):
                continue
            frames[row["frame_id"]][row["obj_id"]] = (x, y, row.get("obj_type", ""))
    ordered = sorted(frames.keys(), key=_frame_key)
    return ordered, frames


def rigid_from_pairs(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares SE(2) (Umeyama, no scaling): dst ~= A @ src + b."""
    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    H = src_c.T @ dst_c
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    A = Vt.T @ D @ U.T
    b = dst.mean(axis=0) - A @ src.mean(axis=0)
    return A, b


def ransac_rigid(
    src: np.ndarray,
    dst: np.ndarray,
    inlier_thresh: float,
    iters: int,
    prev_A: Optional[np.ndarray],
    prev_b: Optional[np.ndarray],
    rng: np.random.Generator,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """RANSAC SE(2) fit; returns (A, b, n_inliers) or (None, None, 0)."""
    n = src.shape[0]
    if n < 2:
        return None, None, 0

    best: Tuple[float, Optional[np.ndarray], Optional[np.ndarray], int] = (-1.0, None, None, 0)
    candidates = []
    for _ in range(iters):
        i, j = rng.choice(n, size=2, replace=False)
        if np.linalg.norm(src[i] - src[j]) < 1.0:
            continue  # nearly coincident points give unstable rotation
        candidates.append((src[[i, j]], dst[[i, j]]))
    # Always also try the all-points fit (helps tiny-n frames).
    candidates.append((src, dst))

    for s, d in candidates:
        A, b = rigid_from_pairs(s, d)
        resid = np.linalg.norm((src @ A.T + b) - dst, axis=1)
        inliers = resid < inlier_thresh
        n_in = int(inliers.sum())
        if n_in < 2:
            continue
        # Refit on the inlier set.
        A, b = rigid_from_pairs(src[inliers], dst[inliers])
        resid = np.linalg.norm((src @ A.T + b) - dst, axis=1)
        inliers = resid < inlier_thresh
        n_in = int(inliers.sum())
        # Smoothness tie-break: prefer transforms close to the previous one.
        smooth = 0.0
        if prev_A is not None and prev_b is not None:
            smooth = -0.5 * (np.linalg.norm(A - prev_A) + np.linalg.norm(b - prev_b))
        score = n_in + smooth
        if score > best[0]:
            best = (score, A, b, n_in)

    return best[1], best[2], best[3]


def compensate_scene(
    csv_path: Path,
    inlier_thresh: float,
    ransac_iters: int,
    min_inliers: int,
    seed: int,
) -> Tuple[List[dict], List[dict], dict]:
    """Return (world rows, ego pose rows, per-scene stats)."""
    scene_id = csv_path.stem
    ordered_frames, frames = read_scene(csv_path)
    rng = np.random.default_rng(seed)

    # Ego pose accumulation: world = R @ ego + o ; frame 0 is the world origin.
    R = np.eye(2)
    o = np.zeros(2)
    poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if ordered_frames:
        poses[ordered_frames[0]] = (R.copy(), o.copy())

    prev_A: Optional[np.ndarray] = None
    prev_b: Optional[np.ndarray] = None
    fallbacks = 0
    inlier_ratios: List[float] = []

    for f_prev, f_curr in zip(ordered_frames[:-1], ordered_frames[1:]):
        shared = sorted(set(frames[f_prev]) & set(frames[f_curr]))
        src = np.array([frames[f_prev][oid][:2] for oid in shared], dtype=np.float64)
        dst = np.array([frames[f_curr][oid][:2] for oid in shared], dtype=np.float64)

        A, b, n_in = (None, None, 0)
        if len(shared) >= 2:
            A, b, n_in = ransac_rigid(src, dst, inlier_thresh, ransac_iters, prev_A, prev_b, rng)

        if A is None or n_in < min_inliers:
            fallbacks += 1
            if prev_A is not None:
                A, b = prev_A, prev_b  # constant-velocity ego assumption
            else:
                A, b = np.eye(2), np.zeros(2)  # no information yet: assume static ego
        else:
            inlier_ratios.append(n_in / max(1, len(shared)))

        prev_A, prev_b = A, b

        # A = R_{t+1}^T R_t  =>  R_{t+1} = R_t A^T
        # b = R_{t+1}^T (o_t - o_{t+1})  =>  o_{t+1} = o_t - R_{t+1} b
        R = R @ A.T
        o = o - R @ b
        poses[f_curr] = (R.copy(), o.copy())

    world_rows: List[dict] = []
    ego_rows: List[dict] = []
    for fid in ordered_frames:
        R_f, o_f = poses[fid]
        yaw = math.atan2(R_f[1, 0], R_f[0, 0])
        ego_rows.append(
            {"scene_id": scene_id, "frame_id": fid,
             "ego_x": f"{o_f[0]:.6f}", "ego_y": f"{o_f[1]:.6f}", "ego_yaw": f"{yaw:.6f}"}
        )
        for oid, (x, y, otype) in frames[fid].items():
            w = R_f @ np.array([x, y]) + o_f
            world_rows.append(
                {"scene_id": scene_id, "frame_id": fid, "obj_id": oid, "obj_type": otype,
                 "x": f"{w[0]:.6f}", "y": f"{w[1]:.6f}"}
            )

    stats = {
        "scene_id": scene_id,
        "frames": len(ordered_frames),
        "fallback_frames": fallbacks,
        "mean_inlier_ratio": float(np.mean(inlier_ratios)) if inlier_ratios else None,
    }
    return world_rows, ego_rows, stats


def speed_percentiles(rows: List[dict], obj_type: str) -> Optional[Dict[str, float]]:
    """Per-track frame-to-frame speeds (m/s) for one class, pooled percentiles."""
    tracks: Dict[Tuple[str, str], List[Tuple[Tuple[int, str], float, float]]] = defaultdict(list)
    for r in rows:
        if r["obj_type"] != obj_type:
            continue
        tracks[(r["scene_id"], r["obj_id"])].append((_frame_key(r["frame_id"]), float(r["x"]), float(r["y"])))
    speeds: List[float] = []
    for pts in tracks.values():
        pts.sort()
        for (k0, x0, y0), (k1, x1, y1) in zip(pts[:-1], pts[1:]):
            gap = k1[0] - k0[0]
            if gap <= 0:
                continue
            speeds.append(math.hypot(x1 - x0, y1 - y0) * FPS / gap)
    if not speeds:
        return None
    arr = np.asarray(speeds)
    return {p: float(np.percentile(arr, int(p[1:]))) for p in ("p50", "p90", "p99")}


def main() -> int:
    ap = argparse.ArgumentParser(description="Ego-motion compensation from annotations (RANSAC SE(2))")
    ap.add_argument("--input-dir", default="outputs/parsed_positions")
    ap.add_argument("--output-dir", default="outputs/parsed_positions_world")
    ap.add_argument("--ego-csv", default="outputs/ego_trajectories.csv")
    ap.add_argument("--stats-json", default="outputs/ego_compensation_stats.json")
    ap.add_argument("--inlier-thresh", type=float, default=0.12,
                    help="RANSAC inlier residual threshold in meters (annotation jitter scale)")
    ap.add_argument("--ransac-iters", type=int, default=200)
    ap.add_argument("--min-inliers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-scenes", type=int, default=None)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    csv_paths = sorted(in_dir.glob("*.csv"))
    if args.limit_scenes:
        csv_paths = csv_paths[: args.limit_scenes]
    if not csv_paths:
        raise SystemExit(f"No CSVs found in {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    all_scene_stats: List[dict] = []
    all_ego_rows: List[dict] = []
    all_world_rows: List[dict] = []
    all_raw_rows: List[dict] = []

    for i, p in enumerate(csv_paths, start=1):
        world_rows, ego_rows, stats = compensate_scene(
            p, args.inlier_thresh, args.ransac_iters, args.min_inliers, args.seed
        )
        out_path = out_dir / p.name
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["scene_id", "frame_id", "obj_id", "obj_type", "x", "y"])
            w.writeheader()
            w.writerows(world_rows)

        with p.open("r", newline="", encoding="utf-8") as f:
            all_raw_rows.extend(csv.DictReader(f))
        all_world_rows.extend(world_rows)
        all_ego_rows.extend(ego_rows)
        all_scene_stats.append(stats)
        if i % 20 == 0 or i == len(csv_paths):
            print(f"[{i}/{len(csv_paths)}] compensated {p.stem}")

    ego_path = Path(args.ego_csv)
    ego_path.parent.mkdir(parents=True, exist_ok=True)
    with ego_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scene_id", "frame_id", "ego_x", "ego_y", "ego_yaw"])
        w.writeheader()
        w.writerows(all_ego_rows)

    # Validation: pedestrian speeds should collapse to walking range after compensation.
    validation = {}
    for cls in ("Pedestrian", "Car"):
        validation[cls] = {
            "raw_ego_frame": speed_percentiles(all_raw_rows, cls),
            "world_frame": speed_percentiles(all_world_rows, cls),
        }

    stats_payload = {
        "method": "annotation-based RANSAC SE(2) ego-motion estimation (approximation; "
                  "prefer official ego poses when available)",
        "params": {
            "inlier_thresh_m": args.inlier_thresh,
            "ransac_iters": args.ransac_iters,
            "min_inliers": args.min_inliers,
            "seed": args.seed,
            "fps": FPS,
        },
        "num_scenes": len(all_scene_stats),
        "total_fallback_frames": int(sum(s["fallback_frames"] for s in all_scene_stats)),
        "speed_validation_mps": validation,
        "per_scene": all_scene_stats,
    }
    stats_path = Path(args.stats_json)
    stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

    ped = validation.get("Pedestrian", {})
    print(f"wrote {len(csv_paths)} world-frame CSVs -> {out_dir}")
    print(f"wrote ego trajectories -> {ego_path}")
    print(f"wrote stats -> {stats_path}")
    if ped.get("raw_ego_frame") and ped.get("world_frame"):
        print(
            "pedestrian median speed: "
            f"{ped['raw_ego_frame']['p50']:.2f} m/s (raw ego frame) -> "
            f"{ped['world_frame']['p50']:.2f} m/s (world frame)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
