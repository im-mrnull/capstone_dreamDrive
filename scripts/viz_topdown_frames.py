"""Top-down frames from replay packs without CARLA (fallback for scripts/viz_carla_replay.py).

Draws what the CARLA replay draws, in the same colours, on a dark ground plane: AV2 lane lines and
crosswalks, every road user as a class-sized box, the ego (red), its planner's next 3 s (pink, closed-loop packs
from scripts/viz_closed_loop_pack.py), most-likely predictions (blue) for moving agents near the ego, the 90%
band (orange outline) for the nearest ones and the true future (aqua dots). The view follows the ego with its
heading pointing up, like the CARLA top-down camera. Frames go to outputs/viz/frames/<pack>/topdown/ so
scripts/viz_compose.py --view topdown composes them.

Usage:
    python scripts/viz_topdown_frames.py outputs/viz/packs/cl_idd_20220118112945_seq_26.npz ...
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FRAMES = REPO / "outputs" / "viz" / "frames"

# BGR, matching viz_carla_replay.py / viz_compose.py
PRED, GT, TUBE, PLAN, EGO = (229, 135, 57), (112, 158, 25), (38, 89, 217), (153, 72, 236), (30, 30, 200)
LANE, XWALK, GROUND, GRID = (235, 235, 235), (60, 200, 235), (38, 38, 36), (52, 52, 50)
CLASS_COL = {"vru": (190, 200, 205), "car": (150, 150, 145), "big": (120, 125, 130)}
SIZE = {"Pedestrian": (0.6, 0.6), "Child": (0.5, 0.5), "Car": (4.3, 1.8), "TourCar": (4.6, 1.9), "Van": (5.0, 2.0),
        "PoliceCar": (4.6, 1.9), "Truck": (6.8, 2.4), "Crane": (7.5, 2.6), "ConcreteTruck": (7.5, 2.6),
        "Bus": (9.0, 2.6)}
TWO_WHEEL = {"MotorcyleRider", "ScooterRider", "BicycleRider", "Motorcycle", "Scooter", "Bicycle", "BicycleGroup"}


def size_of(cls: str) -> tuple[float, float]:
    return (1.9, 0.7) if cls in TWO_WHEEL else SIZE.get(cls, (1.0, 1.0))


def colour_of(cls: str):
    if cls in TWO_WHEEL or cls in ("Pedestrian", "Child"):
        return CLASS_COL["vru"]
    return CLASS_COL["big"] if cls in ("Truck", "Bus", "Crane", "ConcreteTruck") else CLASS_COL["car"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packs", nargs="+")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--ppm", type=float, default=11.0, help="pixels per metre")
    ap.add_argument("--view-radius", type=float, default=45.0)
    ap.add_argument("--n-tube", type=int, default=8)
    ap.add_argument("--tube-steps", type=int, default=30)
    args = ap.parse_args()
    W, H, k = args.width, args.height, args.ppm

    for pack_path in args.packs:
        p = np.load(pack_path, allow_pickle=True)
        stem = Path(pack_path).stem
        pos, ml, heading = p["pos"].astype(float), p["ml"].astype(float), p["heading"].astype(float)
        ego = p["ego"].astype(float)
        ego_h = p["ego_heading"].astype(float) if "ego_heading" in p.files else np.zeros(len(ego))
        plan = p["ego_plan"].astype(float) if "ego_plan" in p.files else None
        classes = [str(c) for c in p["classes"]]
        radii = {str(c): r for c, r in zip(p["radii_classes"], p["radii"])}
        lanes = [np.asarray(l, float) for l in p["lanes"]]
        xws = [np.asarray(x, float) for x in p["crossings"]]
        T, A, _ = pos.shape
        out = FRAMES / stem / "topdown"
        out.mkdir(parents=True, exist_ok=True)
        for t in range(T):
            e, th = ego[t], ego_h[t]
            c, s = math.cos(th), math.sin(th)

            def px(P):
                """world (..., 2) -> pixel (..., 2): ego at 60% height, its heading up."""
                d = np.asarray(P, float) - e
                fwd, left = d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c
                return np.stack([W / 2 - left * k, H * 0.6 - fwd * k], -1)

            img = np.full((H, W, 3), GROUND, np.uint8)
            for g in range(-200, 201, 10):                      # 10 m grid, fixed to the world
                gx = np.floor(e[0] / 10) * 10 + g
                gy = np.floor(e[1] / 10) * 10 + g
                for a_, b_ in (((gx, e[1] - 200), (gx, e[1] + 200)), ((e[0] - 200, gy), (e[0] + 200, gy))):
                    q = px(np.array([a_, b_])).astype(int)
                    cv2.line(img, tuple(q[0]), tuple(q[1]), GRID, 1, cv2.LINE_AA)

            def poly(P, col, th_, closed=False):
                P = P[np.isfinite(P).all(axis=1)]
                if len(P) >= 2:
                    cv2.polylines(img, [px(P).astype(np.int32)], closed, col, th_, cv2.LINE_AA)

            for l in lanes:
                poly(l, LANE, 1)
            for x in xws:
                poly(x, XWALK, 1, closed=True)

            def box(cx, hd, cls, col, fill=True):
                L, Wd = size_of(cls)
                u, v = np.array([math.cos(hd), math.sin(hd)]), np.array([-math.sin(hd), math.cos(hd)])
                cs = [cx + u * L / 2 + v * Wd / 2, cx + u * L / 2 - v * Wd / 2, cx - u * L / 2 - v * Wd / 2,
                      cx - u * L / 2 + v * Wd / 2]
                q = px(np.array(cs)).astype(np.int32)
                if fill:
                    cv2.fillPoly(img, [q], col, cv2.LINE_AA)
                else:
                    cv2.polylines(img, [q], True, col, 2, cv2.LINE_AA)

            near = []
            for a in range(A):
                if not np.isfinite(pos[t, a]).all():
                    continue
                box(pos[t, a], heading[t, a], classes[a], colour_of(classes[a]))
                if np.isfinite(ml[t, a, 0]).all():
                    dd = float(np.linalg.norm(pos[t, a] - e))
                    fin = ml[t, a][np.isfinite(ml[t, a]).all(axis=1)]          # last valid step (see the replay)
                    if dd <= args.view_radius and len(fin) and np.linalg.norm(fin[-1] - pos[t, a]) > 1.5:
                        near.append((dd, a))
            near.sort()
            for dd, a in near:
                for g in pos[t + 1:t + 61, a][::5]:
                    if np.isfinite(g).all():
                        cv2.circle(img, tuple(px(g).astype(int)), 3, GT, -1, cv2.LINE_AA)
                poly(np.vstack([pos[t, a], ml[t, a]]), PRED, 2)
            for dd, a in near[: args.n_tube]:
                path = ml[t, a, : args.tube_steps]
                r = radii.get(classes[a], radii["__marginal__"])[: len(path)]
                tg = np.gradient(path, axis=0)
                nr = np.linalg.norm(tg, axis=1, keepdims=True)
                tg = np.where(nr > 1e-3, tg / np.maximum(nr, 1e-6), [1.0, 0.0])
                nv = np.c_[-tg[:, 1], tg[:, 0]]
                poly(path + nv * r[:, None], TUBE, 1)
                poly(path - nv * r[:, None], TUBE, 1)
            if plan is not None and np.isfinite(plan[t]).all():
                poly(np.vstack([e, plan[t]]), PLAN, 4)
            box(e, th, "Car", EGO)
            cv2.imwrite(str(out / f"{t:04d}.png"), img)
        print(f"{stem}: {T} top-down frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
