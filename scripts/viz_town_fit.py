"""Find where a replay pack's recorded traffic fits inside a CARLA town (scenery for the demo videos).

IDD-3D has no map, so the Indian scenes are replayed on a flat plane. To show them with footpaths and
buildings, this script searches CARLA towns for a rigid placement (rotation + translation) of the whole
recorded scene such that
  - no recorded road user, at any frame, overlaps a building, wall, fence or guard rail of the town
    (hard constraint; vehicles are given a 1.0 m radius, people and two-wheelers 0.4 m, our car 1.2 m);
  - our car's path lies on the town's driving lanes, and as many recorded vehicle positions as possible do;
  - buildings are near (so the scenery shows).
Candidates: every driving-lane waypoint (2 m apart), lateral shifts of -2 / 0 / +2 m, in the direction that
keeps the country's side of the road (--side auto: driving against a CARLA lane is driving on the left, as in
India; with the lane for the US). The traffic itself is never changed:
the same rigid transform is applied to everything. The scenery is illustrative, not data.

Needs a running CARLA 0.9.16 server. Writes outputs/viz/placements/<pack>.json (best per town plus the
chosen one).

Usage:
    python scripts/viz_town_fit.py outputs/viz/packs/cl_idd_20220118112945_seq_26.npz --towns Town10HD_Opt Town02_Opt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import carla
import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "viz" / "placements"
RES = 0.5                                                  # raster cell (m)
HARD = ("Buildings", "Walls", "Fences", "GuardRail")
VEHICLES = {"Car", "TourCar", "Van", "PoliceCar", "Truck", "Crane", "ConcreteTruck", "Bus"}


class Grid:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.x0, self.y0 = xmin, ymin
        self.w, self.h = int((xmax - xmin) / RES) + 1, int((ymax - ymin) / RES) + 1

    def zeros(self):
        return np.zeros((self.h, self.w), np.uint8)

    def idx(self, xy):
        """CARLA (x, y) -> (row, col), clipped; also a mask of points inside the grid."""
        c = np.floor((xy[:, 0] - self.x0) / RES).astype(int)
        r = np.floor((xy[:, 1] - self.y0) / RES).astype(int)
        inside = (c >= 0) & (c < self.w) & (r >= 0) & (r < self.h)
        return np.clip(r, 0, self.h - 1), np.clip(c, 0, self.w - 1), inside

    def poly(self, img, pts):
        q = np.stack([(pts[:, 0] - self.x0) / RES, (pts[:, 1] - self.y0) / RES], 1).astype(np.int32)
        cv2.fillPoly(img, [q], 1)


def town_rasters(world, cmap):
    wps = [w for w in cmap.generate_waypoints(1.0) if w.lane_type == carla.LaneType.Driving]
    xy = np.array([[w.transform.location.x, w.transform.location.y] for w in wps])
    g = Grid(*(xy.min(0) - 60), *(xy.max(0) + 60))
    road = g.zeros()
    for w in wps:
        loc, f = w.transform.location, w.transform.get_forward_vector()
        u, v = np.array([f.x, f.y]), np.array([-f.y, f.x])
        c, hw = np.array([loc.x, loc.y]), w.lane_width / 2
        g.poly(road, np.array([c + u * 0.6 + v * hw, c + u * 0.6 - v * hw, c - u * 0.6 - v * hw, c - u * 0.6 + v * hw]))
    hard, bld = g.zeros(), g.zeros()
    for lab in HARD:
        for bb in world.get_level_bbs(getattr(carla.CityObjectLabel, lab)):
            vs = bb.get_world_vertices(carla.Transform())
            pts = np.array([[p.x, p.y] for p in vs])
            hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
            g.poly(hard, hull)
            if lab == "Buildings":
                g.poly(bld, hull)
    anchors = []
    for w in cmap.generate_waypoints(2.0):
        if w.lane_type != carla.LaneType.Driving:
            continue
        loc, yaw = w.transform.location, w.transform.rotation.yaw
        anchors.append((loc.x, loc.y, loc.z, yaw))
    return g, road, hard, bld, np.array(anchors)


def dilate(img, radius_m):
    r = max(1, int(round(radius_m / RES)))
    return cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))


def fit(pack: Path, towns: list[str], client, flips=(0.0, 180.0), also: tuple = ()) -> dict:
    """also: packs of the same scene and start (e.g. the same traffic driven by another planner) whose ego paths must
    satisfy the same constraints under the same placement."""
    p = np.load(pack, allow_pickle=True)
    pos, ego, eh = p["pos"].astype(float), p["ego"].astype(float), p["ego_heading"].astype(float)
    classes = [str(c) for c in p["classes"]]
    veh = np.array([c in VEHICLES for c in classes])
    frames = np.arange(0, pos.shape[0], 3)
    e0, h0 = ego[0], float(eh[0])
    pts_v = pos[frames][:, veh].reshape(-1, 2); pts_v = pts_v[np.isfinite(pts_v).all(1)] - e0
    pts_p = pos[frames][:, ~veh].reshape(-1, 2); pts_p = pts_p[np.isfinite(pts_p).all(1)] - e0
    egos = [ego]
    for q in also:
        e2 = np.load(q, allow_pickle=True)["ego"].astype(float)
        assert np.allclose(e2[0], e0), f"{q}: another start, not the same placement"
        egos.append(e2)
    pts_e = np.vstack([e[np.isfinite(e).all(1)] for e in egos]) - e0
    res = {"pack": pack.stem, "towns": {}}
    for town in towns:
        world = client.load_world(town)
        if town.endswith("_Opt"):
            world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        g, road, hard, bld, anchors = town_rasters(world, world.get_map())
        hard_v, hard_p, hard_e = dilate(hard, 1.0), dilate(hard, 0.4), dilate(hard, 1.2)
        road_v = dilate(road, 1.0)
        near = cv2.boxFilter(bld.astype(np.float32), -1, (int(70 / RES), int(70 / RES)), normalize=True)
        best = None
        for ax, ay, az, yaw in anchors:
            for flip in flips:
                # scene frame (right-handed) -> CARLA: (x, y) -> (x, -y); heading_carla = -heading_scene
                th = math.radians(-(yaw + flip)) - h0                 # scene-frame rotation
                c, s = math.cos(th), math.sin(th)
                R = np.array([[c, -s], [s, c]])
                fwd = np.array([math.cos(-math.radians(yaw + flip)), math.sin(-math.radians(yaw + flip))])
                lat = np.array([-fwd[1], fwd[0]])
                for shift in (-2.0, 0.0, 2.0):
                    A = np.array([ax, -ay]) + shift * lat              # anchor in the scene frame
                    def carla_xy(P):
                        Q = P @ R.T + A
                        return np.stack([Q[:, 0], -Q[:, 1]], 1)
                    nh = 0
                    for P, H in ((pts_v, hard_v), (pts_p, hard_p), (pts_e, hard_e)):
                        r_, c_, ins = g.idx(carla_xy(P))
                        nh += int((H[r_, c_] & ins).sum())
                        if nh:
                            break
                    if nh:
                        continue
                    r_, c_, ins = g.idx(carla_xy(pts_e))
                    ego_road = float((road[r_, c_] & ins).mean())
                    if ego_road < 0.95:
                        continue
                    r_, c_, ins = g.idx(carla_xy(pts_v))
                    veh_road = float((road_v[r_, c_] & ins).mean()) if len(pts_v) else 1.0
                    r0, c0, _ = g.idx(carla_xy(np.zeros((1, 2))))
                    score = veh_road + 0.5 * float(near[r0[0], c0[0]])
                    if best is None or score > best["score"]:
                        best = {"town": town, "theta": th, "anchor_scene": A.tolist(), "z": az, "yaw": yaw, "flip": flip,
                                "shift": shift, "ego_on_road": ego_road, "vehicles_on_road": veh_road,
                                "building_share_70m": float(near[r0[0], c0[0]]), "score": score,
                                "scene_origin": e0.tolist()}
        res["towns"][town] = best
        print(f"{pack.stem} in {town}: " + ("no placement without overlaps" if best is None else
              f"vehicles on road {best['vehicles_on_road']:.0%}, buildings within 35 m {best['building_share_70m']:.0%}, "
              f"{'against' if best['flip'] else 'with'} the lane direction"), flush=True)
    ok = [b for b in res["towns"].values() if b]
    res["chosen"] = max(ok, key=lambda b: b["score"]) if ok else None
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packs", nargs="+")
    ap.add_argument("--towns", nargs="+", default=["Town10HD_Opt", "Town02_Opt", "Town01_Opt", "Town03_Opt", "Town05_Opt"])
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--side", default="auto", choices=["auto", "any", "left", "right"],
                    help="which side of the road our car keeps: auto = left for IDD-3D packs (India drives on the left; "
                         "against a CARLA lane), right for Argoverse 2 (with the lane)")
    ap.add_argument("--also", nargs="*", default=[],
                    help="with one pack: other packs of the same scene and start (e.g. driven without the planner); "
                         "their ego paths must fit too, and they get the same placement file")
    args = ap.parse_args()
    if args.also and len(args.packs) != 1:
        raise SystemExit("--also needs exactly one pack")
    client = carla.Client("localhost", args.port)
    client.set_timeout(180.0)
    OUT.mkdir(parents=True, exist_ok=True)
    for pk in args.packs:
        src = str(np.load(pk, allow_pickle=True)["source"])
        side = args.side if args.side != "auto" else ("left" if src == "IDD-3D" else "right")
        flips = {"any": (0.0, 180.0), "left": (180.0,), "right": (0.0,)}[side]
        r = fit(Path(pk), args.towns, client, flips, tuple(args.also))
        r["side"] = side
        for stem in [Path(pk).stem, *(Path(q).stem for q in args.also)]:
            (OUT / f"{stem}.json").write_text(json.dumps({**r, "pack": stem}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
