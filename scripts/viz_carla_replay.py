"""Replay a prepared scene pack in CARLA and render camera frames.

Needs a running CARLA 0.9.16 server, e.g.
    C:\\CARLA\\CARLA_0.9.16\\CarlaUE4.exe -RenderOffScreen -quality-level=Epic

For each pack (scripts/viz_prepare.py) this:
  1. generates a flat OpenDRIVE world sized to the scene (CARLA's procedural
     mesh). Neither source has a CARLA map: IDD-3D roads are lane-free and
     unmapped, and Argoverse 2 maps are not CARLA maps. The road surface carries
     no markings; AV2's real lane boundaries and crosswalks are drawn on it, so
     the lane-based vs lane-free contrast comes from the data, not from CARLA;
  2. spawns every agent as a class-matched blueprint (cars, buses, trucks,
     scooters, motorbikes, bicycles, pedestrians) and the ego vehicle, physics
     off, and teleports them along their recorded tracks at 10 Hz (synchronous
     mode, fixed 0.1 s step);
  3. draws, per frame, Trajectron++'s deployable most-likely path, a few
     sampled futures and the class-conditional 90% conformal tube for agents
     near the ego, plus the ground-truth future as dots;
  4. renders a top-down (bev) and a chase camera to PNG frames.
Closed-loop packs (scripts/viz_closed_loop_pack.py) also carry the planner-driven ego's heading
and its next 3 s (drawn in pink).

Coordinates: scene (x, y) is right-handed; CARLA is left-handed, so
CARLA (x, y) = (x, -y) and yaw_carla = -heading. OpenDRIVE input uses the
right-handed frame and CARLA applies the same flip, so road and actors agree.

Usage:
    python scripts/viz_carla_replay.py outputs/viz/packs/idd_20220118103808_seq_18.npz
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import zlib
from pathlib import Path

import cv2
import numpy as np

import carla

REPO = Path(__file__).resolve().parent.parent
FRAMES = REPO / "outputs" / "viz" / "frames"

# Class -> blueprint candidates (first available is used; picked per agent id).
BLUEPRINTS = {
    "Car": ["vehicle.toyota.prius", "vehicle.nissan.micra", "vehicle.audi.a2",
            "vehicle.mini.cooper_s", "vehicle.citroen.c3", "vehicle.seat.leon"],
    "TourCar": ["vehicle.nissan.patrol", "vehicle.volkswagen.t2", "vehicle.jeep.wrangler_rubicon"],
    "PoliceCar": ["vehicle.dodge.charger_police"],
    "Van": ["vehicle.mercedes.sprinter", "vehicle.volkswagen.t2"],
    "Truck": ["vehicle.carlamotors.carlacola", "vehicle.carlamotors.european_hgv"],
    "ConcreteTruck": ["vehicle.carlamotors.european_hgv"],
    "Crane": ["vehicle.carlamotors.firetruck"],
    "Bus": ["vehicle.mitsubishi.fusorosa"],
    "MotorcyleRider": ["vehicle.yamaha.yzf", "vehicle.kawasaki.ninja", "vehicle.harley-davidson.low_rider"],
    "Motorcycle": ["vehicle.yamaha.yzf", "vehicle.kawasaki.ninja"],
    "ScooterRider": ["vehicle.vespa.zx125"],
    "Scooter": ["vehicle.vespa.zx125"],
    "BicycleRider": ["vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets"],
    "Bicycle": ["vehicle.bh.crossbike", "vehicle.gazelle.omafiets"],
    "BicycleGroup": ["vehicle.bh.crossbike"],
    "Pedestrian": ["walker.pedestrian.*"],
    "Child": ["walker.pedestrian.0009", "walker.pedestrian.0010", "walker.pedestrian.*"],
}
EGO_BP = "vehicle.lincoln.mkz_2020"
VRU = {"Pedestrian", "Child", "MotorcyleRider", "ScooterRider", "BicycleRider", "Bicycle",
       "Motorcycle", "Scooter", "BicycleGroup", "Animal"}

# Overlay colours: validated categorical slots (dataviz palette, dark-surface
# steps since the road is dark): blue = prediction, aqua = ground truth,
# orange = 90% tube, neutral grey = sampled futures, white = AV2 lane lines.
# Debug shapes are emissive and the camera auto-exposes the grey ground, so bright colours clip to white
# (seen 2026-11-24); these darker, saturated values keep the legend's hues after tone-mapping.
C_PRED = carla.Color(0, 45, 170)
C_GT = carla.Color(0, 120, 50)
C_TUBE = carla.Color(190, 70, 15)
C_SAMPLE = carla.Color(90, 90, 90)
C_LANE = carla.Color(235, 235, 235)
C_XWALK = carla.Color(235, 200, 60)
C_PLAN = carla.Color(170, 0, 85)        # closed-loop packs: the ego planner's next 3 s
# --overlay image (default): the same overlays drawn onto each camera frame after capture, projected through
# the camera's exact pose and lens, in the legend's colours (BGR, as in viz_compose.py)
BGR_PRED, BGR_GT, BGR_TUBE, BGR_SAMPLE = (229, 135, 57), (112, 158, 25), (38, 89, 217), (170, 170, 170)
BGR_PLAN = (153, 72, 236)
BGR_EGO = (30, 30, 220)
BGR_LED = (225, 225, 225)                   # LED's sampled futures (scripts/viz_planner_compare.py packs)
# the road user a closed-loop planner is reacting to (pack field focus_kind, scripts/viz_planner_compare.py)
BGR_FOCUS = {"watch": (0, 210, 255), "clear": (80, 200, 60), "threat": (0, 150, 255), "contact": (40, 40, 235)}
FOCUS_SIZE = {"Pedestrian": (0.9, 0.9), "Child": (0.8, 0.8), "Truck": (7.2, 2.8), "Bus": (9.4, 3.0)}
TWO_WHEEL = {"MotorcyleRider", "ScooterRider", "BicycleRider", "Motorcycle", "Scooter", "Bicycle", "BicycleGroup"}
NEUTRAL = ["245,245,245", "200,200,200", "150,150,150", "70,70,70", "25,25,25", "40,60,110", "110,120,130",
           "185,175,155"]                                   # other vehicles: no reds, so the ego stands out


def flat_xodr(xmin: float, xmax: float, ymin: float, ymax: float, lane_w: float = 4.0) -> str:
    """One straight road with unmarked driving lanes covering the box: a flat,
    marking-free asphalt plane in OpenDRIVE (right-handed scene frame)."""
    length = xmax - xmin
    yc = 0.5 * (ymin + ymax)
    n = max(1, int(math.ceil(0.5 * (ymax - ymin) / lane_w)))
    mark = '<roadMark sOffset="0" type="none" weight="standard" color="standard" width="0"/>'

    def lane(i):
        return (f'<lane id="{i}" type="driving" level="false"><link/>'
                f'<width sOffset="0" a="{lane_w}" b="0" c="0" d="0"/>{mark}</lane>')
    left = "".join(lane(i) for i in range(n, 0, -1))
    right = "".join(lane(-i) for i in range(1, n + 1))
    return f"""<?xml version="1.0" standalone="yes"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="dreamdrive_flat" version="1.00"
          north="0" south="0" east="0" west="0"/>
  <road name="flat" length="{length:.3f}" id="0" junction="-1">
    <link/>
    <type s="0" type="town"><speed max="50" unit="km/h"/></type>
    <planView><geometry s="0" x="{xmin:.3f}" y="{yc:.3f}" hdg="0" length="{length:.3f}"><line/></geometry></planView>
    <elevationProfile><elevation s="0" a="0" b="0" c="0" d="0"/></elevationProfile>
    <lateralProfile/>
    <lanes>
      <laneSection s="0">
        <left>{left}</left>
        <center><lane id="0" type="none" level="false"><link/>{mark}</lane></center>
        <right>{right}</right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
"""


ZOFF = 0.0                              # ground height of a town placement (0 on the generated flat world)


def loc(x: float, y: float, z: float = 0.0) -> carla.Location:
    return carla.Location(x=float(x), y=float(-y), z=float(z) + ZOFF)


def yaw_deg(heading_rad: float) -> float:
    return -math.degrees(float(heading_rad))


def pick_blueprint(lib, cls: str, key: str):
    cands = BLUEPRINTS.get(cls)
    if cands is None:
        return None                            # Animal / Misc / Unknown*: overlay only
    h = zlib.crc32(key.encode())
    for pat in cands[h % len(cands):] + cands[:h % len(cands)]:
        found = lib.filter(pat)
        if found:
            return found[h % len(found)]
    return None


def smooth_heading(xy: np.ndarray, win: int = 15, min_step: float = 0.05) -> np.ndarray:
    """Ego heading from motion, smoothed (camera stability), held when slow."""
    T = len(xy)
    h = np.zeros(T)
    last = 0.0
    for t in range(T):
        a, b = max(0, t - win // 2), min(T - 1, t + win // 2)
        d = xy[b] - xy[a]
        if np.isfinite(d).all() and np.hypot(*d) > min_step * (b - a):
            last = math.atan2(d[1], d[0])
        h[t] = last
    return h


def polyline(debug, pts: np.ndarray, color, thickness: float, z: float, life: float, step: int = 1):
    pts = pts[np.isfinite(pts).all(axis=1)][::step]
    for p, q in zip(pts[:-1], pts[1:]):
        debug.draw_line(loc(*p, z), loc(*q, z), thickness=thickness, color=color, life_time=life)


def tube_outline(debug, path: np.ndarray, radii: np.ndarray, color, z: float, life: float, th: float = 0.08):
    """Left/right boundaries of path +/- r_t along the path normal, plus an end cap."""
    ok = np.isfinite(path).all(axis=1)
    path, radii = path[ok], radii[: len(path)][ok]
    if len(path) < 2:
        return
    tang = np.gradient(path, axis=0)
    nrm = np.linalg.norm(tang, axis=1, keepdims=True)
    if float(nrm.max()) < 1e-3:                        # static agent: draw a circle
        c, r = path[-1], radii[-1]
        ang = np.linspace(0, 2 * np.pi, 25)
        polyline(debug, c + r * np.c_[np.cos(ang), np.sin(ang)], color, th, z, life)
        return
    tang = np.where(nrm > 1e-3, tang / np.maximum(nrm, 1e-6), np.array([1.0, 0.0]))
    normal = np.c_[-tang[:, 1], tang[:, 0]]
    left, right = path + normal * radii[:, None], path - normal * radii[:, None]
    polyline(debug, left, color, th, z, life, step=3)
    polyline(debug, right, color, th, z, life, step=3)
    polyline(debug, np.vstack([left[-1], right[-1]]), color, th, z, life)


def tube_sides(path: np.ndarray, radii: np.ndarray) -> list:
    """Left / right boundaries and end cap of path +/- r_t (tube_outline's geometry), or a circle if static."""
    ok = np.isfinite(path).all(axis=1)
    path, radii = path[ok], radii[: len(path)][ok]
    if len(path) < 2:
        return []
    tang = np.gradient(path, axis=0)
    nrm = np.linalg.norm(tang, axis=1, keepdims=True)
    if float(nrm.max()) < 1e-3:
        ang = np.linspace(0, 2 * np.pi, 25)
        return [path[-1] + radii[-1] * np.c_[np.cos(ang), np.sin(ang)]]
    tang = np.where(nrm > 1e-3, tang / np.maximum(nrm, 1e-6), np.array([1.0, 0.0]))
    normal = np.c_[-tang[:, 1], tang[:, 0]]
    left, right = path + normal * radii[:, None], path - normal * radii[:, None]
    return [left, right, np.vstack([left[-1], right[-1]])]


def camera_matrix(w: int, h: int, fov_deg: float) -> np.ndarray:
    f = w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def draw_items(img: np.ndarray, items: list, cam: carla.Transform, K: np.ndarray) -> None:
    """Project scene-frame (x, y) points at height z through the camera and draw them with OpenCV."""
    W2C = np.array(cam.get_inverse_matrix())
    for it in items:
        pts, z, bgr, px, dots = it[:5]
        kind = it[5] if len(it) > 5 else None
        pts = np.asarray(pts, float)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) == 0:
            continue
        world = np.c_[pts[:, 0], -pts[:, 1], np.full(len(pts), z + ZOFF), np.ones(len(pts))].T  # CARLA: y flipped
        c = W2C @ world                                       # camera frame: x forward, y right, z up
        depth = c[0]
        uv = (K @ np.vstack([c[1], -c[2], depth]))[:2] / np.where(depth > 0.1, depth, np.nan)
        uv = uv.T
        ok = np.isfinite(uv).all(axis=1) & (np.abs(uv) < 1e5).all(axis=1)
        if kind == "fill":                                    # translucent filled polygon (ego highlight)
            if ok.all():
                q = uv.astype(np.int32)
                layer = img.copy()
                cv2.fillPoly(layer, [q], bgr, cv2.LINE_AA)
                cv2.addWeighted(layer, 0.45, img, 0.55, 0, dst=img)
                cv2.polylines(img, [q], True, bgr, px, cv2.LINE_AA)
            continue
        if isinstance(kind, str) and kind.startswith("tag:"):  # text tag anchored at the first point
            if ok[0]:
                text, (x, y) = kind[4:], uv[0].astype(int)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x - tw // 2 - 6, y - th - 10), (x + tw // 2 + 6, y + 2), bgr, -1)
                ink = (20, 20, 20) if 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2] > 140 else (255, 255, 255)
                cv2.putText(img, text, (x - tw // 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ink, 2, cv2.LINE_AA)
            continue
        if dots:
            for q in uv[ok]:
                cv2.circle(img, (int(q[0]), int(q[1])), px, bgr, -1, cv2.LINE_AA)
            continue
        for i in range(len(uv) - 1):
            if ok[i] and ok[i + 1]:
                cv2.line(img, (int(uv[i, 0]), int(uv[i, 1])), (int(uv[i + 1, 0]), int(uv[i + 1, 1])), bgr, px,
                         cv2.LINE_AA)


PROP_CYCLE = ["static.prop.foodcart", "static.prop.kiosk_01", "static.prop.barrel", "static.prop.box02",
              "static.prop.trashbag", "static.prop.creasedbox01", "static.prop.foodcart", "static.prop.bin",
              "static.prop.clothcontainer", "static.prop.box03", "static.prop.plantpot04", "static.prop.creasedbox02"]


def roadside_props(world, lib, pos, ego, ego_h, n: int) -> list:
    """Town mode: street-vendor style props (carts, kiosks, barrels, boxes, bags) on the town's footpaths along
    the ego's route, only where no recorded road user ever was (> 2.5 m from every recorded position, > 4 m from
    the ego's path). Decoration, not data."""
    cmap = world.get_map()
    seen = pos[np.isfinite(pos).all(axis=-1)]
    path = ego[np.isfinite(ego).all(axis=1)]
    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(step)])
    placed, out = [], []
    for s in np.arange(-25.0, s_cum[-1] + 45.0, 6.0):
        i = int(np.clip(np.searchsorted(s_cum, s), 0, len(path) - 1))
        h = ego_h[0] if s < 0 else ego_h[min(i, len(ego_h) - 1)]
        u = np.array([math.cos(h), math.sin(h)])
        base = path[0] + s * u if s < 0 else path[i] + max(0.0, s - s_cum[-1]) * u
        nrm = np.array([-u[1], u[0]])
        for side in (1.0, -1.0):
            for lat in np.arange(3.0, 16.0, 0.75):
                q = base + side * lat * nrm
                wp = cmap.get_waypoint(loc(*q, 0.0), project_to_road=False, lane_type=carla.LaneType.Sidewalk)
                if wp is None:
                    continue
                if (np.linalg.norm(seen - q, axis=1).min() < 2.5 or np.linalg.norm(path - q, axis=1).min() < 4.0
                        or any(np.linalg.norm(q - r) < 6.0 for r in placed)):
                    continue
                name = PROP_CYCLE[len(out) % len(PROP_CYCLE)]
                bps = lib.filter(name)
                if not bps:
                    break
                face = math.atan2(*(base - q)[::-1])                # facing the road
                a = world.try_spawn_actor(bps[0], carla.Transform(
                    carla.Location(x=float(q[0]), y=float(-q[1]), z=wp.transform.location.z + 0.05),
                    carla.Rotation(yaw=yaw_deg(face))))
                if a is not None:
                    a.set_simulate_physics(False)
                    out.append(a)
                    placed.append(q)
                break
            if len(out) >= n:
                return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packs", nargs="+")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--view-radius", type=float, default=45.0, help="m around ego with overlays")
    ap.add_argument("--n-tube", type=int, default=8, help="nearest moving agents that get a tube")
    ap.add_argument("--tube-steps", type=int, default=30, help="tube horizon in frames (30 = 3 s)")
    ap.add_argument("--life", type=float, default=0.12, help="debug shape lifetime (s, sim time)")
    ap.add_argument("--bev-height", type=float, default=70.0)
    ap.add_argument("--chase-back", type=float, default=16.0, help="chase camera: m behind the ego")
    ap.add_argument("--chase-height", type=float, default=13.0)
    ap.add_argument("--chase-pitch", type=float, default=-32.0)
    ap.add_argument("--overlay", default="image", choices=["image", "carla"],
                    help="image: drawn on each frame in the legend's colours (default); carla: CARLA debug shapes "
                         "(emissive, so the camera's auto-exposure washes them towards white)")
    ap.add_argument("--placements", default=None,
                    help="directory of scripts/viz_town_fit.py results: packs with a placement are replayed inside "
                         "that CARLA town (illustrative scenery) instead of on the generated flat plane")
    ap.add_argument("--town", default=None, help="with --placements: use this town's placement, not the chosen one")
    ap.add_argument("--n-props", type=int, default=14, help="town mode: roadside vendor props on the footpaths")
    ap.add_argument("--no-tube", action="store_true", help="do not draw the 90%% bands")
    ap.add_argument("--no-gt", action="store_true", help="do not draw the true future (dots)")
    ap.add_argument("--n-samples", type=int, default=4, help="sampled futures drawn per agent (nearest --n-tube)")
    ap.add_argument("--ego-tag", default="our car", help="text tag next to the ego")
    ap.add_argument("--substeps", type=int, default=1,
                    help="rendered frames per recorded 0.1 s: positions, headings, our car and the cameras are "
                         "interpolated linearly between recorded frames (smooth video); overlays use the recorded frame")
    ap.add_argument("--ext", default="png", choices=["png", "jpg"], help="frame image format (jpg: smaller)")
    ap.add_argument("--pred-steps", type=int, default=60, help="drawn length of predicted paths / samples (frames)")
    ap.add_argument("--sample-agents", type=int, default=None,
                    help="road users with sampled futures drawn (the planner's focus first, then the nearest moving; "
                         "default --n-tube)")
    ap.add_argument("--props-ref", default=None,
                    help="town mode: place the roadside props from this pack's traffic and ego path (same scene and "
                         "placement), so several drives of one scene show the same street")
    ap.add_argument("--line-scale", type=float, default=1.0,
                    help="multiplies every overlay line width / point size (0.5 suits dense scenes)")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)

    global ZOFF
    for pack_path in args.packs:
        p = np.load(pack_path, allow_pickle=True)
        stem = Path(pack_path).stem
        pl = None
        if args.placements and (Path(args.placements) / f"{stem}.json").exists():
            fit = json.loads((Path(args.placements) / f"{stem}.json").read_text(encoding="utf-8"))
            pl = fit["towns"].get(args.town) if args.town else fit["chosen"]
        if pl:                                  # one rigid transform for everything: the traffic is unchanged
            th, A0, O = pl["theta"], np.array(pl["anchor_scene"]), np.array(pl["scene_origin"])
            Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            tf = lambda X: (np.asarray(X, float) - O) @ Rm.T + A0               # noqa: E731
            ZOFF, rot_add, out_stem = float(pl["z"]), th, f"{stem}_{pl['town']}"
        else:
            c0 = p["center"].astype(float)
            tf = lambda X: np.asarray(X, float) - c0                             # noqa: E731
            ZOFF, rot_add, out_stem = 0.0, 0.0, stem
        pos = tf(p["pos"])
        ml = tf(p["ml"])
        samples = tf(p["samples"])
        heading = p["heading"].astype(float) + rot_add
        ego = tf(p["ego"])
        plan = tf(p["ego_plan"]) if "ego_plan" in p.files else None          # viz_closed_loop_pack.py
        focus = p["focus"].astype(int) if "focus" in p.files else None        # viz_planner_compare.py
        fkind = [str(x) for x in p["focus_kind"]] if focus is not None else None
        ftag = [str(x) for x in p["focus_tag"]] if focus is not None else None
        classes = [str(k) for k in p["classes"]]
        ids = [str(k) for k in p["ids"]]
        akey = str(p["name"]) if "name" in p.files else stem      # look of each road user: per scene, not per pack
        radii = {str(k): r for k, r in zip(p["radii_classes"], p["radii"])}
        T, A, _ = pos.shape

        allxy = np.vstack([pos.reshape(-1, 2), ego])
        allxy = allxy[np.isfinite(allxy).all(axis=1)]
        lo, hi = allxy.min(axis=0) - 70.0, allxy.max(axis=0) + 70.0
        if pl:
            world = client.load_world(pl["town"])
            if pl["town"].endswith("_Opt"):
                world.unload_map_layer(carla.MapLayer.ParkedVehicles)   # the town's parked cars are not data
        else:
            world = client.generate_opendrive_world(
                flat_xodr(lo[0], hi[0], lo[1], hi[1]),
                carla.OpendriveGenerationParameters(
                    vertex_distance=2.0, max_road_length=500.0, wall_height=0.0,
                    additional_width=0.6, smooth_junctions=True, enable_mesh_visibility=True))
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)
        weather = carla.WeatherParameters.ClearNoon
        weather.fog_density, weather.fog_distance, weather.cloudiness = 0.0, 0.0, 0.0
        world.set_weather(weather)
        k = args.line_scale
        lib = world.get_blueprint_library()
        debug = world.debug

        # Real lane geometry (AV2 only), drawn once and kept for the whole replay; not in a town (its own roads).
        for lane in ([] if pl else p["lanes"]):
            polyline(debug, tf(lane), C_LANE, 0.10 * k, 0.03, 0.0)
        for xw in ([] if pl else p["crossings"]):
            xw = tf(xw)
            polyline(debug, np.vstack([xw, xw[:1]]), C_XWALK, 0.10 * k, 0.03, 0.0)

        # Spawn: every agent parked far away, physics off; teleported each frame.
        hidden = lambda i: carla.Transform(carla.Location(x=5000.0 + 15.0 * i, y=5000.0, z=-100.0))
        actors: dict[int, carla.Actor] = {}
        for a in range(A):
            bp = pick_blueprint(lib, classes[a], f"{akey}/{ids[a]}")
            if bp is None:
                continue
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "true")
            if bp.has_attribute("color"):
                bp.set_attribute("color", NEUTRAL[zlib.crc32(f"{akey}/{ids[a]}".encode()) % len(NEUTRAL)])
            act = world.try_spawn_actor(bp, hidden(a))
            if act is not None:
                act.set_simulate_physics(False)
                actors[a] = act
        ego_bp = lib.find(EGO_BP)
        if ego_bp.has_attribute("color"):
            ego_bp.set_attribute("color", "200,30,30")
        ego_actor = world.try_spawn_actor(ego_bp, hidden(A + 1))
        ego_actor.set_simulate_physics(False)
        ego_h = (p["ego_heading"].astype(float) + rot_add) if "ego_heading" in p.files else smooth_heading(ego)
        if pl and args.props_ref:
            rp = np.load(args.props_ref, allow_pickle=True)
            props = roadside_props(world, lib, tf(rp["pos"]), tf(rp["ego"]),
                                   rp["ego_heading"].astype(float) + rot_add, args.n_props)
        else:
            props = roadside_props(world, lib, pos, ego, ego_h, args.n_props) if pl else []

        cams, qs = {}, {}
        K = camera_matrix(args.width, args.height, 90.0)
        for view in ("bev", "chase"):
            cb = lib.find("sensor.camera.rgb")
            cb.set_attribute("image_size_x", str(args.width))
            cb.set_attribute("image_size_y", str(args.height))
            cb.set_attribute("fov", "90")
            # debug overlays are emissive: bloom turns them into glowing smears, so post effects are off
            for attr, val in (("bloom_intensity", "0.0"), ("lens_flare_intensity", "0.0"),
                              ("motion_blur_intensity", "0.0")):
                if cb.has_attribute(attr):
                    cb.set_attribute(attr, val)
            cams[view] = world.spawn_actor(cb, hidden(A + 2))
            qs[view] = queue.Queue()
            cams[view].listen(qs[view].put)
            (FRAMES / out_stem / view).mkdir(parents=True, exist_ok=True)

        world.tick()
        try:
            N = max(1, args.substeps)
            for step in range(T * N):
                t, al = step // N, (step % N) / N
                t2 = min(t + 1, T - 1)
                both = np.isfinite(pos[t]).all(axis=1) & np.isfinite(pos[t2]).all(axis=1)
                Pi = np.where(both[:, None], pos[t] + al * (pos[t2] - pos[t]), pos[t])         # interpolated
                dh = (heading[t2] - heading[t] + np.pi) % (2 * np.pi) - np.pi
                Hi = np.where(both & np.isfinite(dh), heading[t] + al * dh, heading[t])
                e0 = ego[t] if np.isfinite(ego[t]).all() else ego[np.isfinite(ego).all(axis=1)][0]
                e1 = ego[t2] if np.isfinite(ego[t2]).all() else e0
                ex, ey = e0 + al * (e1 - e0)
                eh = ego_h[t] + al * ((ego_h[t2] - ego_h[t] + np.pi) % (2 * np.pi) - np.pi)
                eyaw = yaw_deg(eh)
                batch = [carla.command.ApplyTransform(
                    ego_actor.id, carla.Transform(loc(ex, ey, 0.05), carla.Rotation(yaw=eyaw)))]
                for a, act in actors.items():
                    if np.isfinite(Pi[a]).all():
                        z = 0.95 if classes[a] in ("Pedestrian", "Child") else 0.05
                        tf = carla.Transform(loc(*Pi[a], z), carla.Rotation(yaw=yaw_deg(Hi[a])))
                    else:
                        tf = hidden(a)
                    batch.append(carla.command.ApplyTransform(act.id, tf))
                client.apply_batch_sync(batch, False)

                # Cameras follow the ego: top-down (ego heading = image up) and chase.
                cams["bev"].set_transform(carla.Transform(
                    loc(ex, ey, args.bev_height), carla.Rotation(pitch=-90.0, yaw=eyaw)))
                back = np.array([math.cos(eh), math.sin(eh)])
                cx, cy = np.array([ex, ey]) - args.chase_back * back
                cams["chase"].set_transform(carla.Transform(
                    loc(cx, cy, args.chase_height), carla.Rotation(pitch=args.chase_pitch, yaw=eyaw)))

                cam_tf = {"bev": carla.Transform(loc(ex, ey, args.bev_height), carla.Rotation(pitch=-90.0, yaw=eyaw)),
                          "chase": carla.Transform(loc(cx, cy, args.chase_height),
                                                   carla.Rotation(pitch=args.chase_pitch, yaw=eyaw))}
                items = []                                 # image-space overlays: (points, z, BGR, px, dots)

                def line(pts, color, bgr, width, z, px, step=1):
                    if args.overlay == "carla":
                        polyline(debug, pts, color, width, z, args.life, step=step)
                    else:
                        items.append((pts[::step] if step > 1 else pts, z, bgr, px, False))

                # Overlays for agents near the ego that have a prediction at t.
                near = []
                for a in range(A):
                    if not (np.isfinite(pos[t, a]).all() and np.isfinite(ml[t, a, 0]).all()):
                        continue
                    d = math.hypot(pos[t, a, 0] - ex, pos[t, a, 1] - ey)
                    if d > args.view_radius:
                        continue
                    fin = ml[t, a][np.isfinite(ml[t, a]).all(axis=1)]          # the last valid step: a path
                    moving = len(fin) > 0 and np.linalg.norm(fin[-1] - pos[t, a]) > 1.5   # advanced one frame ends early
                    near.append((d, a, moving))
                near.sort()
                tubes = [a for d, a, mv in near if mv][: args.n_tube]
                n_sa = args.n_tube if args.sample_agents is None else args.sample_agents
                f_a = int(focus[t]) if focus is not None and focus[t] >= 0 else None
                sampled = ([f_a] if f_a is not None else []) + [a for d, a, mv in near if mv and a != f_a]
                sampled = sampled[:n_sa]
                for d, a, mv in near:
                    if not mv:
                        continue
                    path = np.vstack([Pi[a], ml[t, a, : args.pred_steps]])
                    line(path, C_PRED, BGR_PRED, 0.14 * k, 0.35, 2, step=2)
                    gt = pos[t + 1: t + 61, a][::5]
                    gt = gt[np.isfinite(gt).all(axis=1)]
                    if args.no_gt:
                        pass
                    elif args.overlay == "carla":
                        for g in gt:
                            debug.draw_point(loc(*g, 0.3), size=0.09 * k, color=C_GT, life_time=args.life)
                    elif len(gt):
                        items.append((gt, 0.3, BGR_GT, 3, True))
                for a in tubes:
                    r = radii.get(classes[a], radii["__marginal__"])[: args.tube_steps]
                    if args.no_tube:
                        pass
                    elif args.overlay == "carla":
                        tube_outline(debug, ml[t, a, : args.tube_steps], r, C_TUBE, 0.32, args.life, 0.08 * k)
                    else:
                        for side in tube_sides(ml[t, a, : args.tube_steps], r):
                            items.append((side, 0.32, BGR_TUBE, 1, False))
                for a in sampled:
                    for s in samples[t, a, : args.n_samples, : args.pred_steps]:
                        if np.isfinite(s[0]).all():
                            line(np.vstack([Pi[a], s]), C_SAMPLE, BGR_LED if focus is not None else BGR_SAMPLE,
                                 0.04, 0.3, 1, step=3)

                if plan is not None and np.isfinite(plan[t]).all():
                    line(np.vstack([[ex, ey], plan[t]]), C_PLAN, BGR_PLAN, 0.22 * k, 0.45, 4)
                if args.overlay == "image":                # our car: highlighted footprint and a tag above it
                    hu = np.array([math.cos(eh), math.sin(eh)])
                    hv = np.array([-hu[1], hu[0]])
                    cen = np.array([ex, ey]) + 0.3 * hu
                    corners = [cen + hu * 2.6 + hv * 1.15, cen + hu * 2.6 - hv * 1.15, cen - hu * 2.6 - hv * 1.15,
                               cen - hu * 2.6 + hv * 1.15]
                    items.append((np.array(corners), 0.08, BGR_EGO, 3, False, "fill"))
                    items.append((np.array([[ex, ey]]) - 3.4 * hu, 0.3, BGR_EGO, 0, False, f"tag:{args.ego_tag}"))
                if focus is not None and focus[t] >= 0 and np.isfinite(Pi[focus[t]]).all():
                    a = focus[t]                              # the road user the planner reacts to: box, link, distance
                    col = BGR_FOCUS.get(fkind[t], BGR_FOCUS["watch"])
                    L, Wd = (2.2, 1.1) if classes[a] in TWO_WHEEL else FOCUS_SIZE.get(classes[a], (5.0, 2.3))
                    hu = np.array([math.cos(Hi[a]), math.sin(Hi[a])])
                    hv = np.array([-hu[1], hu[0]])
                    c0 = Pi[a]
                    box = np.array([c0 + hu * L / 2 + hv * Wd / 2, c0 + hu * L / 2 - hv * Wd / 2,
                                    c0 - hu * L / 2 - hv * Wd / 2, c0 - hu * L / 2 + hv * Wd / 2])
                    items.append((np.vstack([box, box[:1]]), 0.12, col, 3, False))
                    items.append((np.vstack([[ex, ey], c0]), 0.15, col, 2, False))
                    if ftag[t]:
                        items.append((c0[None], 2.4, col, 0, False, f"tag:{ftag[t]}"))

                frame = world.tick()
                for view, q in qs.items():
                    img = q.get(timeout=30.0)
                    while img.frame < frame:
                        img = q.get(timeout=30.0)
                    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
                    out = np.ascontiguousarray(arr[:, :, :3])
                    if items:
                        draw_items(out, items, cam_tf[view], K)
                    cv2.imwrite(str(FRAMES / out_stem / view / f"{step:05d}.{args.ext}"), out,
                                [cv2.IMWRITE_JPEG_QUALITY, 93] if args.ext == "jpg" else [])
            print(f"{out_stem}: rendered {T * N} frames ({T} recorded x {N}) x {len(cams)} views -> {FRAMES / out_stem}"
                  + (f" ({len(props)} roadside props)" if pl else ""))
        finally:
            for cam in cams.values():
                cam.stop()
                cam.destroy()
            client.apply_batch([carla.command.DestroyActor(x.id) for x in [*actors.values(), ego_actor, *props]])
            settings.synchronous_mode = False
            world.apply_settings(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
