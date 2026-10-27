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

Coordinates: scene (x, y) is right-handed; CARLA is left-handed, so
CARLA (x, y) = (x, -y) and yaw_carla = -heading. OpenDRIVE input uses the
right-handed frame and CARLA applies the same flip, so road and actors agree.

Usage:
    python scripts/viz_carla_replay.py outputs/viz/packs/idd_20220118103808_seq_18.npz
"""

from __future__ import annotations

import argparse
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
C_PRED = carla.Color(57, 135, 229)
C_GT = carla.Color(25, 158, 112)
C_TUBE = carla.Color(217, 89, 38)
C_SAMPLE = carla.Color(170, 170, 170)
C_LANE = carla.Color(235, 235, 235)
C_XWALK = carla.Color(235, 200, 60)


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


def loc(x: float, y: float, z: float = 0.0) -> carla.Location:
    return carla.Location(x=float(x), y=float(-y), z=float(z))


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


def tube_outline(debug, path: np.ndarray, radii: np.ndarray, color, z: float, life: float):
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
        polyline(debug, c + r * np.c_[np.cos(ang), np.sin(ang)], color, 0.08, z, life)
        return
    tang = np.where(nrm > 1e-3, tang / np.maximum(nrm, 1e-6), np.array([1.0, 0.0]))
    normal = np.c_[-tang[:, 1], tang[:, 0]]
    left, right = path + normal * radii[:, None], path - normal * radii[:, None]
    polyline(debug, left, color, 0.08, z, life, step=3)
    polyline(debug, right, color, 0.08, z, life, step=3)
    polyline(debug, np.vstack([left[-1], right[-1]]), color, 0.08, z, life)


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
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)

    for pack_path in args.packs:
        p = np.load(pack_path, allow_pickle=True)
        stem = Path(pack_path).stem
        c = p["center"].astype(float)
        pos = p["pos"].astype(float) - c
        ml = p["ml"].astype(float) - c
        samples = p["samples"].astype(float) - c
        heading = p["heading"].astype(float)
        ego = p["ego"].astype(float) - c
        classes = [str(k) for k in p["classes"]]
        ids = [str(k) for k in p["ids"]]
        radii = {str(k): r for k, r in zip(p["radii_classes"], p["radii"])}
        T, A, _ = pos.shape

        allxy = np.vstack([pos.reshape(-1, 2), ego])
        allxy = allxy[np.isfinite(allxy).all(axis=1)]
        lo, hi = allxy.min(axis=0) - 70.0, allxy.max(axis=0) + 70.0
        world = client.generate_opendrive_world(
            flat_xodr(lo[0], hi[0], lo[1], hi[1]),
            carla.OpendriveGenerationParameters(
                vertex_distance=2.0, max_road_length=500.0, wall_height=0.0,
                additional_width=0.6, smooth_junctions=True, enable_mesh_visibility=True))
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        lib = world.get_blueprint_library()
        debug = world.debug

        # Real lane geometry (AV2 only), drawn once and kept for the whole replay.
        for lane in p["lanes"]:
            polyline(debug, np.asarray(lane, float) - c, C_LANE, 0.10, 0.03, 0.0)
        for xw in p["crossings"]:
            xw = np.asarray(xw, float) - c
            polyline(debug, np.vstack([xw, xw[:1]]), C_XWALK, 0.10, 0.03, 0.0)

        # Spawn: every agent parked far away, physics off; teleported each frame.
        hidden = lambda i: carla.Transform(carla.Location(x=5000.0 + 15.0 * i, y=5000.0, z=-100.0))
        actors: dict[int, carla.Actor] = {}
        for a in range(A):
            bp = pick_blueprint(lib, classes[a], f"{stem}/{ids[a]}")
            if bp is None:
                continue
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "true")
            act = world.try_spawn_actor(bp, hidden(a))
            if act is not None:
                act.set_simulate_physics(False)
                actors[a] = act
        ego_bp = lib.find(EGO_BP)
        if ego_bp.has_attribute("color"):
            ego_bp.set_attribute("color", "200,30,30")
        ego_actor = world.try_spawn_actor(ego_bp, hidden(A + 1))
        ego_actor.set_simulate_physics(False)
        ego_h = smooth_heading(ego)

        cams, qs = {}, {}
        for view in ("bev", "chase"):
            cb = lib.find("sensor.camera.rgb")
            cb.set_attribute("image_size_x", str(args.width))
            cb.set_attribute("image_size_y", str(args.height))
            cb.set_attribute("fov", "90")
            cams[view] = world.spawn_actor(cb, hidden(A + 2))
            qs[view] = queue.Queue()
            cams[view].listen(qs[view].put)
            (FRAMES / stem / view).mkdir(parents=True, exist_ok=True)

        world.tick()
        try:
            for t in range(T):
                ex, ey = ego[t] if np.isfinite(ego[t]).all() else ego[np.isfinite(ego).all(axis=1)][0]
                eyaw = yaw_deg(ego_h[t])
                batch = [carla.command.ApplyTransform(
                    ego_actor.id, carla.Transform(loc(ex, ey, 0.05), carla.Rotation(yaw=eyaw)))]
                for a, act in actors.items():
                    if np.isfinite(pos[t, a]).all():
                        z = 0.95 if classes[a] in ("Pedestrian", "Child") else 0.05
                        tf = carla.Transform(loc(*pos[t, a], z), carla.Rotation(yaw=yaw_deg(heading[t, a])))
                    else:
                        tf = hidden(a)
                    batch.append(carla.command.ApplyTransform(act.id, tf))
                client.apply_batch_sync(batch, False)

                # Cameras follow the ego: top-down (ego heading = image up) and chase.
                cams["bev"].set_transform(carla.Transform(
                    loc(ex, ey, args.bev_height), carla.Rotation(pitch=-90.0, yaw=eyaw)))
                back = np.array([math.cos(ego_h[t]), math.sin(ego_h[t])])
                cx, cy = np.array([ex, ey]) - 16.0 * back
                cams["chase"].set_transform(carla.Transform(
                    loc(cx, cy, 13.0), carla.Rotation(pitch=-32.0, yaw=eyaw)))

                # Overlays for agents near the ego that have a prediction at t.
                near = []
                for a in range(A):
                    if not (np.isfinite(pos[t, a]).all() and np.isfinite(ml[t, a, 0]).all()):
                        continue
                    d = math.hypot(pos[t, a, 0] - ex, pos[t, a, 1] - ey)
                    if d > args.view_radius:
                        continue
                    moving = np.linalg.norm(ml[t, a, -1] - pos[t, a]) > 1.5
                    near.append((d, a, moving))
                near.sort()
                tubes = [a for d, a, mv in near if mv][: args.n_tube]
                for d, a, mv in near:
                    if not mv:
                        continue
                    path = np.vstack([pos[t, a], ml[t, a]])
                    polyline(debug, path, C_PRED, 0.14, 0.35, args.life, step=2)
                    gt = pos[t + 1: t + 61, a]
                    for g in gt[::5]:
                        if np.isfinite(g).all():
                            debug.draw_point(loc(*g, 0.3), size=0.09, color=C_GT, life_time=args.life)
                for a in tubes:
                    r = radii.get(classes[a], radii["__marginal__"])[: args.tube_steps]
                    tube_outline(debug, ml[t, a, : args.tube_steps], r, C_TUBE, 0.32, args.life)
                    for s in samples[t, a, :4]:
                        polyline(debug, s, C_SAMPLE, 0.04, 0.3, args.life, step=3)

                frame = world.tick()
                for view, q in qs.items():
                    img = q.get(timeout=30.0)
                    while img.frame < frame:
                        img = q.get(timeout=30.0)
                    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
                    cv2.imwrite(str(FRAMES / stem / view / f"{t:04d}.png"), arr[:, :, :3])
            print(f"{stem}: rendered {T} frames x {len(cams)} views -> {FRAMES / stem}")
        finally:
            for cam in cams.values():
                cam.stop()
                cam.destroy()
            client.apply_batch([carla.command.DestroyActor(x.id) for x in [*actors.values(), ego_actor]])
            settings.synchronous_mode = False
            world.apply_settings(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
