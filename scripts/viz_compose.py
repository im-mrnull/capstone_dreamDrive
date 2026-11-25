"""Compose side-by-side videos: IDD-3D (left) vs Argoverse 2 (right).

Reads the PNG frames written by scripts/viz_carla_replay.py and the pack
metadata, and writes an H.264 MP4 per (pair, view) with a title bar, per-panel
labels, a running clock, the scene-level numbers from the pack (Trajectron++
most-likely ADE and 90% tube coverage over the whole scene) and a legend.
The shorter replay holds its last frame (IDD-3D scenes are 10 s, AV2 11 s).

Usage:
    python scripts/viz_compose.py \
        --pair idd_20220118103808_seq_18 av2_014586b7 --view chase
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

REPO = Path(__file__).resolve().parent.parent
PACKS = REPO / "outputs" / "viz" / "packs"
FRAMES = REPO / "outputs" / "viz" / "frames"
VIDEOS = REPO / "outputs" / "viz" / "videos"

# BGR versions of the overlay colours used in the replay (see viz_carla_replay.py).
BGR_PRED, BGR_GT, BGR_TUBE, BGR_SAMPLE = (229, 135, 57), (112, 158, 25), (38, 89, 217), (170, 170, 170)
BGR_PLAN, BGR_EGO = (153, 72, 236), (30, 30, 200)
BGR_LED, BGR_WATCH, BGR_CLEAR = (225, 225, 225), (0, 210, 255), (80, 200, 60)
INK, INK2, BAR = (240, 239, 236), (195, 194, 183), (26, 26, 25)
HUD_COL = {"red": (70, 70, 240), "amber": (0, 180, 255), "blue": (235, 185, 95), "green": (90, 210, 75),
           "grey": (175, 175, 175)}
HUD_H = 128                                   # dashboard strip under each panel (scripts/viz_planner_compare.py packs)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, xy, scale=0.6, color=INK, thick=1):
    cv2.putText(img, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


def panel_label(img, lines: list[str]):
    """Translucent box with the panel's source/place lines, top-left."""
    h = 14 + 26 * len(lines)
    w = 16 + max(cv2.getTextSize(s, FONT, 0.62, 1)[0][0] for s in lines)
    roi = img[10:10 + h, 10:10 + w]
    roi[:] = (roi * 0.35 + np.array(BAR) * 0.65).astype(np.uint8)
    for i, s in enumerate(lines):
        text(img, s, (18, 34 + 26 * i), 0.62, INK if i == 0 else INK2, 2 if i == 0 else 1)


def hud_strip(w: int, h: dict, totals: str) -> np.ndarray:
    """The planner's state under a panel: what it is doing (big, coloured), why (the road user it reacts to and the
    distance), speed and command, and the whole drive's numbers."""
    img = np.full((HUD_H, w, 3), (34, 34, 33), np.uint8)
    col = HUD_COL.get(h["col"], INK)
    cv2.rectangle(img, (0, 0), (7, HUD_H), col, -1)
    text(img, h["header"], (22, 28), 0.55, INK2, 1)
    text(img, h["state"], (22, 70), 1.05, col, 2)
    text(img, f"{h['speed'].strip()}   {h['cmd']}", (22, 104), 0.55, INK2, 1)
    text(img, h["line2"], (330, 34), 0.68, INK, 2)
    text(img, h["line3"], (330, 68), 0.64, col, 2)
    text(img, totals, (330, 108), 0.5, INK2, 1)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, required=True, metavar=("IDD_PACK", "AV2_PACK"))
    ap.add_argument("--view", default="chase", choices=["chase", "bev", "topdown"])
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--model", default="Trajectron++ (trained on IDD-3D only)")
    ap.add_argument("--title", default=None, help="title bar (default: the open-loop Trajectron++ one)")
    ap.add_argument("--subtitle", default=None)
    ap.add_argument("--frames", nargs=2, default=None, metavar=("LEFT", "RIGHT"),
                    help="frame directories under outputs/viz/frames (default: the pack stems); e.g. a town render "
                         "<stem>_<Town> from viz_carla_replay.py --placements")
    ap.add_argument("--notes", nargs=2, default=None, metavar=("LEFT", "RIGHT"),
                    help="replace each panel's second label line ('-' keeps it)")
    ap.add_argument("--heads", nargs=2, default=None, metavar=("LEFT", "RIGHT"),
                    help="replace each panel's first label line (the place)")
    ap.add_argument("--subtitle2", default=None, help="a second subtitle line")
    ap.add_argument("--tail", type=float, default=0.0, help="seconds the last frame is held at the end")
    ap.add_argument("--out-name", default=None, help="video file stem (default: <left>__vs__<right>_<view>)")
    ap.add_argument("--slow", type=int, default=1, help="slow motion: each frame shown this many times (clock stays "
                                                         "in recorded time)")
    ap.add_argument("--pause-collision", type=float, default=0.0,
                    help="seconds both panels freeze at the moment a drive collides (planner-compare packs)")
    ap.add_argument("--intro", default=None, help="intro card over the dimmed first frame: lines separated by '|', "
                                                  "the first is the heading")
    ap.add_argument("--intro-sec", type=float, default=4.0)
    ap.add_argument("--substeps", type=int, default=1,
                    help="rendered frames per recorded 0.1 s (viz_carla_replay.py --substeps); playback speed is "
                         "fps / (10 x substeps)")
    args = ap.parse_args()

    panels = []
    for j, stem in enumerate(args.pair):
        meta = np.load(PACKS / f"{stem}.npz", allow_pickle=True)
        m = json.loads(str(meta["metrics"]))
        fdir = args.frames[j] if args.frames else stem
        files = sorted([*(FRAMES / fdir / args.view).glob("*.png"), *(FRAMES / fdir / args.view).glob("*.jpg")])
        if not files:
            raise SystemExit(f"no frames for {fdir}/{args.view}; run viz_carla_replay.py first")
        lane = "lane-free, no map" if str(meta["source"]) == "IDD-3D" else "lane-based (lane lines from the AV2 map)"
        if args.notes and args.notes[j] != "-":
            lane = args.notes[j]
        stats = m["caption"] if "caption" in m else (f"ADE {m['tpp_most_likely_ade_m']:.2f} m  |  90% tube coverage "
                                                      f"{m['tube_coverage']:.0%}  (whole scene)")
        hud = json.loads(str(meta["hud"])) if "hud" in meta.files else None
        panels.append({"files": files, "place": args.heads[j] if args.heads else str(meta["place"]), "lane": lane,
                       "stats": stats, "planner": "ego_plan" in meta.files, "samples": meta["samples"].shape[2] > 0,
                       "hud": hud, "focus": "focus" in meta.files})
    n = max(len(p["files"]) for p in panels)
    h0, w0 = cv2.imread(str(panels[0]["files"][0])).shape[:2]
    huds = all(p["hud"] for p in panels)
    top, bottom = (94, 42 + HUD_H) if huds else (64, 56)
    W, H = 2 * w0, top + h0 + bottom
    VIDEOS.mkdir(parents=True, exist_ok=True)
    names = args.frames if args.frames else args.pair
    out = VIDEOS / f"{args.out_name or f'{names[0]}__vs__{names[1]}_{args.view}'}.mp4"
    writer = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=8, ffmpeg_log_level="error")
    N = max(1, args.substeps)
    slow_by = args.slow * 10 * N / args.fps
    ENDS = ("COLLISION", "CONTACT")                                  # a drive that ends in a contact
    pause_at = {N * next(j for j, h in enumerate(p["hud"]) if h["state"] in ENDS)
                for p in panels if huds and p["hud"][-1]["state"] in ENDS} if args.pause_collision > 0 else set()

    def compose(i: int) -> np.ndarray:
        canvas = np.full((H, W, 3), BAR, dtype=np.uint8)
        text(canvas, args.title or f"DreamDrive  |  {args.model}  |  same model, same 90% tubes (calibrated on IDD-3D)",
             (16, 28), 0.66, INK, 2)
        text(canvas, f"t = {i / N / 10:4.1f} s", (W - 150, 28), 0.66, INK, 2)      # recorded time
        if slow_by > 1.01:
            text(canvas, f"slowed {slow_by:.0f}x", (W - 150, 52), 0.5, INK2)
        text(canvas, args.subtitle or "left: chaotic mixed Indian traffic (test split)   |   right: structured US "
                     "traffic (zero-shot)", (16, 52), 0.5, INK2)
        if args.subtitle2:
            text(canvas, args.subtitle2, (16, 78), 0.5, INK2)
        for k, p in enumerate(panels):
            j = min(i, len(p["files"]) - 1)
            jd = min(i // N, len(p["hud"]) - 1) if p["hud"] else j          # the recorded frame
            img = cv2.imread(str(p["files"][j]))
            panel_label(img, [p["place"], p["lane"]] if huds else [p["place"], p["lane"], p["stats"]])
            if huds and i >= len(p["files"]) and p["hud"][-1]["state"] in ENDS:
                yb = int(h0 * 0.30)                                 # the drive ended in a collision: held frame
                band = img[yb - 36:yb + 36]                         # (above the cars, so the contact stays visible)
                st_end = p["hud"][-1]["state"]
                band[:] = (band * 0.3 + np.array(HUD_COL["red" if st_end == "COLLISION" else "amber"]) * 0.7).astype(np.uint8)
                msg = (f"COLLISION - drive ended at {(len(p['files']) / N - 4) / 10:.1f} s" if st_end == "COLLISION" else
                       f"HIT BY A ROAD USER (not at fault) at {(len(p['files']) / N - 4) / 10:.1f} s")
                tw = cv2.getTextSize(msg, FONT, 1.1, 3)[0][0]
                text(img, msg, ((w0 - tw) // 2, yb + 14), 1.1, (255, 255, 255), 3)
            canvas[top:top + h0, k * w0:(k + 1) * w0] = img
            if huds:
                canvas[top + h0:top + h0 + HUD_H, k * w0:(k + 1) * w0] = hud_strip(w0, p["hud"][jd], "whole drive: " + p["stats"])
        cv2.line(canvas, (w0, top), (w0, top + h0), BAR, 4)
        # Legend (identity is carried by the swatch + label, never colour alone).
        x, y = 16, H - 22
        if any(p["focus"] for p in panels):                      # with vs without the planner
            legend = [(BGR_EGO, "our car", 6), (BGR_PLAN, "its path (next 3 s)", 4),
                      (BGR_PRED, "ST-GAT predicted path", 4), (BGR_LED, "LED possible futures", 2),
                      (BGR_WATCH, "road user the planner is reacting to", 3), (BGR_CLEAR, "gap safe again", 3),
                      ((0, 150, 255), "in its path, no reaction (no planner)", 3)]
        else:
            legend = [(BGR_EGO, "our car (planner-driven)", 6), (BGR_PLAN, "its plan (next 3 s)", 4)]                 if any(p["planner"] for p in panels) else []
            legend += [(BGR_PRED, "predicted path (most-likely)", 4)]
            legend += [(BGR_SAMPLE, "sampled futures", 2)] if any(p["samples"] for p in panels) else []
            legend += [(BGR_TUBE, "90% calibrated band (3 s)", 3), (BGR_GT, "what really happened (dots)", 4)]
        for col, label, thick in legend:
            cv2.line(canvas, (x, y - 5), (x + 34, y - 5), col, thick, cv2.LINE_AA)
            text(canvas, label, (x + 42, y), 0.5 if huds else 0.52, INK2)
            x += (50 if huds else 60) + cv2.getTextSize(label, FONT, 0.5 if huds else 0.52, 1)[0][0]
        return canvas

    n_out = 0
    if args.intro:
        card = (compose(0) * 0.22).astype(np.uint8)
        lines = args.intro.split("|")
        y = H // 2 - 40 * len(lines)
        for k, ln in enumerate(lines):                       # '~' marks a small grey note line
            small = ln.startswith("~")
            ln = ln.lstrip("~")
            sc_, th_ = (1.4, 3) if k == 0 else (0.6, 1) if small else (0.85, 2)
            tw = cv2.getTextSize(ln, FONT, sc_, th_)[0][0]
            text(card, ln, ((W - tw) // 2, y), sc_, INK if k == 0 else INK2, th_)
            y += 80 if k == 0 else 40 if small else 50
        for _ in range(int(round(args.intro_sec * args.fps))):
            writer.append_data(cv2.cvtColor(card, cv2.COLOR_BGR2RGB)); n_out += 1
    for i in range(n):
        canvas = compose(i)
        frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        for _ in range(args.slow):
            writer.append_data(frame); n_out += 1
        if i in pause_at:                                         # freeze on the moment of the collision
            msg = "PAUSED: contact"                               # right-aligned, under the clock
            text(canvas, msg, (W - 16 - cv2.getTextSize(msg, FONT, 0.6, 2)[0][0], 80), 0.6, HUD_COL["red"], 2)
            held = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            for _ in range(int(round(args.pause_collision * args.fps))):
                writer.append_data(held); n_out += 1
    for _ in range(int(round(args.tail * args.fps))):
        writer.append_data(frame); n_out += 1
    writer.close()
    print(f"wrote {out} ({n} scene frames, {n_out} video frames = {n_out / args.fps:.1f} s, {W}x{H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
