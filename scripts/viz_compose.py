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
INK, INK2, BAR = (240, 239, 236), (195, 194, 183), (26, 26, 25)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, required=True, metavar=("IDD_PACK", "AV2_PACK"))
    ap.add_argument("--view", default="chase", choices=["chase", "bev"])
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--model", default="Trajectron++ (trained on IDD-3D only)")
    args = ap.parse_args()

    panels = []
    for stem in args.pair:
        meta = np.load(PACKS / f"{stem}.npz", allow_pickle=True)
        m = json.loads(str(meta["metrics"]))
        files = sorted((FRAMES / stem / args.view).glob("*.png"))
        if not files:
            raise SystemExit(f"no frames for {stem}/{args.view}; run viz_carla_replay.py first")
        lane = "lane-free, no map" if str(meta["source"]) == "IDD-3D" else "lane-based (lane lines from the AV2 map)"
        panels.append({"files": files, "place": str(meta["place"]), "lane": lane,
                       "stats": f"ADE {m['tpp_most_likely_ade_m']:.2f} m  |  90% tube coverage "
                                f"{m['tube_coverage']:.0%}  (whole scene)"})
    n = max(len(p["files"]) for p in panels)
    h0, w0 = cv2.imread(str(panels[0]["files"][0])).shape[:2]
    top, bottom = 64, 56
    W, H = 2 * w0, top + h0 + bottom
    VIDEOS.mkdir(parents=True, exist_ok=True)
    out = VIDEOS / f"{args.pair[0]}__vs__{args.pair[1]}_{args.view}.mp4"
    writer = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=8, ffmpeg_log_level="error")
    for i in range(n):
        canvas = np.full((H, W, 3), BAR, dtype=np.uint8)
        text(canvas, f"DreamDrive  |  {args.model}  |  same model, same 90% tubes (calibrated on IDD-3D)",
             (16, 28), 0.66, INK, 2)
        text(canvas, f"t = {i / args.fps:4.1f} s", (W - 150, 28), 0.66, INK, 2)
        text(canvas, "left: chaotic mixed Indian traffic (test split)   |   right: structured US traffic "
                     "(zero-shot)", (16, 52), 0.5, INK2)
        for k, p in enumerate(panels):
            img = cv2.imread(str(p["files"][min(i, len(p["files"]) - 1)]))
            panel_label(img, [p["place"], p["lane"], p["stats"]])
            canvas[top:top + h0, k * w0:(k + 1) * w0] = img
        cv2.line(canvas, (w0, top), (w0, top + h0), BAR, 4)
        # Legend (identity is carried by the swatch + label, never colour alone).
        x, y = 16, H - 22
        for col, label, thick in ((BGR_PRED, "predicted path (most-likely)", 4),
                                  (BGR_SAMPLE, "sampled futures", 2),
                                  (BGR_TUBE, "90% conformal tube (per class, 3 s)", 3),
                                  (BGR_GT, "ground-truth future (dots)", 4)):
            cv2.line(canvas, (x, y - 5), (x + 34, y - 5), col, thick, cv2.LINE_AA)
            text(canvas, label, (x + 42, y), 0.52, INK2)
            x += 60 + cv2.getTextSize(label, FONT, 0.52, 1)[0][0]
        writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    writer.close()
    print(f"wrote {out} ({n} frames, {W}x{H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
