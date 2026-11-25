"""Join composed demo clips into one reel with an opening and a closing card (scripts/viz_compose.py outputs).

Cards are plain text on the title-bar colour; the clips are copied frame by frame (same size and fps required).

Usage:
    python scripts/viz_reel.py --out outputs/viz/videos/reel.mp4 clip1.mp4 clip2.mp4 \
        --open "Heading|line|line" --close "Heading|line|line"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

INK, INK2, BAR = (240, 239, 236), (195, 194, 183), (26, 26, 25)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def card(w: int, h: int, spec: str) -> np.ndarray:
    """Heading (large) and lines; a line starting with '~' is a small grey note."""
    img = np.full((h, w, 3), BAR, np.uint8)
    lines = spec.split("|")
    sizes = [(1.5, 3) if k == 0 else (0.6, 1) if ln.startswith("~") else (0.85, 2) for k, ln in enumerate(lines)]
    gaps = [90 if k == 0 else 40 if ln.startswith("~") else 52 for k, ln in enumerate(lines)]
    y = (h - sum(gaps)) // 2 + 40
    for k, ln in enumerate(lines):
        ln = ln.lstrip("~")
        sc, th = sizes[k]
        tw = cv2.getTextSize(ln, FONT, sc, th)[0][0]
        cv2.putText(img, ln, ((w - tw) // 2, y), FONT, sc, INK if k == 0 else INK2, th, cv2.LINE_AA)
        y += gaps[k]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--open", default=None)
    ap.add_argument("--close", default=None)
    ap.add_argument("--card-sec", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()
    first = imageio.get_reader(args.clips[0])
    h, w = first.get_data(0).shape[:2]
    first.close()
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8, macro_block_size=8,
                                ffmpeg_log_level="error")
    n = 0
    for spec in [args.open] if args.open else []:
        for _ in range(int(args.card_sec * args.fps)):
            writer.append_data(card(w, h, spec)); n += 1
    for c in args.clips:
        r = imageio.get_reader(c)
        for fr in r:
            assert fr.shape[:2] == (h, w), f"{c}: frame size {fr.shape[:2]} != {(h, w)}"
            writer.append_data(fr); n += 1
        r.close()
    for spec in [args.close] if args.close else []:
        for _ in range(int(args.card_sec * 1.5 * args.fps)):
            writer.append_data(card(w, h, spec)); n += 1
    writer.close()
    print(f"wrote {args.out} ({n / args.fps:.0f} s, {len(args.clips)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
