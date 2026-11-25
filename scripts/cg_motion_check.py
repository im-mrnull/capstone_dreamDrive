"""Chaos-gap motion sanity check (development): does a model's most-likely prediction move like the
agents do? Fixed on 2026-11-03, before the Trajectron++ 10-epoch pilot it gates, after a 2-epoch
pilot predicted about 0.4 m of motion at 1 s at every speed.

On the canonical windows of one split (default val, i.e. AV2 val in the chaos-gap files):
  - speed: observed displacement over the last observed second (frames 10 -> 20), in m/s;
  - ratio: |prediction(1 s) - last observed| / |truth(1 s) - last observed|, for windows whose
    future is valid at 1 s;
  - pass: the median ratio of windows moving at >= --min-speed lies in --ratio-range, and the
    split's most-likely ADE (masked, as in the metrics JSON) is at most --max-ade.
Reads a prediction dump and its metrics JSON (whose data and split files locate the observations;
dump rows are the canonical rows, val then test, as the evaluation-contract gate checks). Prints
the ratio by speed band; exit code 0 on pass, 1 on fail.

Usage:
    python scripts/cg_motion_check.py --metrics <run>.json --dump <run>_preds.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from dd_split import load_splits  # noqa: E402

BANDS = ((0, 0.5), (0.5, 3), (3, 8), (8, 15), (15, np.inf))


def _path(p) -> Path:
    p = Path(str(p).replace("\\", "/"))
    return p if p.is_absolute() else REPO / p


def observations(data: Path, splits: Path, split: str) -> np.ndarray:
    """(N, 20, 2) observed positions of the split's canonical rows, in file order."""
    d, m = np.load(data, allow_pickle=True), load_splits(splits)
    if "seq_start_end" in d.files:                                      # canonical multi-agent format
        se = d["seq_start_end"]
        scene = np.repeat(np.asarray(d["scene_id"], dtype=object), se[:, 1] - se[:, 0])
        obs = d["obs_traj"].transpose(0, 2, 1)
    else:                                                               # per-agent windows (CV, MLP)
        scene, obs = np.asarray(d["scene_id"], dtype=object), d["obs"]
    return np.asarray(obs[[m.get(str(s)) == split for s in scene]], np.float64)


def check(obs, pred, gt, fm, min_speed=3.0, ratio_range=(0.8, 1.2), max_ade=2.24) -> dict:
    speed = np.linalg.norm(obs[:, -1] - obs[:, -11], axis=1) / 1.0
    err = np.linalg.norm(pred - gt, axis=-1)
    ade = float(np.mean([e[k].mean() for e, k in zip(err, fm)]))
    ok1 = fm[:, 9]
    ratio = np.full(len(obs), np.nan)
    true1 = np.linalg.norm(gt[ok1, 9] - obs[ok1, -1], axis=1)
    ratio[ok1] = np.linalg.norm(pred[ok1, 9] - obs[ok1, -1], axis=1) / np.maximum(true1, 1e-6)
    bands = {f"{lo:g}-{hi:g}": (int(np.sum(ok1 & (speed >= lo) & (speed < hi))),
                                float(np.nanmedian(ratio[ok1 & (speed >= lo) & (speed < hi)]))
                                if np.any(ok1 & (speed >= lo) & (speed < hi)) else None)
             for lo, hi in BANDS}
    moving = ok1 & (speed >= min_speed)
    med = float(np.median(ratio[moving])) if moving.any() else float("nan")
    return {"n": int(len(obs)), "ade": ade, "n_moving": int(moving.sum()), "median_ratio_moving": med,
            "bands": bands, "pass": bool(ratio_range[0] <= med <= ratio_range[1] and ade <= max_ade)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metrics", required=True, help="the run's metrics JSON (its data and splits_file)")
    ap.add_argument("--dump", required=True, help="the run's prediction dump (pred, gt, fut_mask, split)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--min-speed", type=float, default=3.0)
    ap.add_argument("--ratio-range", type=float, nargs=2, default=(0.8, 1.2))
    ap.add_argument("--max-ade", type=float, default=2.24)
    args = ap.parse_args()
    meta = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    p = np.load(args.dump, allow_pickle=True)
    sel = np.asarray(p["split"]) == args.split
    obs = observations(_path(meta.get("data") or meta["input"]), _path(meta["splits_file"]), args.split)
    if len(obs) != sel.sum():
        raise SystemExit(f"{sel.sum()} dump rows but {len(obs)} canonical {args.split} windows")
    r = check(obs, p["pred"][sel].astype(np.float64), p["gt"][sel].astype(np.float64), p["fut_mask"][sel].astype(bool),
              args.min_speed, tuple(args.ratio_range), args.max_ade)
    print(f"{args.split}: n {r['n']} | most-likely ADE {r['ade']:.4f} (max {args.max_ade}) | median 1 s displacement "
          f"ratio at >= {args.min_speed:g} m/s {r['median_ratio_moving']:.3f} (n {r['n_moving']}; "
          f"range {args.ratio_range[0]:g}-{args.ratio_range[1]:g})")
    print("  ratio by speed band (m/s): " + ", ".join(f"{b} {m:.2f} (n {n})" for b, (n, m) in r["bands"].items()
                                                         if m is not None))
    print("PASS" if r["pass"] else "FAIL")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
