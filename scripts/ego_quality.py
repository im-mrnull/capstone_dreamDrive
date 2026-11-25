"""Ego-motion quality gate for IDD-3D scenes (data quality only: no prediction or metric is read).

The ego poses in outputs/ego_trajectories.csv are not official: scripts/0_compensate_ego_motion.py estimates them
from the annotations (RANSAC over objects seen in consecutive frames, treating the largest rigid group as the static
world). Every road user's world position uses the same estimate. Found 2026-11-25 (docs/wiki/chaos-gap.md, "Ego-motion
estimate reversed"): in some scenes the estimate locks onto moving traffic, so the estimated ego travels against
its own heading and pedestrians appear to move at 5-13 m/s; in others the estimated ego speed jumps by up to
24 m/s within 0.5 s.

Gate (thresholds fixed from the dev diagnostics on 2026-11-25, before any drive on the new scenes and before the
gate was applied to the frozen test). A scene is CLEAN when all hold:
  ego-only (poses + the RANSAC log):
    reversed    False   median cos(heading, estimated travel) over frames > 1 m/s is >= 0 (>= 10 such frames), or
                        fewer than 10 such frames
    fallback    <= 10   RANSAC fallback frames (of ~100; constant-velocity or static-ego assumption)
    jump_max    <= 5.0  m/s: largest change of estimated ego speed within 0.5 s (cg_ego_common_mode's definition)
  road users (dev only; not computed with --ego-only):
    n_big       <= 5    frames where all objects change velocity together (|common dv| >= 2 m/s per 0.5 s and share
                        >= 0.5; cg_ego_common_mode's definition)
    ped_med     <= 2.0  m/s median pedestrian speed over 0.5 s (only judged with >= 20 samples)

Usage:
    python scripts/ego_quality.py --split dev                   # train + val, all checks
    python scripts/ego_quality.py --split test --ego-only       # frozen test: ego poses and the RANSAC log only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop_sim as cl                                              # noqa: E402
from dd_split import load_splits                                          # noqa: E402

REPO = cl.REPO
OUT = REPO / "outputs" / "data_quality"
STATS = REPO / "outputs" / "ego_compensation_stats.json"
GATE = {"fallback_max": 10, "jump_max": 5.0, "n_big_max": 5, "ped_med_max": 2.0, "ped_min_n": 20,
        "moving_min": 10, "lag": 5}


def ego_flags(sid: str, e, yaw_off: float, stats: dict) -> dict:
    fr = np.load(cl.SCENES / f"{sid}.npz", allow_pickle=True)["frame_id"]
    g = e[e.scene_id == sid].set_index("f").loc[[int(x) for x in fr]]
    raw = g[["ego_x", "ego_y"]].to_numpy(float)
    yaw = g.ego_yaw.to_numpy() + yaw_off
    head = np.stack([np.cos(yaw), np.sin(yaw)], 1)
    mv = (raw[2:] - raw[:-2]) / (2 * cl.DT)
    sp = np.linalg.norm(mv, axis=1)
    ok = sp > 1.0
    med_cos = float(np.median((mv[ok] * head[1:-1][ok]).sum(1) / sp[ok])) if ok.sum() else None
    step_sp = np.linalg.norm(np.diff(raw, axis=0), axis=1) / cl.DT
    L = GATE["lag"]
    jump = float(np.abs(step_sp[L:] - step_sp[:-L]).max()) if len(step_sp) > L else 0.0
    P = cl.smooth(cl.medfilt(raw))
    st = stats.get(sid)
    return {"moving": int(ok.sum()), "median_cos": med_cos,
            "reversed": bool(ok.sum() >= GATE["moving_min"] and med_cos < 0),
            "fallback": None if st is None else int(st["fallback_frames"]),
            "jump_max": round(jump, 2), "path_m": round(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), 1)}


def agent_flags(sid: str) -> dict:
    d = np.load(cl.SCENES / f"{sid}.npz", allow_pickle=True)
    pos, mask, cls = d["pos"].astype(float), d["mask"].astype(bool), np.asarray(d["agent_type"], str)
    L = GATE["lag"]
    v = []
    for a in np.where(cls == "Pedestrian")[0]:
        ok = mask[L:, a] & mask[:-L, a]
        v += list(np.linalg.norm(pos[L:, a] - pos[:-L, a], axis=1)[ok] / (L * cl.DT))
    n_big = 0
    for t in range(L, pos.shape[0] - L):
        ok = mask[t - L] & mask[t] & mask[t + L]
        if ok.sum() < 4:
            continue
        dv = (pos[t + L, ok] - pos[t, ok]) / (L * cl.DT) - (pos[t, ok] - pos[t - L, ok]) / (L * cl.DT)
        c = dv.mean(0)
        share = len(dv) * (c @ c) / max(float((dv ** 2).sum()), 1e-9)
        n_big += int(np.linalg.norm(c) >= 2.0 and share >= 0.5)
    return {"ped_n": len(v), "ped_med": round(float(np.median(v)), 2) if v else None, "n_big": n_big}


def verdict(r: dict) -> list[str]:
    bad = []
    if r["reversed"]:
        bad.append("reversed")
    if r["fallback"] is None or r["fallback"] > GATE["fallback_max"]:
        bad.append("fallback")
    if r["jump_max"] > GATE["jump_max"]:
        bad.append("jump")
    if r.get("n_big") is not None and r["n_big"] > GATE["n_big_max"]:
        bad.append("common_mode")
    if r.get("ped_med") is not None and r["ped_n"] >= GATE["ped_min_n"] and r["ped_med"] > GATE["ped_med_max"]:
        bad.append("pedestrian_speed")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--ego-only", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.ego_only:
        raise SystemExit("the frozen test is checked with --ego-only (ego poses and the RANSAC log only)")
    split = load_splits(REPO / "splits" / "scene_splits.json")
    want = {"dev": ("train", "val"), "test": ("test",)}[args.split]
    ids = sorted(s for s, v in split.items() if v in want)
    e, yaw_off = cl.ego_table()
    stats = {s["scene_id"]: s for s in json.loads(STATS.read_text(encoding="utf-8"))["per_scene"]}
    rows = []
    for sid in ids:
        r = {"scene": sid, "split": split[sid], **ego_flags(sid, e, yaw_off, stats)}
        if not args.ego_only:
            r.update(agent_flags(sid))
        r["fails"] = verdict(r)
        r["clean"] = not r["fails"]
        rows.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"ego_quality_{args.split}{'_egoonly' if args.ego_only else ''}.json"
    f.write_text(json.dumps({"gate": GATE, "ego_only": args.ego_only, "rows": rows}, indent=1), encoding="utf-8")
    for sp_ in want:
        rs = [r for r in rows if r["split"] == sp_]
        print(f"{sp_}: {sum(r['clean'] for r in rs)} of {len(rs)} clean")
        for k in ("reversed", "fallback", "jump", "common_mode", "pedestrian_speed"):
            n = [r["scene"] for r in rs if k in r["fails"]]
            if n:
                print(f"   {k}: {len(n)}  {n if len(n) <= 8 else ''}")
    print(f"-> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
