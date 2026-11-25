"""Common-mode velocity check (development, post hoc): do all objects in a scene change velocity together?
A shared change across objects moving in different directions points at the ego-motion estimate, not at traffic.
Dev scenes only (IDD train + val); the frozen test is not read."""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(sys.argv[1]); sys.path.insert(0, str(REPO / "scripts"))
from dd_split import load_splits
ft = load_splits(REPO / "splits/chaos_gap/ft/idd_k72_r0_dev.json")
frozen = {s for s, v in load_splits(REPO / "splits/scene_splits.json").items() if v == "test"}
stats = {s["scene_id"]: s for s in json.load(open(REPO / "outputs/ego_compensation_stats.json"))["per_scene"]}
ego = defaultdict(list)
with open(REPO / "outputs/ego_trajectories.csv") as f:
    for r in csv.DictReader(f):
        if r["scene_id"] in ft:
            ego[r["scene_id"]].append((r["frame_id"], float(r["ego_x"]), float(r["ego_y"])))
LAG = 5                                      # 0.5 s
rows = []
for scene, split in sorted(ft.items()):
    assert scene not in frozen
    tr = defaultdict(dict)
    with open(REPO / "outputs/parsed_positions_world" / f"{scene}.csv") as f:
        for r in csv.DictReader(f):
            tr[r["obj_id"]][int(r["frame_id"])] = (float(r["x"]), float(r["y"]))
    frames = sorted({fr for t in tr.values() for fr in t})
    f0 = frames[0] if frames else 0
    shares, commons, n_obj, per_frame = [], [], [], []
    for t in frames:
        dv = []
        for o in tr.values():
            if all(k in o for k in (t - LAG, t, t + LAG)):
                p0, p1, p2 = (np.array(o[k]) for k in (t - LAG, t, t + LAG))
                dv.append((p2 - p1) / (LAG * 0.1) - (p1 - p0) / (LAG * 0.1))
        if len(dv) >= 4:
            dv = np.array(dv); c = dv.mean(0)
            share = len(dv) * (c @ c) / max((dv ** 2).sum(), 1e-9)
            shares.append(share); commons.append(np.linalg.norm(c)); n_obj.append(len(dv))
            per_frame.append((t - f0, len(dv), float(np.linalg.norm(c)), float(share)))
    e = sorted(ego[scene]); xy = np.array([[x, y] for _, x, y in e])
    sp = np.linalg.norm(np.diff(xy, axis=0), axis=1) * 10 if len(xy) > 1 else np.zeros(1)
    jumps = np.abs(sp[LAG:] - sp[:-LAG]) if len(sp) > LAG else np.zeros(1)
    shares, commons = np.array(shares), np.array(commons)
    big = (commons >= 2.0) & (shares >= 0.5) if len(commons) else np.zeros(0, bool)
    rows.append({"scene": scene, "split": {"train": "train", "val": "val A", "test": "val B"}[split],
                 "fallback": stats[scene]["fallback_frames"], "frames": stats[scene]["frames"],
                 "inlier": stats[scene]["mean_inlier_ratio"], "ego_speed_med": float(np.median(sp)),
                 "ego_jump_max": float(jumps.max()), "share_med": float(np.median(shares)) if len(shares) else None,
                 "n_big": int(big.sum()), "n_frames_tested": int(len(shares)),
                 "big_frames": [pf for pf, b in zip(per_frame, big) if b]})
json.dump(rows, open(sys.argv[2], "w"), indent=1)
tr_rows = [r for r in rows if r["split"] == "train" and r["n_frames_tested"]]
print("train scenes:", len(tr_rows), "| with >=1 shared jump (|common dv| >= 2 m/s per 0.5 s and share >= 0.5):",
      sum(r["n_big"] > 0 for r in tr_rows), "| total such frames:", sum(r["n_big"] for r in tr_rows),
      "of", sum(r["n_frames_tested"] for r in tr_rows))
print("train median of per-scene median share:", round(float(np.median([r["share_med"] for r in tr_rows])), 3))
print(f"{'scene':24s} {'split':6s} fallbk inlier egoV egoJump shareMed nBig/frames")
for r in sorted(rows, key=lambda r: -r["n_big"])[:14] + [r for r in rows if r["split"] != "train" and r["n_big"] == 0]:
    print(f"{r['scene']:24s} {r['split']:6s} {r['fallback']:3d}/{r['frames']:3d} {r['inlier'] or 0:5.2f} {r['ego_speed_med']:5.1f} {r['ego_jump_max']:6.1f} "
          f"{(r['share_med'] or 0):6.2f} {r['n_big']:3d}/{r['n_frames_tested']}")
s27 = next(r for r in rows if r["scene"] == "20220118103308_seq_27")
print("seq_27 shared-jump frames (frame, n objects, |common dv| m/s, share):", [tuple(round(x, 2) for x in pf) for pf in s27["big_frames"]])
