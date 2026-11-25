"""Diagnostic report on the IDD val halves (development): why does a model score so differently on
half A (selection) and half B (dev reporting)? Prompted by AutoBot AV2-matched: 1.794 on half A,
4.074 on half B.

For each half and each scene: windows, agent-windows, agents per window (density), class groups,
moving share (>= 0.5 m/s over the last observed second), complete-future share and mean valid
future length; and per-scene most-likely ADE for each model given, with CV as a model-free
difficulty reference. The halves are never changed and results are never averaged across them.

Aggregation is verified: each half's ADE is recomputed from the dump (masked mean over valid steps,
then over agent-windows) and must equal the run's metrics JSON; the agent-weighted mean of the
per-scene ADEs must equal the half's ADE; and the agent-weighted mean of the two halves must equal
the model's ADE over all of IDD val when that run exists.

Usage:
    python scripts/cg_halves_report.py
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

CANON = REPO / "outputs" / "canonical_world_obs20_pred60.npz"
HALVES = REPO / "splits" / "chaos_gap" / "ft" / "idd_k72_r0_dev.json"
DEV = REPO / "outputs" / "chaos_gap" / "dev"
GROUPS = {"vehicle": {"Car", "PoliceCar", "TourCar", "Van", "Bus", "Truck"},
          "two-wheeler": {"MotorcyleRider", "ScooterRider", "BicycleRider", "BicycleGroup", "Motorcycle",
                          "Scooter", "Bicycle"},
          "pedestrian": {"Pedestrian"}}
# (label, dump, its metrics JSON, the same model's metrics over all of IDD val as the chaos-gap "test")
MODELS = [
    ("AutoBot, AV2-trained (zero-shot)", DEV / "autobot_matched_on_idd_halves_preds.npz",
     DEV / "autobot_matched_on_idd_halves.json", DEV / "autobot_matched_ent8_x4.json"),
    ("AutoBot, IDD-trained (selected on half A)", DEV / "autobot_idd_ref_preds.npz", DEV / "autobot_idd_ref.json", None),
]


def group(c: str) -> str:
    return next((g for g, s in GROUPS.items() if c in s), "other")


def ade(pred, gt, fm) -> np.ndarray:
    e = np.linalg.norm(pred - gt, axis=-1)
    return np.array([r[m].mean() for r, m in zip(e, fm)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEV / "halves_report.json"))
    args = ap.parse_args()
    c = np.load(CANON, allow_pickle=True)
    m = load_splits(HALVES)
    rows, half, scene, t0 = [], [], [], []
    for want in ("val", "test"):                                        # dump order: half A then half B
        for q, (a, b) in enumerate(c["seq_start_end"]):
            if m.get(str(c["scene_id"][q])) == want:
                rows += range(a, b)
                half += ["A" if want == "val" else "B"] * (b - a)
                scene += [str(c["scene_id"][q])] * (b - a)
                t0 += [int(c["t0"][q])] * (b - a)
    rows, half, scene = np.array(rows), np.array(half), np.array(scene)
    obs = c["obs_traj"][rows].transpose(0, 2, 1).astype(np.float64)
    gt = c["pred_traj"][rows].transpose(0, 2, 1).astype(np.float64)
    fm = c["fut_mask"][rows].astype(bool)
    cls = np.array([group(str(x)) for x in c["agent_type"][rows]])
    moving = np.linalg.norm(obs[:, -1] - obs[:, -11], axis=1) >= 0.5
    cv = obs[:, -1:] + (obs[:, -1] - obs[:, -6])[:, None] / 5 * np.arange(1, 61)[None, :, None]
    per = {"CV (vel_k = 5)": ade(cv, gt, fm)}
    checks = []
    for label, dump, metrics, full in MODELS:
        if not dump.exists():
            continue
        p = np.load(dump, allow_pickle=True)
        assert np.array_equal(p["fut_mask"].astype(bool), fm) and list(p["scene_id"]) == list(scene), label
        assert np.abs(np.where(fm[..., None], p["gt"] - gt, 0)).max() < 1e-3, label
        a = ade(p["pred"].astype(np.float64), gt, fm)
        per[label] = a
        j = json.loads(metrics.read_text(encoding="utf-8"))
        for h, split in (("A", "val"), ("B", "test")):
            mine = a[half == h].mean()
            want = j[split]["most_likely"]["ade"]
            assert abs(mine - want) < 1e-3, (label, h, mine, want)
            by_scene = sum(a[scene == s].mean() * (scene == s).sum() for s in set(scene[half == h])) / (half == h).sum()
            assert abs(by_scene - mine) < 1e-9
            checks.append(f"{label}, half {h}: recomputed {mine:.4f} = JSON {want:.4f}; scene-weighted {by_scene:.4f}")
        if full is not None and full.exists():
            whole = json.loads(full.read_text(encoding="utf-8"))["test"]["most_likely"]["ade"]
            assert abs(a.mean() - whole) < 1e-3
            checks.append(f"{label}: agent-weighted mean of both halves {a.mean():.4f} = its all-IDD-val run {whole:.4f}")

    def describe(sel):
        n = int(sel.sum())
        wins = len({(s, t) for s, t, k in zip(scene, t0, sel) if k})
        return {"agent_windows": n, "windows": wins, "agents_per_window": round(n / wins, 1),
                "class_share": {g: round(float(np.mean(cls[sel] == g)), 3) for g in [*GROUPS, "other"]},
                "moving_share": round(float(moving[sel].mean()), 3),
                "complete_future_share": round(float(fm[sel].all(1).mean()), 3),
                "mean_valid_future_s": round(float(fm[sel].sum(1).mean() / 10), 2),
                "ade": {k: round(float(v[sel].mean()), 3) for k, v in per.items()}}

    report = {"halves_file": str(HALVES.relative_to(REPO)).replace("\\", "/"), "checks": checks, "halves": {}, "scenes": {},
              "by_group_and_motion": {}}
    for h in ("A", "B"):
        report["halves"][h] = {"scenes": sorted(set(scene[half == h])), **describe(half == h)}
        for s in sorted(set(scene[half == h])):
            report["scenes"][s] = {"half": h, **describe(scene == s)}
        for g in [*GROUPS, "other"]:
            for mv, name in ((True, "moving"), (False, "stationary")):
                sel = (half == h) & (cls == g) & (moving == mv)
                if sel.sum():
                    report["by_group_and_motion"].setdefault(f"{g}, {name}", {})[h] = {
                        "n": int(sel.sum()), **{k: round(float(v[sel].mean()), 3) for k, v in per.items()}}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    names = list(per)
    print("\n".join(checks))
    print(f"\n{'':28s} {'n':>5s} {'win':>4s} {'dens':>5s} {'veh':>5s} {'2w':>5s} {'ped':>5s} {'oth':>5s} {'mov':>5s} {'full':>5s} "
          + " ".join(f"{k[:22]:>22s}" for k in names))
    for label, d in [*((f"HALF {h}", report["halves"][h]) for h in "AB"), *((f"  {s} ({v['half']})", v) for s, v in report["scenes"].items())]:
        cs = d["class_share"]
        print(f"{label:28s} {d['agent_windows']:5d} {d['windows']:4d} {d['agents_per_window']:5.1f} {cs['vehicle']:5.2f} "
              f"{cs['two-wheeler']:5.2f} {cs['pedestrian']:5.2f} {cs['other']:5.2f} {d['moving_share']:5.2f} "
              f"{d['complete_future_share']:5.2f} " + " ".join(f"{d['ade'][k]:22.3f}" for k in names))
    print("\nby class group and motion (ADE per half):")
    for key, d in report["by_group_and_motion"].items():
        print(f"  {key:26s} " + " | ".join(f"{h}: n {v['n']:4d} " + " ".join(f"{v[k]:6.2f}" for k in names)
                                            for h, v in d.items()))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
