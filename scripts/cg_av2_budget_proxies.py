"""Size-matched Argoverse 2 proxies for the chaos-gap fine-tuning budget (Phase 4b, development).

The fine-tuning schedule (epochs, lr milestones) is selected on AV2 validation data, on AV2 training
sets built to match each IDD-3D fine-tuning draw (splits/chaos_gap/ft/idd_k<k>_r<r>_dev.json). No IDD
validation or test outcome is used: IDD training data defines only the target sizes and the sampling
structure.

Matched by construction, per IDD draw (k, r):
  distinct tracks     one greedy pass over the candidates in sha256(SALT-budget<r>:scene_id) order adds
                      each scenario whose tracks still fit under the IDD draw's count (one that would
                      overshoot is skipped, as in cg_build_splits.fill_exact); a remainder no candidate
                      fits is reported as the shortfall;
  windows per track   stride-3 windows ending at the canonical t0 = 30 (IDD's own sliding windows are
                      stride 3), using n_w window starts; n_w in 5..11 minimises |mean windows per track
                      - the IDD draw's| (ties: the smaller n_w).
Reported, not matched: scene count, agent-windows per scene, road-user class mix, moving share.
Acceptance gate (fixed before any run): the proxy's agent-windows and mean windows per track are each
within +-15% of its IDD draw's; any failure stops the experiment (exit 1) and nothing is adjusted.

Excluded from every proxy: the AV2-matched train scenarios. Validation sets (frozen, AV2 val in
sha256(SALT:scene_id) order): val = the first 250 (the AV2-matched val set, the only set used for any
choice); test = the next 250, a descriptive readout never used for any model, epoch, lr or budget choice.

Steps:
  1. extract  the first --candidates scenarios of each draw's order are streamed out of the AV2 train
              archive (parsed as data only; maps skipped) into data_external/av2/budget/<id>/;
  2. build    per draw: outputs/chaos_gap/budget/av2_budget_k<k>_r<r>.npz (+ _agents.npz for the MLP),
              splits/chaos_gap/budget/av2_k<k>_r<r>.json, and the match report
              outputs/chaos_gap/dev/budget_proxy_match.json.

Usage:
    python scripts/cg_av2_budget_proxies.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
CG = REPO / "outputs" / "chaos_gap"
SALT = "dreamdrive-chaosgap-v1"
KS, DRAWS = (5, 10, 20, 40, 72), 3
NW, STRIDE, TOL = range(5, 12), 3, 0.15
N_VAL = 250


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cg = _load("cg_av2_canonical")
bs = _load("cg_build_splits")
ft = _load("cg_ft_splits")


def t0s(n_w: int) -> list[int]:
    """The n_w stride-3 window starts ending at the canonical T0, ascending."""
    return [cg.T0 - STRIDE * i for i in reversed(range(n_w))]


def scenario_windows(arrays) -> dict[int, dict]:
    """{t0: window} for every stride-3 start up to T0 at which the canonical window exists."""
    out = {}
    for t0 in t0s(max(NW)):
        w = cg.scenario_window(*arrays, t0=t0)
        if w is not None:
            out[t0] = w
    return out


def fill(order: list[str], windows: dict[str, dict[int, dict]], n_w: int, target: int) -> list[str]:
    """Scenario ids in `order` whose distinct tracks (over their n_w windows) fit under the target, in one
    greedy pass: one that would overshoot is skipped; stops at the target exactly or at the end of the
    candidates (the remainder is the shortfall)."""
    keep, total, starts = [], 0, set(t0s(n_w))
    for sid in order:
        ws = [w for t0, w in windows[sid].items() if t0 in starts]
        n = len({int(a) for w in ws for a in w["agent_id"]})
        if n and total + n <= target:
            keep.append(sid)
            total += n
            if total == target:
                break
    return keep


def structure(seqs: list[tuple[str, dict]]) -> dict:
    """Size and composition of a training set given as (scene id, window) pairs."""
    per_track, per_scene, moving, groups = Counter(), Counter(), 0, Counter()
    for sid, w in seqs:
        n = len(w["agent_id"])
        per_scene[sid] += n
        for a in w["agent_id"]:
            per_track[(sid, int(a))] += 1
        obs = np.asarray(w["obs_traj"])
        moving += int((np.linalg.norm(obs[:, :, -1] - obs[:, :, -11], axis=1) >= ft.MOVING).sum())
        groups.update(ft.group(str(t)) for t in w["agent_type"])
    wpt = np.array(list(per_track.values()))
    aw = int(sum(per_scene.values()))
    return {"scenes": len(per_scene), "windows": len(seqs), "agent_windows": aw, "distinct_tracks": len(per_track),
            "windows_per_track": {"mean": float(wpt.mean()), "median": float(np.median(wpt)),
                                  "histogram": {int(k): int(v) for k, v in sorted(Counter(wpt.tolist()).items())}},
            "agent_windows_per_scene": {"mean": aw / len(per_scene), "median": float(np.median(list(per_scene.values())))},
            "moving_share": moving / aw,
            "class_groups": {g: groups[g] / aw for g in (*ft.GROUPS, "other")}}


def idd_seqs(idd, scenes: set[str]) -> list[tuple[str, dict]]:
    out = []
    for q, (a, b) in enumerate(idd["seq_start_end"]):
        if str(idd["scene_id"][q]) in scenes:
            out.append((str(idd["scene_id"][q]), {k: idd[k][a:b] for k in ("agent_id", "agent_type", "obs_traj")}))
    return out


def pool_order(av2_train_ids, excluded: set[str], r: int) -> list[str]:
    pool = [s for s in map(str, av2_train_ids) if s not in excluded]
    return ft.hash_sorted(pool, f"{SALT}-budget{r}")


def extract(tar_path: Path, wanted: set[str], out_dir: Path) -> None:
    """Stream the wanted scenarios' parquet files (data only; maps skipped) into out_dir/<id>/."""
    missing = {s for s in wanted if not (out_dir / s / f"scenario_{s}.parquet").exists()}
    if not missing:
        return
    print(f"streaming {len(missing)} scenarios out of {tar_path} ...", flush=True)
    with tarfile.open(tar_path, mode="r|*") as tar:
        for m in tar:
            if not (m.isfile() and m.name.endswith(".parquet")):
                continue
            sid = Path(m.name).stem.removeprefix("scenario_")
            if sid in missing:
                (out_dir / sid).mkdir(parents=True, exist_ok=True)
                (out_dir / sid / f"scenario_{sid}.parquet").write_bytes(tar.extractfile(m).read())
                missing.discard(sid)
                if not missing:
                    return
    if missing:
        raise SystemExit(f"{len(missing)} scenarios not found in {tar_path}")


def to_canonical(seqs: list[tuple[str, int, dict]]) -> dict:
    """Canonical arrays of (scene id, t0, window) sequences."""
    n = np.cumsum([0] + [len(w["agent_id"]) for _, _, w in seqs])
    out = {k: np.concatenate([w[k] for _, _, w in seqs], axis=0) for k in bs.KEYS}
    out["seq_start_end"] = np.stack([n[:-1], n[1:]], axis=1).astype(np.int64)
    out["scene_id"] = np.asarray([s for s, _, _ in seqs], dtype=object)
    out["t0"] = np.asarray([t for _, t, _ in seqs], dtype=np.int32)
    out["agent_type"] = out["agent_type"].astype(object)
    return out


def git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default=str(REPO / "data_external" / "av2" / "tars" / "train.tar"))
    ap.add_argument("--scenario-dir", default=str(REPO / "data_external" / "av2" / "budget"))
    ap.add_argument("--candidates", type=int, default=300, help="scenarios streamed per draw (hash order)")
    ap.add_argument("--ft-dir", default=str(REPO / "splits" / "chaos_gap" / "ft"))
    ap.add_argument("--out-dir", default=str(CG / "budget"))
    ap.add_argument("--splits-dir", default=str(REPO / "splits" / "chaos_gap" / "budget"))
    ap.add_argument("--report", default=str(CG / "dev" / "budget_proxy_match.json"))
    args = ap.parse_args()
    out_dir, sp_dir, sc_dir = Path(args.out_dir), Path(args.splits_dir), Path(args.scenario_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sp_dir.mkdir(parents=True, exist_ok=True)

    matched = json.loads((REPO / "splits" / "chaos_gap" / "av2_matched_dev.json").read_text(encoding="utf-8"))["splits"]
    av2_train_ids = np.load(CG / "av2_train.npz", allow_pickle=True)["scene_id"]      # this key only (npz is lazy)
    av2_val = np.load(CG / "av2_val.npz", allow_pickle=True)
    va_order = bs.hash_order(av2_val["scene_id"])
    val_seqs, test_seqs = va_order[:N_VAL].tolist(), va_order[N_VAL:2 * N_VAL].tolist()
    assert sorted(map(str, av2_val["scene_id"][val_seqs])) == sorted(matched["val"]), "val is not the AV2-matched val set"
    val_part, test_part = bs.take(av2_val, val_seqs), bs.take(av2_val, test_seqs)

    orders = {r: pool_order(av2_train_ids, set(matched["train"]), r) for r in range(DRAWS)}
    wanted = {s.removeprefix("av2_") for r in range(DRAWS) for s in orders[r][:args.candidates]}
    extract(Path(args.tar), wanted, sc_dir)
    windows = {}
    for f in sorted(sc_dir.glob("*/scenario_*.parquet")):
        sid = "av2_" + f.parent.name
        if f.parent.name in wanted:
            windows[sid] = scenario_windows(cg.scenario_arrays(cg.pq.read_table(f, columns=cg.COLUMNS).to_pandas()))

    idd = np.load(ft.IDD, allow_pickle=True)
    blob = git_blob(Path(__file__))
    report, failed = {"generator_git_blob": blob, "tolerance": TOL, "draws": {}}, []
    for k in KS:
        for r in range(DRAWS):
            ft_file = Path(args.ft_dir) / f"idd_k{k:02d}_r{r}_dev.json"
            target = structure(idd_seqs(idd, set(json.loads(ft_file.read_text(encoding="utf-8"))["splits"]["train"])))
            order = [s for s in orders[r][:args.candidates] if s in windows]
            fits = {}
            for n_w in NW:
                ids = fill(order, windows, n_w, target["distinct_tracks"])
                seqs = [(s, t0, windows[s][t0]) for s in ids for t0 in t0s(n_w) if t0 in windows[s]]
                fits[n_w] = (abs(structure([(s, w) for s, _, w in seqs])["windows_per_track"]["mean"]
                                 - target["windows_per_track"]["mean"]), ids, seqs)
            n_w = min(NW, key=lambda n: (fits[n][0], n))
            ids, seqs = fits[n_w][1], fits[n_w][2]
            proxy = structure([(s, w) for s, _, w in seqs])
            ratio = {q: proxy[q] / target[q] for q in ("agent_windows", "distinct_tracks")}
            ratio["windows_per_track"] = proxy["windows_per_track"]["mean"] / target["windows_per_track"]["mean"]
            ok = all(abs(ratio[q] - 1) <= TOL for q in ("agent_windows", "windows_per_track"))
            if not ok:
                failed.append(f"k={k} r={r}")
            name = f"av2_k{k:02d}_r{r}"
            data = bs.merge([to_canonical(seqs), val_part, test_part])
            npz = out_dir / f"av2_budget_k{k:02d}_r{r}.npz"
            np.savez_compressed(npz, **data)
            subprocess.run([sys.executable, str(SCRIPTS / "9_make_agent_windows_from_canonical.py"), "--input", str(npz),
                            "--output", str(npz.with_name(npz.stem + "_agents.npz")),
                            "--stats", str(npz.with_name(npz.stem + "_agents_stats.json"))],
                           cwd=REPO, check=True, stdout=subprocess.DEVNULL)
            splits = {"train": sorted(ids), "val": sorted(map(str, val_part["scene_id"])),
                      "test": sorted(map(str, test_part["scene_id"]))}
            assert not set(splits["train"]) & (set(splits["val"]) | set(splits["test"]) | set(matched["train"]))
            (sp_dir / f"{name}.json").write_text(json.dumps(
                {"method": f"AV2 train scenarios (AV2-matched train excluded) in sha256({SALT}-budget{r}:scene_id) order, "
                           f"filled to the distinct tracks of {ft_file.name}; stride-{STRIDE} windows at t0 {t0s(n_w)}; "
                           f"val = AV2 val in sha256({SALT}:scene_id) order, first {N_VAL} (the AV2-matched val set, "
                           f"the only set used for any choice); test = the next {N_VAL} (descriptive readout, never "
                           "used for any model, epoch, lr or budget choice)",
                 "salt": SALT, "note": "DEV ONLY: AV2 only; no IDD scene in this file", "k": k, "draw": r, "n_w": n_w,
                 "t0s": t0s(n_w), "idd_draw": ft_file.name, "generator_git_blob": blob,
                 "counts": {p: len(v) for p, v in splits.items()}, "splits": splits}, indent=1), encoding="utf-8")
            report["draws"][name] = {"n_w": n_w, "accepted": ok, "ratio": ratio, "idd": target, "proxy": proxy,
                                     "shortfall_tracks": target["distinct_tracks"] - proxy["distinct_tracks"]}
            print(f"k={k:2d} r={r}: n_w={n_w:2d} | tracks {proxy['distinct_tracks']:5d}/{target['distinct_tracks']:5d} | "
                  f"agent-windows {proxy['agent_windows']:5d}/{target['agent_windows']:5d} ({ratio['agent_windows']:.2f}) | "
                  f"windows/track {proxy['windows_per_track']['mean']:.2f}/{target['windows_per_track']['mean']:.2f} | "
                  f"scenes {proxy['scenes']:3d}/{target['scenes']:2d} | moving {proxy['moving_share']:.2f}/"
                  f"{target['moving_share']:.2f} | {'ok' if ok else 'FAIL'}", flush=True)
    report["accepted"] = not failed
    report["failed"] = failed
    Path(args.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.report}: " + ("all draws accepted" if not failed else f"FAILED: {', '.join(failed)}"))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
