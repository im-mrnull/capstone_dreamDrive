"""Convert Argoverse 2 motion-forecasting scenarios to the canonical benchmark format.

The chaos gap (Western -> Indian transfer) trains the benchmark's models on Argoverse 2
(Wilson et al., NeurIPS 2021 Datasets & Benchmarks; licence CC BY-NC-SA 4.0, academic use
only, never redistributed) and evaluates them on IDD-3D. Both are 10 Hz, so no resampling.

One window per scenario, at Argoverse 2's own split point: observation = frames 30-49 (the
last 2 s of its 5 s history), prediction = frames 50-109 (its 6 s future). The window rule is
the canonical one of scripts/5_export_stgat_format.py (presence "obs+1fut", at least 2
agents), and the arrays are built with that script's own helpers, so the output has exactly
the keys, shapes and dtypes of outputs/canonical_world_obs20_pred60.npz:

    obs_traj / obs_traj_rel (N, 2, 20), pred_traj / pred_traj_rel (N, 2, 60) float32,
    fut_mask (N, 60) bool, seq_start_end (S, 2) int64, agent_id (N,) int32,
    agent_type (N,) object, scene_id (S,) object ("av2_<scenario id>"), t0 (S,) int32,
    plus city (S,) object.

Kept: the dynamic road users, renamed to IDD-3D classes as in scripts/viz_prepare.py.
Dropped: the ego vehicle (track "AV"; IDD-3D has no ego track either) and the static,
background, construction, riderless-bicycle and unknown types. Positions are city-frame
metres (a world frame, as IDD-3D's ego-compensated positions).

Sources: the tar archives (streamed; maps are skipped) or a directory of scenario folders
(scripts/viz_fetch_av2.py). Files are parsed as data only.

Usage:
    python scripts/cg_av2_canonical.py --src data_external/av2/tars/val.tar --out outputs/chaos_gap/av2_val.npz
    python scripts/cg_av2_canonical.py --src data_external/av2/val --out outputs/chaos_gap/av2_val120.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

AV2_CLASS = {"vehicle": "Car", "bus": "Bus", "pedestrian": "Pedestrian",
             "cyclist": "BicycleRider", "motorcyclist": "MotorcyleRider"}
EGO = "AV"
T_SCENARIO, OBS_LEN, PRED_LEN = 110, 20, 60
T0 = 30                                          # obs frames 30-49, pred frames 50-109
MIN_AGENTS = 2
COLUMNS = ["track_id", "object_type", "timestep", "position_x", "position_y", "scenario_id", "city"]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                      # its dataclass looks the module up while loading
    spec.loader.exec_module(mod)
    return mod


exp5 = _load_module("exp5", SCRIPTS / "5_export_stgat_format.py")


def scenario_arrays(df) -> tuple:
    """One scenario's kept agents as (pos (T, A, 2) float32, mask (T, A) bool, agent_id (A,)
    int32, agent_type (A,) object), agents in track-id order; the ego and non-agent types
    dropped."""
    df = df[(df["track_id"] != EGO) & df["object_type"].isin(list(AV2_CLASS))]
    ids = sorted(df["track_id"].unique(), key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s))
    col = {tid: i for i, tid in enumerate(ids)}
    pos = np.zeros((T_SCENARIO, len(ids), 2), np.float32)
    mask = np.zeros((T_SCENARIO, len(ids)), bool)
    t, a = df["timestep"].to_numpy(), df["track_id"].map(col).to_numpy()
    pos[t, a, 0], pos[t, a, 1] = df["position_x"].to_numpy(), df["position_y"].to_numpy()
    mask[t, a] = True
    first = df.drop_duplicates("track_id").set_index("track_id")["object_type"]
    agent_type = np.asarray([AV2_CLASS[first[tid]] for tid in ids], dtype=object)
    agent_id = np.asarray([int(tid) if tid.isdigit() else 1_000_000_000 + i for i, tid in enumerate(ids)], np.int32)
    return pos, mask, agent_id, agent_type


def scenario_window(pos, mask, agent_id, agent_type) -> dict | None:
    """The canonical window at T0 (scripts/5_export_stgat_format.py, presence "obs+1fut"),
    or None when fewer than MIN_AGENTS agents qualify."""
    win_mask = mask[T0:T0 + OBS_LEN + PRED_LEN]
    keep = win_mask[:OBS_LEN].all(axis=0) & win_mask[OBS_LEN:].any(axis=0)
    idx = np.where(keep)[0]
    if idx.size < MIN_AGENTS:
        return None
    traj = np.transpose(pos[T0:T0 + OBS_LEN + PRED_LEN, idx, :], (1, 2, 0)).astype(np.float32)
    obs, pred = traj[:, :, :OBS_LEN], traj[:, :, OBS_LEN:]
    fut_mask = np.transpose(win_mask[OBS_LEN:, idx], (1, 0))
    pred = np.where(fut_mask[:, None, :], pred, 0.0).astype(np.float32)
    obs_rel = np.zeros_like(obs, dtype=np.float32)
    exp5._traj_rel_inplace(obs_rel, obs)
    pred_rel = exp5._pred_rel(pred, obs)
    rel_valid = fut_mask.copy()
    rel_valid[:, 1:] &= fut_mask[:, :-1]
    pred_rel = np.where(rel_valid[:, None, :], pred_rel, 0.0).astype(np.float32)
    return {"obs_traj": obs, "pred_traj": pred, "obs_traj_rel": obs_rel, "pred_traj_rel": pred_rel,
            "fut_mask": fut_mask, "agent_id": agent_id[idx], "agent_type": agent_type[idx]}


def iter_scenarios(src: Path) -> Iterator:
    """Scenario tables from a tar archive (streamed, maps skipped) or a directory tree."""
    if src.is_dir():
        for f in sorted(src.glob("*/scenario_*.parquet")):
            yield pq.read_table(f, columns=COLUMNS).to_pandas()
        return
    with tarfile.open(src, mode="r|*") as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(".parquet"):
                yield pq.read_table(io.BytesIO(tar.extractfile(m).read()), columns=COLUMNS).to_pandas()


def convert(src: Path, limit: int | None = None) -> tuple[dict, dict]:
    parts, scene_id, city, t0 = [], [], [], []
    seen = skipped = 0
    for df in iter_scenarios(src):
        seen += 1
        win = scenario_window(*scenario_arrays(df))
        if win is None:
            skipped += 1
        else:
            parts.append(win)
            scene_id.append(f"av2_{df['scenario_id'].iloc[0]}")
            city.append(str(df["city"].iloc[0]))
            t0.append(T0)
        if limit and seen >= limit:
            break
        if seen % 10000 == 0:
            print(f"  {seen} scenarios, {len(parts)} kept", flush=True)
    if not parts:
        raise SystemExit(f"no scenario in {src} has {MIN_AGENTS} qualifying agents")
    n = np.cumsum([0] + [p["obs_traj"].shape[0] for p in parts])
    ds = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    ds.update({"seq_start_end": np.stack([n[:-1], n[1:]], axis=1).astype(np.int64),
               "scene_id": np.asarray(scene_id, dtype=object), "t0": np.asarray(t0, dtype=np.int32),
               "city": np.asarray(city, dtype=object)})
    types, counts = np.unique(ds["agent_type"], return_counts=True)
    stats = {"source": str(src), "scenarios_read": seen, "scenarios_kept": len(parts), "scenarios_skipped": skipped,
             "agents": int(n[-1]), "agents_by_type": {str(k): int(v) for k, v in zip(types, counts)},
             "window": {"t0": T0, "obs_len": OBS_LEN, "pred_len": PRED_LEN, "presence": "obs+1fut",
                        "min_agents": MIN_AGENTS}, "class_map": AV2_CLASS}
    return ds, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="an Argoverse 2 tar archive or a directory of scenario folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="stop after this many scenarios (smoke runs)")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds, stats = convert(Path(args.src), args.limit)
    np.savez_compressed(out, **ds)
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=1))
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
