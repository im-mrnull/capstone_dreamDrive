"""Fine-tuning split files for the chaos gap: k IDD-3D train scenes per file, for the fine-tuning
curve (AV2-pretrained against from-scratch, as a function of k).

For each draw r, scenes come in sha256(SALT-ft<r>:scene_id) order, so one draw's k-scene sets
are nested (5 ⊂ 10 ⊂ 20 ⊂ 40 ⊂ 72) and membership depends only on the scene ids, never on an
RNG. Only scenes with at least one agent are drawn (one IDD train scene has none), so k = 72 is
all of IDD train, the same in every draw.

IDD val is split once into two halves by balanced_halves: of all partitions into
floor/ceil-halves of its scenes, the one with the smallest imbalance, the largest of the
agent-count difference, the class-group total variation and the moving-share difference (a
criterion fixed before any fine-tuning run). Half A, the one with more agents, selects
checkpoints; half B reports during development.

    train  the first k non-empty IDD train scenes of draw r.
    val    half A of IDD val (checkpoint selection), in development and in the final runs.
    test   development: half B, so a dev number is never selected on its own scenes. --final:
           the frozen IDD test, for the pre-registered runs only.

The data file is the IDD canonical file (or its per-agent windows, for CV and the MLP). Every
benchmark script reads it unchanged through --splits-file and skips scenes a file does not list.

Writes <out-dir>/idd_k<k>_r<r>_{dev,final}.json.

--stop (Phase 4b option 2, a diagnostic): early stopping on held-out IDD *train* scenes. S = the first 8
non-empty IDD train scenes in sha256(SALT-stop:scene_id) order, used only to stop training; train = draw
r's hash order with S removed, first k (k up to 64); test = all of IDD val (both halves, split in the
analysis). No IDD val or test scene selects anything. Writes splits/chaos_gap/ft_stop/idd_k<k>_r<r>_stop.json.

Usage:
    python scripts/cg_ft_splits.py              # development files
    python scripts/cg_ft_splits.py --final      # the pre-registered runs only
    python scripts/cg_ft_splits.py --stop       # option-2 diagnostic files
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dd_split import load_splits

REPO = Path(__file__).resolve().parent.parent
SALT = "dreamdrive-chaosgap-v1"                                         # as in cg_build_splits.py
IDD = REPO / "outputs" / "canonical_world_obs20_pred60.npz"
IDD_SPLITS = REPO / "splits" / "scene_splits.json"
KS = (5, 10, 20, 40, 72)
STOP_KS, N_STOP = (5, 10, 20, 40, 64), 8
GROUPS = {"vehicle": {"Car", "Bus", "Truck", "TourCar", "Van", "PoliceCar"},
          "two-wheeler": {"MotorcyleRider", "ScooterRider", "BicycleRider", "Motorcycle", "Scooter", "Bicycle",
                          "BicycleGroup"},
          "pedestrian": {"Pedestrian"}}                                 # every other class: "other"
MOVING = 0.5                                                            # m/s over the last observed second


def hash_sorted(scenes, salt: str) -> list[str]:
    """Scene ids in ascending sha256(salt:scene_id) order."""
    return sorted(scenes, key=lambda s: hashlib.sha256(f"{salt}:{s}".encode()).hexdigest())


def group(cls: str) -> str:
    return next((g for g, members in GROUPS.items() if cls in members), "other")


def scene_stats(d) -> dict[str, Counter]:
    """Per scene id of a canonical file: agents ("n"), moving agents ("moving", at least MOVING m/s
    over the last observed second) and agents per class group."""
    out: dict[str, Counter] = {}
    for (a, b), s in zip(d["seq_start_end"], d["scene_id"]):
        obs = np.asarray(d["obs_traj"][a:b])
        speed = np.linalg.norm(obs[:, :, -1] - obs[:, :, -11], axis=1)  # metres in 1 s (10 frames)
        st = out.setdefault(str(s), Counter())
        st["n"] += int(b - a)
        st["moving"] += int((speed >= MOVING).sum())
        for t in d["agent_type"][a:b]:
            st[group(str(t))] += 1
    return out


def imbalance(A, B, stats: dict[str, Counter]) -> float:
    """The largest of the agent-count difference (share of all agents), the class-group total
    variation and the moving-share difference between two sets of scenes."""
    a, b = sum((stats[s] for s in A), Counter()), sum((stats[s] for s in B), Counter())
    if not a["n"] or not b["n"]:
        return float("inf")
    tv = 0.5 * sum(abs(a[g] / a["n"] - b[g] / b["n"]) for g in (*GROUPS, "other"))
    return max(abs(a["n"] - b["n"]) / (a["n"] + b["n"]), tv, abs(a["moving"] / a["n"] - b["moving"] / b["n"]))


def balanced_halves(scenes, stats: dict[str, Counter]) -> tuple[list[str], list[str]]:
    """(selection, reporting): the floor/ceil partition of `scenes` with the smallest imbalance
    (ties: the lexicographically first half); selection is the half with more agents."""
    scenes = sorted(scenes)
    best = min((imbalance(A, [s for s in scenes if s not in A], stats), A)
               for A in itertools.combinations(scenes, len(scenes) // 2))
    A = list(best[1])
    B = [s for s in scenes if s not in A]
    n = lambda half: sum(stats[s]["n"] for s in half)                  # noqa: E731
    return (A, B) if n(A) >= n(B) else (B, A)


def build(idd_splits: dict[str, str], stats: dict[str, Counter], ks, draws: int,
          final: bool) -> dict[tuple[int, int], dict]:
    """{(k, r): {"train", "val", "test"}} for every k and draw r."""
    by = {p: sorted(s for s, q in idd_splits.items() if q == p) for p in ("train", "val", "test")}
    selection, reporting = balanced_halves(by["val"], stats)
    val, test = (selection, by["test"]) if final else (selection, reporting)
    pool = [s for s in by["train"] if stats.get(s, Counter())["n"] > 0]
    out = {}
    for r in range(draws):
        order = hash_sorted(pool, f"{SALT}-ft{r}")
        for k in ks:
            if not 0 < k <= len(order):
                raise SystemExit(f"k={k} is outside 1..{len(order)} (non-empty IDD train scenes)")
            out[(k, r)] = {"train": sorted(order[:k]), "val": val, "test": test}
    return out


def stop_set_splits(idd_splits: dict[str, str], stats: dict[str, Counter], ks, draws: int,
                    n_stop: int = N_STOP) -> dict[tuple[int, int], dict]:
    """{(k, r): {"train", "val", "test"}} for the option-2 diagnostic: val = the stop set S (held-out
    IDD train scenes), train = draw r's order without S, test = all of IDD val."""
    by = {p: sorted(s for s, q in idd_splits.items() if q == p) for p in ("train", "val")}
    pool = [s for s in by["train"] if stats.get(s, Counter())["n"] > 0]
    stop = sorted(hash_sorted(pool, f"{SALT}-stop")[:n_stop])
    out = {}
    for r in range(draws):
        order = [s for s in hash_sorted(pool, f"{SALT}-ft{r}") if s not in stop]
        for k in ks:
            if not 0 < k <= len(order):
                raise SystemExit(f"k={k} is outside 1..{len(order)} (non-empty IDD train scenes without S)")
            out[(k, r)] = {"train": sorted(order[:k]), "val": stop, "test": by["val"]}
    return out


def composition(scenes, stats: dict[str, Counter]) -> dict:
    """Scenes, agents, agents per class group and the moving share of a set of scenes."""
    c = sum((stats[s] for s in scenes), Counter())
    return {"scenes": len(scenes), "agents": c["n"], **{g: c[g] for g in (*GROUPS, "other")},
            "moving_share": round(c["moving"] / c["n"], 4) if c["n"] else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS))
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--final", action="store_true", help="test = the frozen IDD test: the pre-registered runs only")
    ap.add_argument("--stop", action="store_true", help="option-2 diagnostic files (early stopping on held-out IDD train)")
    ap.add_argument("--idd", default=str(IDD))
    ap.add_argument("--idd-splits", default=str(IDD_SPLITS))
    ap.add_argument("--out-dir", default=str(REPO / "splits" / "chaos_gap" / "ft"))
    ap.add_argument("--stop-out-dir", default=str(REPO / "splits" / "chaos_gap" / "ft_stop"))
    args = ap.parse_args()
    if args.stop:
        return write_stop(args)
    tag = "final" if args.final else "dev"
    note = ("PRE-REGISTERED: val = IDD-3D val half A; test = the frozen IDD-3D test" if args.final
            else "DEV ONLY: val and test are halves A and B of IDD-3D val; the frozen IDD-3D test is not in this file")
    stats = scene_stats(np.load(args.idd, allow_pickle=True))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for (k, r), splits in build(load_splits(args.idd_splits), stats, args.ks, args.draws, args.final).items():
        n = {p: sum(stats[s]["n"] for s in v) for p, v in splits.items()}
        (out / f"idd_k{k:02d}_r{r}_{tag}.json").write_text(json.dumps(
            {"method": f"IDD train scenes with >= 1 agent in sha256({SALT}-ft{r}:scene_id) order, first {k}; "
                       "val = IDD val half A of balanced_halves (minimax of the agent-count difference, class-group "
                       f"TV and moving-share difference; moving >= {MOVING} m/s over the last observed second); "
                       + ("test = IDD test" if args.final else "test = IDD val half B"),
             "salt": SALT, "note": note, "k": k, "draw": r, "counts": {p: len(v) for p, v in splits.items()},
             "agents": n, "composition": {p: composition(splits[p], stats) for p in ("val", "test")},
             "splits": splits}, indent=1), encoding="utf-8")
        print(f"k={k:2d} r={r}: train {len(splits['train'])} scenes / {n['train']} agents | "
              f"val {n['val']} | test {n['test']}")
    print("saved:", out)
    return 0


def write_stop(args) -> int:
    stats = scene_stats(np.load(args.idd, allow_pickle=True))
    ks = list(STOP_KS) if args.ks == list(KS) else args.ks
    out = Path(args.stop_out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idd_splits = load_splits(args.idd_splits)
    train_all = [s for s, q in idd_splits.items() if q == "train" and stats.get(s, Counter())["n"] > 0]
    for (k, r), splits in stop_set_splits(idd_splits, stats, ks, args.draws).items():
        n = {p: sum(stats[s]["n"] for s in v) for p, v in splits.items()}
        (out / f"idd_k{k:02d}_r{r}_stop.json").write_text(json.dumps(
            {"method": f"S = the first {N_STOP} non-empty IDD train scenes in sha256({SALT}-stop:scene_id) order "
                       f"(early stopping only); train = sha256({SALT}-ft{r}:scene_id) order without S, first {k}; "
                       "test = all of IDD val (both halves)",
             "salt": SALT, "note": "DEV ONLY, option-2 diagnostic: val = held-out IDD train scenes; test = IDD-3D val; "
                                   "the frozen IDD-3D test is not in this file",
             "k": k, "draw": r, "counts": {p: len(v) for p, v in splits.items()}, "agents": n,
             "composition": {"stop_set": composition(splits["val"], stats), "idd_train": composition(train_all, stats)},
             "splits": splits}, indent=1), encoding="utf-8")
        print(f"k={k:2d} r={r}: train {len(splits['train'])} scenes / {n['train']} agents | stop {n['val']} | test {n['test']}")
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
