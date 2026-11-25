"""Freeze the fine-tuning budget rule (Phase 4b option 1, step 5, development) from the size-matched AV2
proxy runs (outputs/chaos_gap/dev/budget/<model>_k<k>_r<r>.json, built by budget_proxy_queue.sh).

No IDD validation or test outcome is used to select the training schedule. IDD training data is used only
to define the target training-budget sizes and sampling structure. All epoch and learning-rate decisions
are selected on AV2 validation data.

Per model (LED: per stage, core then initializer) and k, from the three draws' AV2-val histories:
  budget      S* = the median over draws of the optimizer step at the selected (best AV2-val) epoch;
  milestones  the lr halvings, matched by ordinal occurrence (first, second, ...): an occurrence is kept
              if it exists in >= 2 of 3 draws, at the median of its steps, and dropped at or beyond S*.
              A halving's step is the cumulative step count when it takes effect (the end of the last
              epoch at the higher lr). The MLP keeps a constant lr.
Converted to each IDD draw (splits/chaos_gap/ft/idd_k<k>_r<r>_dev.json) with that draw's own steps per
epoch, R = round(7406 / train agents) as in the IDD grid; n = train agent-windows (MLP) or train sequences:
  ceil(n * R / batch) (MLP, ST-GAT, transformer, MFP; LED with its per-stage batch), n * R (AgentFormer, one
  sequence per step), or for AutoBot, whose batches hold one sequence size, the sum over sizes m of
  ceil(count_m * R / min(batch_seqs, max_slots // m));
  epochs      max(1, round-half-up(S* / steps per epoch));
  milestones  max(1, round-half-up(step / steps per epoch)), the epoch after which the lr halves;
              one that lands at or beyond the converted epochs is dropped.
Every draw also gets the exact training flags; a schedule flag is always given, empty when no halving is
kept, so no script falls back to a val-driven schedule.
The steps-per-epoch formula is checked against every proxy run's recorded history before anything is
written; a mismatch, or a missing proxy run, stops it (exit 1). An existing rule file is never overwritten.

--verify GRID checks a finished IDD grid against a rule: the flags each run recorded, and the lr trajectory,
epochs, final steps and train repeat in its history (exit 1 on any mismatch).

Usage:
    python scripts/cg_budget_rule.py                                   # MLP, ST-GAT, transformer -> budget_rule.json
    python scripts/cg_budget_rule.py --models mfp --out outputs/chaos_gap/dev/budget_rule_mfp.json
    python scripts/cg_budget_rule.py --verify outputs/chaos_gap/dev/ft_budget --models mlp stgat transformer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from dd_split import load_splits

REPO = Path(__file__).resolve().parent.parent
CG = REPO / "outputs" / "chaos_gap"
KS, DRAWS = (5, 10, 20, 40, 72), 3
MODELS = ("mlp", "stgat", "transformer")                                # the first frozen rule, budget_rule.json
ALL_MODELS = (*MODELS, "mfp", "agentformer", "autobot", "led")
TARGET_WINDOWS = 7406                                                   # AV2-matched train agent-windows
IDD_AGENTS = REPO / "outputs" / "agent_windows_canonical_obs20_pred60.npz"
IDD_SEQS = REPO / "outputs" / "canonical_world_obs20_pred60.npz"
# per stage: its name, the history key, the selected-epoch key; the schedule flag; the epochs flag
STAGES = {"led": (("core", "core_history", "core_best_epoch"), ("init", "history", "best_epoch"))}
ONE_STAGE = (("main", "history", "best_epoch"),)
SCHEDULE_FLAG = {"stgat": "--lr-milestones", "transformer": "--lr-milestones", "mfp": "--lr-milestones",
                 "agentformer": "--lr-milestones", "autobot": "--lr-sched",
                 "core": "--core-lr-milestones", "init": "--init-lr-milestones"}
EPOCHS_FLAG = {"core": "--core-epochs", "init": "--init-epochs"}
KEEP_LAST = {"mfp": "--keep last"}                                      # the others: --select last


def stages(model: str):
    return STAGES.get(model, ONE_STAGE)


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def best_step(history: list[dict], best_epoch: int) -> int:
    """The cumulative optimizer step at the selected epoch."""
    return next(e["steps"] for e in history if e["epoch"] == best_epoch)


def halvings(history: list[dict]) -> list[dict]:
    """Every lr decrease, in order: the step it takes effect at and its factor."""
    trained = [e for e in history if e["lr"] is not None]               # epoch 0 (the starting weights) trains at none
    return [{"after_epoch": a["epoch"], "step": a["steps"], "factor": round(b["lr"] / a["lr"], 6)}
            for a, b in zip(trained, trained[1:]) if b["lr"] < a["lr"]]


def rule(runs: list[dict], constant_lr: bool, history_key: str = "history", epoch_key: str = "best_epoch") -> dict:
    """S* and the milestone steps from one model (stage) and k's draws."""
    s_star = statistics.median(best_step(m[history_key], m[epoch_key]) for m in runs)
    occ = [halvings(m[history_key]) for m in runs]
    milestones = []
    for i in range(max(map(len, occ), default=0)):
        found = [h[i] for h in occ if len(h) > i]
        if constant_lr or len(found) < 2:
            continue
        step = statistics.median(h["step"] for h in found)
        if step < s_star:
            milestones.append({"ordinal": i + 1, "draws": len(found), "step": step,
                               "factors": [h["factor"] for h in found]})
    return {"s_star": s_star, "milestones": milestones}


def convert(r: dict, steps_per_epoch: int) -> dict:
    """Epochs and MultiStepLR milestones on a run with this many steps per epoch."""
    epochs = max(1, round_half_up(r["s_star"] / steps_per_epoch))
    ms = [max(1, round_half_up(m["step"] / steps_per_epoch)) for m in r["milestones"]]
    return {"epochs": epochs, "lr_milestones": [m for m in ms if m < epochs],
            "dropped_milestones": [m for m in ms if m >= epochs], "steps": epochs * steps_per_epoch}


def train_mask(scene_id, split_file: Path) -> np.ndarray:
    """Rows (agent-windows or sequences) whose scene is in the split file's train set."""
    mapping = load_splits(split_file)
    return np.asarray([mapping.get(str(s)) == "train" for s in scene_id])


def train_count(scene_id, split_file: Path) -> int:
    return int(train_mask(scene_id, split_file).sum())


def steps_per_epoch(n: int, repeat: int, batch: int) -> int:
    return math.ceil(n * repeat / batch)


def model_steps_per_epoch(model: str, stage: str, sizes: np.ndarray, repeat: int, hyper: dict) -> int:
    """Optimizer steps per epoch. sizes: agents per train sequence (the MLP: one entry per agent-window)."""
    if model == "agentformer":                                          # one sequence per step
        return len(sizes) * repeat
    if model == "autobot":                                              # one-size batches (model9 batches())
        return sum(math.ceil(c * repeat / max(1, min(hyper["batch_seqs"], hyper["max_slots"] // m)))
                   for m, c in Counter(int(s) for s in sizes).items())
    if model == "led":
        return steps_per_epoch(len(sizes), repeat, hyper["batch_seqs" if stage == "core" else "init_batch_seqs"])
    return steps_per_epoch(len(sizes), repeat, hyper["batch_size"])


class TrainSizes:
    """Agents per train sequence (or one entry per agent-window for the MLP), by data file and split file."""

    def __init__(self):
        self.cache = {}

    def __call__(self, model: str, data: Path, split_file: Path) -> np.ndarray:
        key = (data, model == "mlp")
        if key not in self.cache:
            z = np.load(data, allow_pickle=True)
            se = None if model == "mlp" else z["seq_start_end"]
            self.cache[key] = (z["scene_id"], None if se is None else se[:, 1] - se[:, 0])
        scene_id, sizes = self.cache[key]
        mask = train_mask(scene_id, split_file)
        return np.ones(int(mask.sum()), int) if sizes is None else sizes[mask]


def flags(model: str, per_stage: dict, repeat: int) -> str:
    """The training flags for one IDD draw: epochs, an explicit (possibly empty) schedule, the repeat."""
    parts = []
    for name, *_ in stages(model):
        c = per_stage[name]
        parts.append(f"{EPOCHS_FLAG.get(name, '--epochs')} {c['epochs']}")
        sched = SCHEDULE_FLAG.get(name if model == "led" else model)
        if sched:
            parts.append(" ".join([sched, *map(str, c["lr_milestones"])]))
    parts.append(f"--train-repeat {repeat}")
    parts.append(KEEP_LAST.get(model, "--select last"))
    return " ".join(parts)


def stage_value(e: dict):
    return e.get("val_ade", e.get("val_loss"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob(path: Path) -> str:
    return subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True, cwd=REPO).stdout.strip()


def rel(path: Path) -> str:
    p = path.resolve()
    return p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else p.as_posix()


def verify(grid: Path, rule_files: dict) -> list[str]:
    """Every run of the grid against its rule: recorded flags, and the lr trajectory, epochs, final steps and
    train repeat in its history. Returns the mismatches."""
    bad, n = [], 0
    for f in sorted(grid.glob("*_k*_r*_*.json")):
        m = re.fullmatch(r"([a-z]+)_k(\d\d)_r(\d)_(pre|scr)", f.stem)
        if not m or m.group(1) not in rule_files:
            continue
        model, k, r, _ = m.groups()
        d, want = json.loads(f.read_text(encoding="utf-8")), rule_files[model]["models"][model][f"k{k}"]["idd_draws"][f"r{r}"]
        h, n = d["hyper"], n + 1
        if h.get("train_repeat") != want["train_repeat"]:
            bad.append(f"{f.stem}: train repeat {h.get('train_repeat')} != {want['train_repeat']}")
        if h.get("keep" if model == "mfp" else "select") != "last":
            bad.append(f"{f.stem}: not run with the final epoch kept")
        for name, hist_key, epoch_key in stages(model):
            c = want if name == "main" else want[name]
            hist = [e for e in d[hist_key] if e["epoch"] >= 1]
            got = [x["after_epoch"] for x in halvings(d[hist_key])]
            if got != c["lr_milestones"]:
                bad.append(f"{f.stem} {name}: lr halved after epochs {got}, rule {c['lr_milestones']}")
            if len(hist) != c["epochs"] or d[epoch_key] != c["epochs"] or hist[-1]["steps"] != c["steps"]:
                bad.append(f"{f.stem} {name}: {len(hist)} epochs (kept {d[epoch_key]}), {hist[-1]['steps']} steps; "
                           f"rule {c['epochs']} epochs, {c['steps']} steps")
            sched = SCHEDULE_FLAG.get(name if model == "led" else model)
            if sched and h.get(sched[2:].replace("-", "_")) != c["lr_milestones"]:
                bad.append(f"{f.stem} {name}: {sched} {h.get(sched[2:].replace('-', '_'))}, rule {c['lr_milestones']}"
                           " (None: the script's val-driven or default schedule ran)")
    if n == 0:
        bad.append(f"no runs of {sorted(rule_files)} in {rel(grid)}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(MODELS))
    ap.add_argument("--proxy-dir", default=str(CG / "dev" / "budget"))
    ap.add_argument("--ft-splits", default=str(REPO / "splits" / "chaos_gap" / "ft"))
    ap.add_argument("--out", default=str(CG / "dev" / "budget_rule.json"))
    ap.add_argument("--verify", default=None, help="check a finished IDD grid against the rule (--out) instead")
    args = ap.parse_args()
    proxy_dir, ft_splits, path = Path(args.proxy_dir), Path(args.ft_splits), Path(args.out)

    if args.verify:
        rules = {m: json.loads(path.read_text(encoding="utf-8")) for m in args.models}
        bad = verify(Path(args.verify), rules)
        print(*(bad or [f"every {'/'.join(args.models)} run in {args.verify} is on the rule ({rel(path)})"]), sep="\n")
        return 1 if bad else 0
    if path.exists():
        print(f"{rel(path)} exists (sha256 {sha256(path)}): a frozen rule is never overwritten")
        return 1

    files = {(m, k, r): proxy_dir / f"{m}_k{k:02d}_r{r}.json" for m in args.models for k in KS for r in range(DRAWS)}
    missing = [rel(f) for f in files.values() if not f.exists()]
    if missing:
        print(f"{len(missing)} proxy runs missing, e.g. {missing[0]}")
        return 1
    runs = {key: json.loads(f.read_text(encoding="utf-8")) for key, f in files.items()}

    # the steps-per-epoch formula, against every proxy run's recorded history
    bad, sizes_of = [], TrainSizes()
    for (m, k, r), d in runs.items():
        sizes = sizes_of(m, REPO / (d["input"] if m == "mlp" else d["data"]), REPO / d["splits_file"])
        for name, hist_key, _ in stages(m):
            h = [e for e in d[hist_key] if e["epoch"] >= 1]
            want = model_steps_per_epoch(m, name, sizes, d["hyper"]["train_repeat"], d["hyper"])
            got = {b["steps"] - a["steps"] for a, b in zip([{"steps": 0}] + h, h)}
            if got != {want}:
                bad.append(f"{m} {name} k{k:02d} r{r}: history {sorted(got)} vs formula {want}")
    if bad:
        print("steps per epoch do not match the formula:", *bad, sep="\n  ")
        return 1

    out = {"note": "DEV ONLY, Phase 4b option 1: the frozen fine-tuning budget rule. No IDD validation or test outcome "
                   "is used to select the training schedule. IDD training data is used only to define the target "
                   "training-budget sizes and sampling structure. All epoch and learning-rate decisions are selected "
                   "on AV2 validation data.",
           "method": "S* = median over 3 draws of the optimizer step at the best AV2-val epoch; lr halvings matched by "
                     "ordinal occurrence, kept if present in >= 2 of 3 draws, at the median step, dropped at or beyond "
                     "S*; MLP constant lr. Per IDD draw: epochs = max(1, round-half-up(S* / steps per epoch)), "
                     "milestones likewise (halve after that epoch), dropped at or beyond the epochs; steps per epoch = "
                     f"ceil(n * R / batch), R = round({TARGET_WINDOWS} / train agents).",
           "generator_git_blob": git_blob(Path(__file__)),
           "match_report": {"path": rel(CG / "dev" / "budget_proxy_match.json"),
                            "sha256": sha256(CG / "dev" / "budget_proxy_match.json")},
           "models": {}}
    if set(args.models) - set(MODELS):
        out["method"] += (" Steps per epoch by model: n * R for AgentFormer (one sequence per step); for AutoBot the "
                          "sum over sequence sizes m of ceil(count_m * R / min(batch_seqs, max_slots // m)); LED per "
                          "stage (core, then initializer), each with its own S*, milestones and batch. Each IDD draw "
                          "lists its exact flags; a schedule flag is always given, empty when no halving is kept.")
    for m in args.models:
        out["models"][m] = {}
        for k in KS:
            draws = [runs[(m, k, r)] for r in range(DRAWS)]
            per_stage = {name: rule(draws, constant_lr=(m == "mlp"), history_key=hk, epoch_key=ek)
                         for name, hk, ek in stages(m)}
            hyper = draws[0]["hyper"]
            idd_draws = {}
            for r in range(DRAWS):
                spl = ft_splits / f"idd_k{k:02d}_r{r}_dev.json"
                agents = json.loads(spl.read_text(encoding="utf-8"))["agents"]["train"]
                repeat = max(1, round(TARGET_WINDOWS / agents))
                sizes = sizes_of(m, IDD_AGENTS if m == "mlp" else IDD_SEQS, spl)
                conv = {}
                for name, *_ in stages(m):
                    spe = model_steps_per_epoch(m, name, sizes, repeat, hyper)
                    conv[name] = {"steps_per_epoch": spe, **convert(per_stage[name], spe)}
                entry = {"splits_file": rel(spl), "train_agents": agents, "train_repeat": repeat, "n_train": len(sizes)}
                if m in MODELS:                                         # the first rule's layout, unchanged
                    entry.update({"batch_size": hyper["batch_size"], **conv["main"]})
                else:
                    entry.update({**(conv["main"] if m not in STAGES else conv), "flags": flags(m, conv, repeat)})
                idd_draws[f"r{r}"] = entry
            proxy = {}
            for r in range(DRAWS):
                d = runs[(m, k, r)]
                rec = {"metrics": rel(files[(m, k, r)]), "sha256": sha256(files[(m, k, r)])}
                for name, hk, ek in stages(m):
                    best = d[ek]
                    val = (("best_val_ade", d["val"]["ade"]) if m in MODELS     # the first rule's layout, unchanged
                           else ("best_val", stage_value(next(e for e in d[hk] if e["epoch"] == best))))
                    st = {"best_epoch": best, "best_step": best_step(d[hk], best), val[0]: val[1],
                          "steps_per_epoch": d[hk][-1]["steps"] // d[hk][-1]["epoch"],
                          "halvings": halvings(d[hk]), "history": d[hk]}
                    rec.update(st) if name == "main" else rec.update({name: st})
                proxy[f"r{r}"] = rec
            key = {"proxy_draws": proxy, "idd_draws": idd_draws}
            out["models"][m][f"k{k:02d}"] = ({**per_stage["main"], **key} if m not in STAGES
                                             else {"stages": per_stage, **key})
            for name, *_ in stages(m):
                c = [v if name == "main" else v[name] for v in idd_draws.values()]
                row = " | ".join(f"r{r} {v['epochs']} ep {v['lr_milestones']}" for r, v in enumerate(c))
                print(f"{m:11s} {'' if name == 'main' else name + ' '}k={k:2d}: S* {per_stage[name]['s_star']:>7} "
                      f"steps, milestones {[x['step'] for x in per_stage[name]['milestones']]} | IDD {row}")
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved:", rel(path), "sha256", sha256(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
