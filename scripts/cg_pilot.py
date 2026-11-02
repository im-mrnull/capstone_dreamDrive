"""Chaos-gap pilot (development only): run the benchmark models on a chaos-gap dataset for a few
epochs and measure their wall time per epoch, to fix AV2-large's size and the epoch budgets.

No benchmark script is modified: the time per epoch comes from timestamps on each script's own
epoch log lines ("epoch N", LED's "core epoch N" / "init epoch N"), read as the script runs. The
median gap between consecutive epoch lines is the per-epoch time, including that epoch's
validation pass. Each run's metrics JSON is kept (val = the AV2 val subset; test = IDD-3D val
during development), but a few-epoch run says nothing about accuracy.

CV and the MLP read the per-agent windows (scripts/9_make_agent_windows_from_canonical.py), the
others the multi-agent file, as in the benchmark.

Usage:
    python scripts/cg_pilot.py --name av2_matched_dev --epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CG = REPO / "outputs" / "chaos_gap"
EPOCH_LINE = re.compile(r"^(core |init )?epoch\s+\d+")


def commands(name: str, epochs: int, out: Path, models: list[str]) -> dict[str, list[str]]:
    py, data, agents = sys.executable, str(CG / f"{name}.npz"), str(CG / f"{name}_agents.npz")
    sp, e = str(REPO / "splits" / "chaos_gap" / f"{name}.json"), str(epochs)
    m = lambda k: str(out / f"{k}.json")                                   # noqa: E731
    gpu = ["--device", "cuda", "--seed", "0", "--data", data, "--splits-file", sp]
    all_cmds = {
        "cv": [py, "scripts/model1_baseline_cv_eval.py", "--input-npz", agents, "--splits-file", sp, "--vel-k", "5",
               "--save-metrics-json", m("cv")],
        "mlp": [py, "scripts/model2_baseline_mlp_train_eval.py", "--input", agents, "--splits-file", sp, "--seed", "0",
                "--epochs", e, "--save-metrics-json", m("mlp"), "--save-best", str(out / "mlp_best.npz")],
        "stgat": [py, "scripts/model3_stgat_train_eval.py", *gpu, "--epochs", e, "--save-metrics-json", m("stgat"),
                  "--save-ckpt", str(out / "stgat.pt")],
        "transformer": [py, "scripts/model5_transformer_train_eval.py", *gpu, "--epochs", e,
                        "--save-metrics-json", m("transformer"), "--save-ckpt", str(out / "transformer.pt")],
        "led": [py, "scripts/model7_led_train_eval.py", *gpu, "--core-epochs", e, "--init-epochs", e,
                "--save-metrics-json", m("led"), "--save-ckpt", str(out / "led.pt")],
        "autobot": [py, "scripts/model9_autobot_train_eval.py", *gpu, "--epochs", e, "--save-metrics-json", m("autobot"),
                    "--save-ckpt", str(out / "autobot.pt")],
    }
    return {k: all_cmds[k] for k in models}


def run_timed(cmd: list[str], log: Path) -> dict:
    """Run cmd, timestamping its epoch log lines; returns the timing summary."""
    t0 = time.perf_counter()
    stamps = {"": [], "core ": [], "init ": []}
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", bufsize=1,
                             env={**os.environ, "PYTHONUNBUFFERED": "1"})      # epoch lines arrive as printed
        for line in p.stdout:
            now = time.perf_counter() - t0
            f.write(f"[{now:9.2f}] {line}")
            mt = EPOCH_LINE.match(line)
            if mt:
                stamps[mt.group(1) or ""].append(now)
        p.wait()
    out = {"returncode": p.returncode, "wall_s": round(time.perf_counter() - t0, 2)}
    prev = None                                     # LED's stage 2 starts at its last stage-1 line
    for stage, ts in stamps.items():
        name = stage.strip() or "train"
        gaps = np.diff(([prev] if prev is not None and stage == "init " else []) + ts)
        if len(gaps):
            out[f"{name}_epoch_s"] = round(float(np.median(gaps)), 3)
            out[f"{name}_epoch_gaps_s"] = [round(float(g), 2) for g in gaps]
        if ts:
            out[f"{name}_epochs_seen"] = len(ts)
            prev = ts[-1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="chaos-gap dataset, e.g. av2_matched_dev")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=["cv", "mlp", "stgat", "transformer", "led", "autobot"])
    args = ap.parse_args()
    out = CG / "pilot" / args.name
    out.mkdir(parents=True, exist_ok=True)
    agents = CG / f"{args.name}_agents.npz"
    if not agents.exists() or agents.stat().st_mtime < (CG / f"{args.name}.npz").stat().st_mtime:
        subprocess.run([sys.executable, "scripts/9_make_agent_windows_from_canonical.py", "--input",
                        str(CG / f"{args.name}.npz"), "--output", str(agents), "--stats",
                        str(CG / f"{args.name}_agents_stats.json")], cwd=REPO, check=True)
    stats = json.loads((CG / f"{args.name}.stats.json").read_text(encoding="utf-8"))
    tp = out / "throughput.json"
    result = {"label": "development pilot (throughput only)", "dataset": args.name, "epochs": args.epochs,
              "train_agents": stats["train"]["agents"], "val_agents": stats["val"]["agents"],
              "models": json.loads(tp.read_text(encoding="utf-8"))["models"] if tp.exists() else {}}
    for k, cmd in commands(args.name, args.epochs, out, args.models).items():
        print("+", " ".join(cmd), flush=True)
        result["models"][k] = run_timed(cmd, out / f"{k}.log")
        print(k, result["models"][k], flush=True)
        tp.write_text(json.dumps(result, indent=2), encoding="utf-8")       # a rerun replaces only its models
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
