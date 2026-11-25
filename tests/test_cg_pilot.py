"""Chaos-gap pilot runner (scripts/cg_pilot.py): every GPU model is launched under the CUDA cache
cap, every script it launches defines the flag, and a failed model run fails the pilot (an
uncapped MFP pilot once spilled 58 GB into shared memory and was still logged as exit 0)."""

import json
import os
import sys

from conftest import load_script

pilot = load_script("cg_pilot.py")
MODELS = ["cv", "mlp", "stgat", "transformer", "led", "autobot", "agentformer", "mfp"]


def test_every_gpu_model_runs_under_the_memory_cap(tmp_path):
    cmds = pilot.commands("av2_matched_dev", 3, tmp_path, MODELS, 0.8)
    gpu = [k for k, cmd in cmds.items() if "--device" in cmd]
    assert gpu == ["stgat", "transformer", "led", "autobot", "agentformer", "mfp"]   # CV and the MLP are numpy
    for k in gpu:
        cmd = cmds[k]
        assert cmd[cmd.index("--gpu-mem-frac") + 1] == "0.8", k
        assert '"--gpu-mem-frac"' in (pilot.REPO / cmd[1]).read_text(encoding="utf-8"), k


def test_a_failed_model_run_fails_the_pilot(tmp_path, monkeypatch):
    (tmp_path / "d.npz").write_bytes(b"")
    (tmp_path / "d_agents.npz").write_bytes(b"")
    os.utime(tmp_path / "d_agents.npz", (2e9, 2e9))                     # newer than d.npz: not rebuilt
    (tmp_path / "d.stats.json").write_text(json.dumps({"train": {"agents": 1}, "val": {"agents": 1}}))
    monkeypatch.setattr(pilot, "CG", tmp_path)
    monkeypatch.setattr(pilot, "commands", lambda *a: {"ok": [sys.executable, "-c", "pass"],
                                                       "bad": [sys.executable, "-c", "raise SystemExit(3)"]})
    monkeypatch.setattr(sys, "argv", ["cg_pilot.py", "--name", "d"])
    assert pilot.main() == 1
    tp = json.loads((tmp_path / "pilot" / "d" / "throughput.json").read_text(encoding="utf-8"))
    assert tp["models"]["ok"]["returncode"] == 0 and tp["models"]["bad"]["returncode"] == 3
