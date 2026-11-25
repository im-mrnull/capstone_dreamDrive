"""Fine-tuning options in MFP (the K-mode decoder), AgentFormer, LED and AutoBot, as in models 2, 3 and 5:
--train-repeat (the per-epoch history counts optimizer steps and agent-windows), keeping the final epoch
(--select last; MFP --keep last), the starting weights as a candidate (--select-include-init: with a zero lr
nothing later is strictly better, so epoch 0 is kept and saved), and a fixed lr schedule that halves at the
given epochs. LED applies each to both stages. Tiny data from test_init_ckpt (4 train sequences of 3 agents).

Skips cleanly when an upstream repo (third_party/agentformer, led, autobots) is not cloned."""

import json
import math
import sys

import pytest

from conftest import load_script
from test_init_ckpt import _data

torch = pytest.importorskip("torch")


def _available(name):
    try:
        if name == "agentformer":
            import dd_agentformer
            dd_agentformer.add_af_to_path()
        elif name == "led":
            import dd_led
            dd_led.add_led_to_path()
        elif name == "autobot":
            load_script("model9_autobot_train_eval.py").load_autobot_joint()
        return None
    except Exception as e:  # pragma: no cover - environment-dependent
        return repr(e)


# script, base args (2 epochs, 2 sequences per step), keep-last, milestone at epoch 1, zero lr
SPECS = {
    "mfp": ("model4_mfp_train_eval.py", ["--epochs", "2", "--batch-size", "2", "--k", "2"], ["--keep", "last"],
            ["--lr-milestones", "1"], ["--lr", "0"]),
    "agentformer": ("model6_agentformer_train_eval.py", ["--smoke", "--epochs", "2"], ["--select", "last"],
                    ["--lr-milestones", "1"], ["--lr", "0"]),
    "led": ("model7_led_train_eval.py", ["--core-epochs", "2", "--init-epochs", "2", "--k", "4", "--batch-seqs", "2",
                                         "--init-batch-seqs", "2"], ["--select", "last"],
            ["--core-lr-milestones", "1", "--init-lr-milestones", "1"], ["--core-lr", "0", "--init-lr", "0"]),
    "autobot": ("model9_autobot_train_eval.py", ["--epochs", "2", "--batch-seqs", "2", "--hidden-size", "16",
                                                 "--tx-hidden-size", "32", "--tx-num-heads", "2"], ["--select", "last"],
                ["--lr-sched", "1"], ["--lr", "0"]),
}
TRAIN_SEQS, AGENTS = 4, 3
PER_STEP = {"mfp": 2, "agentformer": 1, "led": 2, "autobot": 2}                # sequences per optimizer step
NAMES = [pytest.param(n, marks=pytest.mark.skipif(_available(n) is not None, reason=f"{n} unavailable"))
         for n in SPECS]


def _histories(name, metrics):
    return [metrics["history"], metrics["core_history"]] if name == "led" else [metrics["history"]]


def _run(name, tmp_path, monkeypatch, tag, extra, spec=None):
    script, base, *_ = spec or SPECS[name]
    mod = load_script(script)
    monkeypatch.setattr(sys, "argv", [script, "--device", "cpu", "--seed", "0", "--data", str(tmp_path / "d.npz"),
                                      "--splits-file", str(tmp_path / "s.json"), *base,
                                      "--save-ckpt", str(tmp_path / f"{tag}.pt"),
                                      "--save-metrics-json", str(tmp_path / f"{tag}.json"), *extra])
    assert mod.main() == 0
    return json.loads((tmp_path / f"{tag}.json").read_text()), torch.load(tmp_path / f"{tag}.pt", map_location="cpu",
                                                                         weights_only=False)


@pytest.mark.parametrize("name", NAMES)
def test_train_repeat_scales_the_steps_and_agent_windows_per_epoch(name, tmp_path, monkeypatch):
    _data(tmp_path)
    metrics, _ = _run(name, tmp_path, monkeypatch, "rep", ["--train-repeat", "2"])
    spe = math.ceil(TRAIN_SEQS * 2 / PER_STEP[name])
    for h in _histories(name, metrics):
        assert [(e["epoch"], e["steps"], e["agent_windows"]) for e in h] == \
            [(i, i * spe, i * TRAIN_SEQS * 2 * AGENTS) for i in (1, 2)], name
    assert metrics["hyper"]["train_repeat"] == 2


@pytest.mark.parametrize("name", NAMES)
def test_keeping_the_last_epoch_ignores_val(name, tmp_path, monkeypatch):
    _data(tmp_path)
    metrics, ck = _run(name, tmp_path, monkeypatch, "last", SPECS[name][2])
    assert metrics["best_epoch"] == 2 and ck["epoch"] == 2
    if name == "led":
        assert metrics["core_best_epoch"] == 2 and ck["core_epoch"] == 2


@pytest.mark.parametrize("name", NAMES)
def test_include_init_keeps_the_starting_weights_when_nothing_beats_them(name, tmp_path, monkeypatch):
    _data(tmp_path)
    _, pre = _run(name, tmp_path, monkeypatch, "pre", [])
    metrics, ft = _run(name, tmp_path, monkeypatch, "ft", ["--init-ckpt", str(tmp_path / "pre.pt"),
                                                         "--select-include-init", *SPECS[name][4]])
    assert metrics["best_epoch"] == 0 and ft["epoch"] == 0
    for h in _histories(name, metrics):
        assert h[0]["epoch"] == 0 and h[0]["lr"] is None and h[0]["steps"] == 0 and len(h) == 3
    for key in (("core", "init") if name == "led" else ("model",)):
        assert all(torch.equal(ft[key][k], v) for k, v in pre[key].items()), (name, key)
    if name == "led":
        assert metrics["core_best_epoch"] == 0 and ft["core_epoch"] == 0


@pytest.mark.parametrize("name", NAMES)
def test_fixed_milestones_halve_the_lr_after_that_epoch(name, tmp_path, monkeypatch):
    _data(tmp_path)
    metrics, _ = _run(name, tmp_path, monkeypatch, "ms", SPECS[name][3])
    for h in _histories(name, metrics):
        assert h[1]["lr"] == pytest.approx(h[0]["lr"] / 2), name


EMPTY = {"stgat": ("model3_stgat_train_eval.py", ["--epochs", "2"], ["--lr-milestones"]),
         "transformer": ("model5_transformer_train_eval.py", ["--epochs", "2"], ["--lr-milestones"]),
         "mfp": (*SPECS["mfp"][:2], ["--lr-milestones"]),
         "agentformer": (*SPECS["agentformer"][:2], ["--lr-milestones"]),
         "led": (*SPECS["led"][:2], ["--core-lr-milestones", "--init-lr-milestones"])}


@pytest.mark.parametrize("name", ["stgat", "transformer", *[p for p in NAMES if p.values[0] in EMPTY]])
def test_milestones_given_with_no_epochs_mean_a_constant_lr_never_a_val_schedule(name, tmp_path, monkeypatch):
    """The budget rule can drop every halving. The run must then train at a constant lr: falling back to the
    val plateau (or step) schedule would let val data choose the lr."""
    def refuse(*a, **k):
        raise AssertionError("a plateau or step schedule was built")
    for sched in ("ReduceLROnPlateau", "StepLR"):
        monkeypatch.setattr(torch.optim.lr_scheduler, sched, refuse)
    _data(tmp_path)
    metrics, _ = _run(name, tmp_path, monkeypatch, "empty", EMPTY[name][2], spec=EMPTY[name])
    for h in _histories(name, metrics):
        assert len({e["lr"] for e in h}) == 1, name


def test_autobot_with_no_milestones_keeps_a_constant_lr(tmp_path, monkeypatch):
    if _available("autobot") is not None:
        pytest.skip("autobot unavailable")
    _data(tmp_path)
    metrics, _ = _run("autobot", tmp_path, monkeypatch, "const", ["--lr-sched"])
    assert metrics["history"][0]["lr"] == metrics["history"][1]["lr"] == metrics["hyper"]["lr"]
