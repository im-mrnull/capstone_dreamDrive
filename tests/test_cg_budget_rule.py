"""The frozen fine-tuning budget rule (scripts/cg_budget_rule.py): S* is the median best-epoch step,
lr halvings match by ordinal occurrence in >= 2 of 3 draws and are dropped at or beyond S*, conversion
uses each run's own steps per epoch, and the steps formula reproduces every proxy run's history. For
the models added later: per-model steps per epoch (AutoBot's equals its own batching), LED's two stages,
flags that always name the schedule, --verify, and a frozen rule is never overwritten."""

import json
import sys

import numpy as np
import pytest

from conftest import REPO, load_script

br = load_script("cg_budget_rule.py")
PROXIES = sorted((REPO / "outputs" / "chaos_gap" / "dev" / "budget").glob("*_k*_r*.json"))


def _run(best_epoch, lrs, spe=10, include_init=False):
    """Metrics with a history: epoch e trained at lrs[e - 1], spe steps per epoch."""
    h = [{"epoch": 0, "lr": None, "val_ade": 9.0, "steps": 0, "agent_windows": 0}] if include_init else []
    h += [{"epoch": e, "lr": lr, "val_ade": 1.0, "steps": e * spe, "agent_windows": e * spe * 64}
          for e, lr in enumerate(lrs, 1)]
    return {"best_epoch": best_epoch, "history": h}


def test_halvings_take_effect_at_the_end_of_the_last_epoch_at_the_higher_lr():
    m = _run(5, [1e-3] * 3 + [5e-4] * 4 + [2.5e-4] * 2, include_init=True)
    assert br.halvings(m["history"]) == [{"after_epoch": 3, "step": 30, "factor": 0.5},
                                         {"after_epoch": 7, "step": 70, "factor": 0.5}]
    assert br.best_step(m["history"], m["best_epoch"]) == 50


def test_rule_takes_the_median_and_matches_halvings_by_ordinal():
    runs = [_run(4, [1e-3] * 2 + [5e-4] * 8),                          # best 40; halvings at 20
            _run(6, [1e-3] * 3 + [5e-4] * 2 + [2.5e-4] * 5),            # best 60; 30, 50
            _run(9, [1e-3] * 4 + [5e-4] * 6)]                           # best 90; 40
    r = br.rule(runs, constant_lr=False)
    assert r["s_star"] == 60
    assert [(m["ordinal"], m["draws"], m["step"]) for m in r["milestones"]] == [(1, 3, 30)]   # 2nd: one draw only
    assert br.rule(runs, constant_lr=True)["milestones"] == []


def test_milestones_at_or_beyond_the_budget_are_dropped():
    runs = [_run(3, [1e-3] * 3 + [5e-4] * 7)] * 3                       # best 30; halving at 30
    assert br.rule(runs, constant_lr=False) == {"s_star": 30, "milestones": []}


def test_convert_uses_the_run_steps_per_epoch_and_rounds_half_up():
    r = {"s_star": 145, "milestones": [{"step": 25}, {"step": 140}]}
    assert br.convert(r, 10) == {"epochs": 15, "lr_milestones": [3, 14], "dropped_milestones": [], "steps": 150}
    assert br.convert(r, 20) == {"epochs": 7, "lr_milestones": [1], "dropped_milestones": [7], "steps": 140}
    assert br.convert({"s_star": 3, "milestones": []}, 100)["epochs"] == 1
    assert br.convert({"s_star": 100, "milestones": [{"step": 95}]}, 10) == \
        {"epochs": 10, "lr_milestones": [], "dropped_milestones": [10], "steps": 100}


def test_steps_per_epoch_counts_the_partial_last_batch():
    assert br.steps_per_epoch(36, 17, 64) == 10 and br.steps_per_epoch(36, 17, 128) == 5
    assert br.steps_per_epoch(442, 17, 256) == 30


@pytest.mark.skipif(not PROXIES, reason="no proxy runs yet")
def test_the_steps_formula_reproduces_every_finished_proxy_run():
    sizes_of = br.TrainSizes()
    for f in PROXIES:
        d, model = json.loads(f.read_text(encoding="utf-8")), f.stem.split("_")[0]
        sizes = sizes_of(model, REPO / (d["input"] if model == "mlp" else d["data"]), REPO / d["splits_file"])
        for name, hist_key, _ in br.stages(model):
            want = br.model_steps_per_epoch(model, name, sizes, d["hyper"]["train_repeat"], d["hyper"])
            h = [e for e in d[hist_key] if e["epoch"] >= 1]
            assert {b["steps"] - a["steps"] for a, b in zip([{"steps": 0}] + h, h)} == {want}, (f.name, name)


def test_autobot_steps_equal_its_own_one_size_batching():
    m9 = load_script("model9_autobot_train_eval.py")
    sizes = np.array([3, 3, 3, 200, 200, 7, 7, 7, 7, 300])
    hyper = {"batch_seqs": 2, "max_slots": 512}
    for repeat in (1, 2, 5):
        order = list(range(len(sizes))) * repeat
        got = len(list(m9.batches(sizes, order, hyper["batch_seqs"], hyper["max_slots"])))
        assert br.model_steps_per_epoch("autobot", "main", sizes, repeat, hyper) == got
    # size 3 fits 64 per batch, size 200 only 2 (512 // 200), size 300 one
    assert br.model_steps_per_epoch("autobot", "main", np.array([3, 3, 3, 200, 200, 300]), 2,
                                    {"batch_seqs": 64, "max_slots": 512}) == 1 + 2 + 2


def test_other_models_steps_per_epoch():
    sizes = np.full(10, 4)
    assert br.model_steps_per_epoch("agentformer", "main", sizes, 3, {}) == 30          # one sequence per step
    led = {"batch_seqs": 8, "init_batch_seqs": 4}
    assert br.model_steps_per_epoch("led", "core", sizes, 3, led) == 4                  # ceil(30 / 8)
    assert br.model_steps_per_epoch("led", "init", sizes, 3, led) == 8                  # ceil(30 / 4)
    assert br.model_steps_per_epoch("mfp", "main", sizes, 3, {"batch_size": 64}) == 1


def test_flags_always_name_the_schedule_even_when_no_halving_is_kept():
    c = {"epochs": 12, "lr_milestones": []}
    assert br.flags("mfp", {"main": c}, 3) == "--epochs 12 --lr-milestones --train-repeat 3 --keep last"
    assert br.flags("autobot", {"main": {"epochs": 40, "lr_milestones": [10, 20]}}, 1) ==         "--epochs 40 --lr-sched 10 20 --train-repeat 1 --select last"
    assert br.flags("led", {"core": {"epochs": 30, "lr_milestones": [8]}, "init": c}, 6) ==         "--core-epochs 30 --core-lr-milestones 8 --init-epochs 12 --init-lr-milestones --train-repeat 6 --select last"


def test_led_rule_is_per_stage():
    def led(core_best, init_best):
        core, init = _run(core_best, [1e-3] * 4 + [5e-4] * 6), _run(init_best, [1e-3] * 2 + [5e-4] * 8, spe=20)
        return {"core_history": core["history"], "core_best_epoch": core_best, "history": init["history"],
                "best_epoch": init_best}
    runs = [led(9, 3), led(8, 4), led(10, 5)]
    assert br.rule(runs, False, "core_history", "core_best_epoch") ==         {"s_star": 90, "milestones": [{"ordinal": 1, "draws": 3, "step": 40, "factors": [0.5] * 3}]}
    assert br.rule(runs, False) ==         {"s_star": 80, "milestones": [{"ordinal": 1, "draws": 3, "step": 40, "factors": [0.5] * 3}]}


def test_verify_catches_a_val_schedule_fallback_and_passes_a_run_on_the_rule(tmp_path):
    rule = {"models": {"stgat": {"k10": {"idd_draws": {"r0": {"train_repeat": 6, "epochs": 4, "lr_milestones": [],
                                                                "steps": 40}}}}}}
    on = _run(4, [1e-3] * 4)
    on.update(hyper={"train_repeat": 6, "select": "last", "lr_milestones": []})
    off = _run(4, [1e-3] * 3 + [5e-4])                                  # the plateau halved it after epoch 3
    off.update(hyper={"train_repeat": 6, "select": "last", "lr_milestones": None})
    (tmp_path / "stgat_k10_r0_pre.json").write_text(json.dumps(on))
    assert br.verify(tmp_path, {"stgat": rule}) == []
    (tmp_path / "stgat_k10_r0_scr.json").write_text(json.dumps(off))
    bad = br.verify(tmp_path, {"stgat": rule})
    assert len(bad) == 2 and all(b.startswith("stgat_k10_r0_scr") for b in bad)


def test_a_frozen_rule_is_never_overwritten(tmp_path, monkeypatch):
    out = tmp_path / "rule.json"
    out.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["cg_budget_rule.py", "--out", str(out)])
    assert br.main() == 1 and out.read_text() == "{}"
