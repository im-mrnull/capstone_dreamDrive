"""Gap recovery in scripts/cg_ft_diagnose.py: paired scene-bootstrap statistics, the explicit recovery
definition (computed only where G0 > 0, never capped), the "no demonstrable gap" flag, and 50% / 90%
points only on a demonstrable gap and a curve monotone within its CIs."""

import numpy as np

from conftest import load_script

fd = load_script("cg_ft_diagnose.py")
KS = ("05", "10", "20", "40", "72")
SCENES = np.repeat([f"s{i}" for i in range(5)], 4)                      # 5 scenes x 4 agents


def _const(v):
    return np.full(len(SCENES), float(v))


def _curve(z, pre, scr=None, cv=1.0):
    """per_k from {k: ADE} (scalars, or per-agent arrays) for each arm; the scratch arm defaults to pre."""
    scr = scr or pre
    as_arr = lambda v: v if isinstance(v, np.ndarray) else _const(v)   # noqa: E731
    per_k = {(k, arm): as_arr(src[k]) for k in KS for arm, src in (("pre", pre), ("scr", scr))}
    return fd.recovery_curve(fd.half_stat(SCENES, n_boot=2000), _const(cv), as_arr(z), per_k, KS)


def test_half_stat_shares_one_count_matrix_so_differences_are_paired():
    stat = fd.half_stat(SCENES, n_boot=500)
    rng = np.random.default_rng(0)
    v, w = rng.normal(size=len(SCENES)), rng.normal(size=len(SCENES))
    (ev, rv), (ew, rw), (ed, rd) = stat(v), stat(w), stat(v - w)
    assert np.isclose(ev - ew, ed) and np.allclose(rv - rw, rd)


def test_recovery_is_gk_over_g0_and_the_points_are_the_first_k_reaching_them():
    pre = {"05": 2.8, "10": 2.5, "20": 2.2, "40": 2.1, "72": 2.0}
    d = _curve(3.0, pre, scr={**pre, "72": 2.0})
    assert d["gap"] == "demonstrable" and np.isclose(d["G0"]["estimate"], 1.0)
    assert [round(d[f"k{k}_pre"]["recovery_pct"]["estimate"]) for k in KS] == [20, 50, 80, 90, 100]
    assert d["monotone_pre"]["monotone"] and d["points_pre"] == {"50": "k=10", "90": "k=40"}


def test_recovery_is_never_capped():
    d = _curve(3.0, {"05": 3.5, "10": 2.5, "20": 2.2, "40": 2.1, "72": 1.5}, scr={k: 2.0 for k in KS})
    assert np.isclose(d["k05_pre"]["recovery_pct"]["estimate"], -50)    # worse than zero-shot
    assert np.isclose(d["k72_pre"]["recovery_pct"]["estimate"], 150)    # better than the reference


def test_a_step_wholly_on_the_worse_side_makes_the_points_not_identifiable():
    d = _curve(3.0, {"05": 2.8, "10": 2.5, "20": 2.9, "40": 2.1, "72": 2.0})
    assert d["monotone_pre"] == {"monotone": False, "worse_steps": ["k10->k20"]}
    assert d["points_pre"] == {"50": fd.NOT_IDENTIFIABLE, "90": fd.NOT_IDENTIFIABLE}


def test_best_of_6_takes_a_samplers_first_six_and_a_scored_models_six_most_probable(tmp_path):
    gt, fm = np.zeros((1, 4, 2)), np.ones((1, 4), bool)
    sets = np.stack([np.full((4, 2), float(d)) for d in (5, 4, 3, 2, 1, 6, 7, 0.5)])[None]   # ADE sqrt(2) * d
    np.savez(tmp_path / "s.npz", samples=sets)
    assert np.isclose(fd.best_of_6(np.load(tmp_path / "s.npz"), gt, fm)[0], np.sqrt(2) * 1)     # first six: 1
    logits = np.array([[0, 0, 0, 0, 0, 0, 0, 9.0]])                                         # 0.5 is the most probable
    np.savez(tmp_path / "m.npz", modes=sets, mode_logits=logits)
    assert np.isclose(fd.best_of_6(np.load(tmp_path / "m.npz"), gt, fm)[0], np.sqrt(2) * 0.5)
    np.savez(tmp_path / "p.npz", pred=sets[:, 0])
    assert fd.best_of_6(np.load(tmp_path / "p.npz"), gt, fm) is None


def test_no_recovery_without_a_gap_and_a_flag_when_its_ci_includes_zero():
    d = _curve(2.0, {k: 2.5 for k in KS})                               # the reference is worse than zero-shot
    assert d["gap"].startswith("no gap") and "recovery_pct" not in d["k05_pre"]
    z = _const(3.0) + np.repeat([1.0, -1.2, 0.9, -0.8, 0.3], 4)        # G0 > 0 on average, not in every scene
    d = _curve(z, {k: 3.0 for k in KS})
    assert d["G0"]["estimate"] > 0 and d["G0"]["ci95"][0] <= 0 and d["gap"] == "no demonstrable gap"
    assert "recovery_pct" in d["k05_pre"] and d["points_pre"]["50"] == fd.NOT_IDENTIFIABLE
