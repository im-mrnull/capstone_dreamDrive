"""Lateral-intent statistics (scripts/counterfactual_sim.py): max-vs-max H1, interaction, Holm, family."""

import numpy as np

from conftest import load_script

cf = load_script("counterfactual_sim.py")
SCENES = [f"s{i:02d}" for i in range(40)]


def make_cells(shift: dict | None = None, noise: float = 0.02, seed: int = 0, af_shift: dict | None = None) -> dict:
    """Every lateral cell on 40 scenes x 50 episodes; the same scene means (common episodes),
    small independent cell noise, optional per-cell score / at-fault shifts."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.7, 0.9, len(SCENES))
    af = rng.uniform(0.0, 0.04, len(SCENES))
    cells = {}
    for c in cf.LAT_SINGLES + cf.LAT_SETS:
        s = base + (shift or {}).get(c, 0.0) + rng.normal(0, noise, len(SCENES))
        a = af + (af_shift or {}).get(c, 0.0) + rng.normal(0, noise / 4, len(SCENES))
        cells[c] = {"per_scene": {sid: [50, 50 * s[j], 50 * a[j]] for j, sid in enumerate(SCENES)}}
    return cells


ALL = {"proposal", "proposal+margin", "contingency", cf.NB}


def test_family_has_ten_tests_and_holm_is_monotone():
    r = cf.lateral_tests(make_cells(), ALL)
    assert r["family_size"] == 10
    assert [t["test"] for t in r["tests"]].count("H1") == 2 and [t["test"] for t in r["tests"]].count("H5") == 2
    ps = sorted((t["p"], t["p_holm"]) for t in r["tests"])
    assert all(ph >= p for p, ph in ps) and all(b[1] >= a[1] for a, b in zip(ps, ps[1:]))


def test_h1_detects_a_clear_set_advantage_and_names_the_winner():
    r = cf.lateral_tests(make_cells({"contingency | Ideal set": 0.05}), ALL)
    h1 = next(t for t in r["tests"] if t["test"] == "H1" and t["metric"] == "score")
    assert h1["x"] == "contingency | Ideal set" and h1["gain"] > 0.03 and h1["p_holm"] < 0.05
    assert r["h1_reading"] == "meaningful benefit"


def test_h1_at_fault_direction_lower_is_better():
    r = cf.lateral_tests(make_cells(af_shift={"proposal | Ideal set": -0.02}), ALL)
    h1 = next(t for t in r["tests"] if t["test"] == "H1" and t["metric"] == "at_fault")
    assert h1["x"] == "proposal | Ideal set" and h1["gain"] > 0.01 and h1["p"] < 0.05


def test_h1_no_effect_is_not_significant_and_singles_better_gives_large_p():
    r = cf.lateral_tests(make_cells({c: 0.05 for c in cf.LAT_SINGLES}), ALL)
    h1 = next(t for t in r["tests"] if t["test"] == "H1" and t["metric"] == "score")
    assert h1["gain"] < 0 and h1["p"] > 0.5 and r["h1_reading"] == "no demonstrated benefit"


def test_recentred_null_is_not_fooled_by_selection_among_many_sets():
    """Equal cells: the max over three noisy sets beats the max over four singles only by chance;
    across seeds the raw p is roughly uniform (rejection rate near alpha)."""
    rej = [next(t for t in cf.lateral_tests(make_cells(noise=0.03, seed=s), ALL, seed=s)["tests"]
                if t["test"] == "H1" and t["metric"] == "score")["p"] < 0.05 for s in range(200)]
    assert np.mean(rej) <= 0.10


def test_interaction_sign():
    sh = {"contingency | Ideal set": 0.04}                    # the set helps only in the contingency planner
    r = cf.lateral_tests(make_cells(sh), ALL)
    h5 = next(t for t in r["tests"] if t["test"] == "H5" and t["metric"] == "score")
    assert h5["gain"] > 0.03 and h5["p"] < 0.05


def test_ineligible_planner_cells_drop_out():
    r = cf.lateral_tests(make_cells({"contingency | Ideal set": 0.05}), {"proposal", "proposal+margin"})
    assert r["family_size"] == 4 and all("contingency" not in str(t.get("x")) for t in r["tests"])


def test_part_1b_reading():
    cells = make_cells(af_shift={"contingency | Ideal single (mean)": -0.02})
    assert cf.tests_1b(cells)["reading"] == "confirmed best single setup"
    cells = make_cells()
    assert cf.tests_1b(cells)["reading"] == "not confirmed"
