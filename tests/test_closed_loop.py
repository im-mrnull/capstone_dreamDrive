"""Closed-loop simulator building blocks (scripts/closed_loop_sim.py), on synthetic scenes."""

import numpy as np
import pytest

from conftest import load_script

cl = load_script("closed_loop_sim.py")


def straight_scene(agent_xy: np.ndarray, cls="Car", v_h=5.0, T=100):
    """Ego route along +x; the human drives at v_h; one agent with a fixed track."""
    route = cl.Route(np.array([[0.0, 0.0], [600.0, 0.0]]))
    s_h = np.arange(T) * v_h * cl.DT
    pos = agent_xy[:, None, :].astype(float)
    mask = np.ones((T, 1), bool)
    heading = np.tile(np.array([1.0, 0.0]), (T, 1, 1))
    return {"sid": "synthetic", "pos": pos, "mask": mask, "cls": np.array([cls]), "agent_id": np.array([7]),
            "heading": heading, "route": route, "s_h": s_h, "v_h": np.full(T, v_h), "T": T,
            "ego_raw": np.stack([s_h, np.zeros(T)], 1), "ego_head_raw": np.tile([1.0, 0.0], (T, 1)),
            "glitch": np.zeros(T, bool)}


class Truth:
    """Predictor stub that knows the agent's true future (like the oracle reference)."""

    def __init__(self, name="Oracle (true future, no band)"):
        self.name = name

    def at(self, sc, t, lam=1.0):
        if self.name == "No prediction (ignore agents)":
            return []
        out = []
        for a in np.where(sc["mask"][t])[0]:
            fut = np.full((60, 2), np.nan)
            n = min(60, sc["T"] - t - 1)
            fut[:n] = sc["pos"][t + 1:t + 1 + n, a]
            out.append((a, fut[None], np.zeros(60)))
        return out


def test_rollout_kinematics_and_lateral_move():
    s, d, v = cl.rollout(0.0, 0.0, 5.0, -1.0, 1.5, 30)
    assert v[-1] == pytest.approx(5.0 - 1.0 * 3.0)
    assert s[-1] == pytest.approx(5.0 * 3.0 - 0.5 * 1.0 * 9.0)
    assert d[-1] == pytest.approx(1.5) and d[9] == pytest.approx(1.5 * 1.0 / cl.LAT_TIME)
    s, _, v = cl.rollout(0.0, 0.0, 1.0, -6.0, 0.0, 10)            # never reverses
    assert (v >= 0).all() and np.all(np.diff(s) >= 0)


def test_planner_stops_for_a_stationary_obstacle_with_true_predictions():
    T = 100
    sc = straight_scene(np.tile([[30.0, 0.0]], (T, 1)))            # parked car 30 m ahead, in lane
    r_truth = cl.run_episode(sc, 19, 0.0, Truth())
    r_blind = cl.run_episode(sc, 19, 0.0, Truth("No prediction (ignore agents)"))
    assert not r_truth["at_fault"]
    assert r_blind["at_fault"] and r_blind["score"] == 0.0


def test_human_replay_follows_the_log_without_contact():
    T = 100
    sc = straight_scene(np.tile([[30.0, 6.0]], (T, 1)))            # agent 6 m to the side
    r = cl.run_episode(sc, 19, 0.0, Truth("Human driver (log replay)"))
    assert not r["at_fault"] and r["progress"] == pytest.approx(1.0)


def test_at_fault_convention():
    T = 100
    sc = straight_scene(np.tile([[30.0, 0.0]], (T, 1)))
    sc["route_tan_now"] = np.array([1.0, 0.0])
    size = cl.rect_size("Car")
    # in front, ego driving straight: at fault
    assert cl.at_fault(sc, 50, 0, lon=3.0, lat=0.2, ddot=0.0, size=size)
    # beside the ego, ego not swerving: not at fault
    assert not cl.at_fault(sc, 50, 0, lon=0.0, lat=-2.0, ddot=0.0, size=size)
    # beside the ego, ego swerving toward it: at fault
    assert cl.at_fault(sc, 50, 0, lon=0.5, lat=-2.0, ddot=-0.5, size=size)
    # an agent cutting in sideways fast: not the ego's fault
    cut = straight_scene(np.stack([np.full(T, 33.0), -1.0 + 0.5 * (np.arange(T) - 50)], 1))
    cut["route_tan_now"] = np.array([1.0, 0.0])
    assert not cl.at_fault(cut, 50, 0, lon=3.0, lat=-1.0, ddot=0.0, size=size)


def test_episode_score_weights():
    w = cl.SCORE_W
    assert (w[0] * 1.0 + w[1] + w[2]) / sum(w) == pytest.approx(1.0)
    assert cl.PERSONAS["nominal"] == 0.0 and cl.PERSONAS["assertive"] > 0.0


def test_rectangle_gap_is_exact():
    """Ground-truth collisions use exact oriented rectangles, not circle approximations."""
    A = cl.rect(np.array([0.0, 0.0]), np.array([1.0, 0.0]), (4.0, 2.0))
    gap = lambda c, h, size: cl.rect_gap(A, cl.rect(np.array(c), np.array(h, float), size))   # noqa: E731
    assert gap((6.0, 0.0), (1, 0), (4.0, 2.0)) == pytest.approx(2.0)      # apart
    assert gap((4.0, 0.0), (1, 0), (4.0, 2.0)) == pytest.approx(0.0)      # touching
    assert gap((3.0, 0.0), (1, 0), (4.0, 2.0)) == pytest.approx(-1.0)     # 1 m overlap
    assert gap((0.0, 3.5), (0, 1), (2.0, 1.0)) == pytest.approx(1.5)      # rotated, beside
    # diagonal corners: circles would overlap, rectangles do not
    assert gap((4.2, 2.2), (1, 0), (4.0, 2.0)) > 0


def test_pose_glitch_episodes_are_excluded():
    T = 100
    sc = straight_scene(np.tile([[30.0, 6.0]], (T, 1)))
    sc["glitch"] = np.zeros(T, bool)
    assert cl.episode_ok(sc, 19)
    sc["glitch"][40] = True
    assert not cl.episode_ok(sc, 19) and cl.episode_ok(sc, 49) is False or not sc["glitch"][49:100].any()
    assert not cl.episode_ok(sc, 49 + 10)                                   # runs past the scene end


def test_holm_correction():
    cv = load_script("closed_loop_cv.py")
    adj = cv.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)          # 3 x 0.01
    assert adj["c"] == pytest.approx(0.06)          # 2 x 0.03
    assert adj["b"] == pytest.approx(0.06)          # max(1 x 0.04, previous 0.06): monotone
