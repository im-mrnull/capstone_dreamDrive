"""Reactive agents (scripts/reactive_sim.py) on synthetic scenes."""

import numpy as np

from conftest import load_script

rs = load_script("reactive_sim.py")
cl = rs.cl

T = 100


def scene(track: np.ndarray, cls: str):
    """A stationary ego at the origin facing +x and one agent following `track` (T,2)."""
    route = cl.Route(np.array([[0.0, 0.0], [600.0, 0.0]]))
    pos = track[:, None, :].astype(float)
    mask = np.ones((T, 1), bool)
    sc = {"sid": "synthetic", "pos": pos, "mask": mask, "cls": np.array([cls]), "agent_id": np.array([7]),
          "heading": cl.agent_headings(pos, mask, route), "route": route, "s_h": np.zeros(T),
          "v_h": np.zeros(T), "T": T}
    log = {"sc": sc, "pos": pos, "mask": mask, "heading": sc["heading"], "vlog": rs.logged_speed(pos, mask)}
    return rs.episode_scene(log), log


def drive(track, cls, react, ego_x=lambda g: 0.0, t_start=19, n=60):
    sc, log = scene(track, cls)
    w = rs.World(sc, log, t_start, np.array([react]))
    for g in range(t_start + 1, t_start + 1 + n):
        x = ego_x(g)          # before the episode the ego is the (stationary) human at the origin
        w.step(g, np.array([x + cl.EGO_CENTRE, 0.0]), np.array([1.0, 0.0]), 0.0)
    return sc, w


def crossing():
    """A pedestrian walking across the ego's nose at 1.2 m/s (x = 1 m, y from -8 m)."""
    t = np.arange(T) * cl.DT
    return np.stack([np.full(T, 1.0), -8.0 + 1.2 * t], 1)


def test_non_reactive_agent_replays_its_log():
    sc, w = drive(crossing(), "Pedestrian", react=False)
    assert np.array_equal(sc["pos"], scene(crossing(), "Pedestrian")[0]["pos"])
    assert not np.isfinite(w.first_dev).any()


def test_reactive_pedestrian_stops_before_a_blocking_ego():
    sc, w = drive(crossing(), "Pedestrian", react=True)
    y = sc["pos"][20:80, 0, 1]
    assert y.max() < -1.0                       # never walks into the ego's footprint
    assert w.tau[0] < 79 - 10                   # held back by more than a second


def test_vehicle_behind_the_ego_does_not_rear_end_it():
    t = np.arange(T) * cl.DT
    track = np.stack([-40.0 + 6.0 * t, np.zeros(T)], 1)          # would drive through the ego
    sc, _ = drive(track, "Car", react=True)
    x = sc["pos"][20:80, 0, 0]
    ego_rear = cl.EGO_CENTRE - cl.EGO_RECT[0] / 2
    assert (x + cl.rect_size("Car")[0] / 2 < ego_rear).all()


def test_agent_resumes_its_path_with_a_lag_once_the_ego_leaves():
    sc, w = drive(crossing(), "Pedestrian", react=True, ego_x=lambda g: 0.0 if g < 60 else 50.0, n=78)
    y = sc["pos"][:, 0, 1]
    assert y[60] < -1.5 and y[97] > 0.5          # waited for the ego, then crossed its lane
    lag = 97 - w.tau[0]
    assert 10 < lag < 40                         # 1-4 s behind its log, but moving again
    assert np.allclose(sc["pos"][97, 0], w.position(0, w.tau[0])[0])
    assert sc["pos"][97, 0, 1] - sc["pos"][96, 0, 1] > 0.1   # back at walking pace


def test_reactive_draw_is_nested_and_paired():
    a = rs.reactive_draw("s", 19, "nominal", 0, 500, 0.5)
    b = rs.reactive_draw("s", 19, "nominal", 0, 500, 1.0)
    assert b.all() and 0.4 < a.mean() < 0.6
    assert np.array_equal(a, rs.reactive_draw("s", 19, "nominal", 0, 500, 0.5))
    assert not np.array_equal(a, rs.reactive_draw("s", 19, "nominal", 1, 500, 0.5))
