# CARLA Replay Harness

## Status: 🟡 Visualization working (2026-11-24); no CARLA-based evaluation

- CARLA 0.9.16 runs: the DirectX End-User Runtime is installed, and the server is ready in 3–12 s.
- **Open-loop replay:** `scripts/viz_prepare.py` → `viz_carla_replay.py` → `viz_compose.py`, driven by
  `viz_run_carla.py`.
- **Closed-loop demo packs:** `scripts/viz_closed_loop_pack.py`. The smarter planner drives, ST-GAT predicts:
  India from IDD val, the US from AV2 val.
- **Overlays:** drawn in image space by default (`--overlay image`). CARLA debug shapes wash out to white under
  auto-exposure.
- **Fallback without CARLA:** `viz_topdown_frames.py`.
- **Videos:** `outputs/viz/videos/cl_*`. They hold AV2 data, so keep them local.
- Details: [[chaos-gap]], section "Review 3 demo". The sections below predate this.

## Goal

Import IDD-3D world-frame trajectories as scripted CARLA actors. Replay real-world
scenarios in simulation for closed-loop evaluation.

## What's Needed

1. **Scenario exporter** — convert DreamDrive world-frame trajectories to CARLA-compatible format (OpenSCENARIO `.xosc` or direct Python API)
2. **CARLA Python API integration** — spawn actors, replay trajectories, capture metrics
3. **Collision detection** — measure collision rates, TTC, PET
4. **Map matching** — handle the fact that IDD-3D scenes don't have CARLA maps

## Current Evidence in Repo

"CARLA" appears only in:
- README known-limitations section ("planned work")
- `research/report.txt` and `research/presentation.txt`

No CARLA scripts, no `.xosc` files, no CARLA Python API code.

## Dependencies

- CARLA simulator installation
- CARLA Python API
- GPU for simulation rendering
- Potentially: custom map or map-free scenarios
