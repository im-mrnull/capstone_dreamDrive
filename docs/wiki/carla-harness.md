# CARLA Replay Harness

## Status: ❌ Not Started

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
