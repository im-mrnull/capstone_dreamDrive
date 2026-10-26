# Frenét / MPC Planner

## Status: ❌ Not Started

## Goal

Trajectory sampler with cost = progress + comfort + collision risk against predicted futures.
Use conformal prediction radii as per-class calibrated constraints with coverage guarantees.

## What's Needed

1. **Frenét frame trajectory sampling** — generate candidate trajectories
2. **Cost function** — progress + comfort (jerk/acceleration) + collision risk
3. **Conformal integration** — consume per-class radii from `conformal_metrics.json`
4. **Evaluation** — compare planner performance with different base predictors

## Existing Foundations

- Conformal radii exist (`conformal_metrics.json`) — can be used as constraints
- All 6 deployable predictor outputs exist as prediction dumps
- Per-class radii computed (class-conditional conformal)

## Dependencies

- No planner code exists (zero grep hits for "planner", "frenet", "MPC", "cost function")
- Needs CARLA for closed-loop testing (also not started)
