# Frenét / MPC Planner

## Status: 🟡 First version (2026-10-27): sampling planner in `scripts/closed_loop_sim.py`, see [[closed-loop-test]]. Planner v2 (2026-10-28): 1 s hard constraint, soft penalty beyond it, band ×0, tuned on the 32 dev scenes, see [[closed-loop-cv]]

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
