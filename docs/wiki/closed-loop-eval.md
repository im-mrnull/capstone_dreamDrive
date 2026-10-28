# Closed-Loop Evaluation

## Status: 🟡 Non-reactive Python closed loop done (2026-10-27, [[closed-loop-test]]); cross-validated confirmatory version done (2026-10-28, [[closed-loop-cv]]); reactive / CARLA version not started

## Goal

Run predictors inside a planner, measure collision rates, and test whether ADE ranking
predicts collision-rate ranking.

## Depends On

- Frenét/MPC planner (not started)
- CARLA replay harness (not started)

## What's Needed

- Collision rate computation
- TTC (time-to-collision) and PET (post-encroachment time) metrics
- Comparison: does lower ADE → lower collision rate?
