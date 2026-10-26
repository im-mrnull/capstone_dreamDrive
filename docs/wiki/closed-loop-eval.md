# Closed-Loop Evaluation

## Status: ❌ Not Started

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
