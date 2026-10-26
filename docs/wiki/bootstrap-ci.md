# Bootstrap Confidence Intervals

## Status: ❌ Not Started

## Goal

Scene-level resampling CIs on headline numbers. Bootstrap over scenes (not agents) to
get honest confidence intervals that account for within-scene correlation.

## What's Needed

- Bootstrap resampling at the scene level
- CI computation for ADE, FDE, per-class metrics
- Integration with `make_tables.py` for reporting

## Complexity: LOW — straightforward implementation
