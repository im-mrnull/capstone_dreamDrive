# Frenét / MPC Planner

## Status: 🟡 First version (2026-10-27): sampling planner in `scripts/closed_loop_sim.py`, see [[closed-loop-test]]. Planner v2 (2026-10-28): 1 s hard constraint, soft penalty beyond it, band ×0, tuned on the 32 dev scenes, see [[closed-loop-cv]]. Planner v3 dev (2026-10-29): gate failed, see below

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

## Planner v3 development (2026-10-29, dev scenes only)

[`scripts/closed_loop_v3.py`](../../scripts/closed_loop_v3.py), 32 dev scenes, frozen-split
models at seed 0. The 73 primary scenes were not used.

**v3 options** (`closed_loop_sim.V3_DEFAULTS`; the defaults reproduce v2 episode for
episode):
- finer accelerations;
- a clearance cost Σ max(0, 1 m − predicted gap)²;
- a jerk cost (a − a_prev)²;
- a lateral-change cost.

**Diagnosis** (`closed_loop_v3_diagnosis_dev.json`). Each failure is attributed by a
counterfactual with the oracle on the same episode:
- **Prediction has no headroom on dev:** the oracle scores 0.901 and CV 0.901.
- **Near-misses (≈ 19%, human 6%) are the planner's.** The oracle has them too, nearly all
  while moving.
- **Single-trajectory harsh braking is split:** about half comes from prediction and about
  half from having no clear manoeuvre.
- **Multi-future sets under planner v2:** no clear manoeuvre in 46% (K-mode) and 16% (LED)
  of replans, against 1.5–3% for single trajectories.

**Grid** (`closed_loop_v3_dev_grid.json`, 18 settings):

| Setting | Mean score (8 single models) | Harsh braking | Near-miss | Oracle |
|---|---|---|---|---|
| v2 | 0.894 | 9% | 19% | 0.901 |
| Fine accelerations (pre-fixed pick) | 0.906 | 7% | 18% | 0.914 |
| + clearance 2, jerk 0.2 (best) | 0.910 | 5% | 17% | 0.931 |

- **Pre-registered gate: failed.** CV gained +0.006, short of the required +0.01. The oracle
  gained +0.013, and at-fault did not rise.
- Gains are small on dev, and per-package at-fault moves by a few episodes out of 224 (noise).
- **The clearance cost helps the oracle most** (0.901 → 0.931). Better planning makes
  accurate prediction worth more: the oracle's lead over the single models grows from about
  0.007 to about 0.02.
- **Latency:** fine accelerations take about 3 ms per replan with a single trajectory and
  about 14 ms with a 10-mode set (v2: 2 and 7 ms).

## Dependencies

- Reactive agents are needed to test interaction (not started).
- CARLA is used only for visualization.
