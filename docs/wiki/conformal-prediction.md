# Conformal Prediction Wrapper

## Status: ✅ Done — revised Oct 2026 (scene-level folds, trajectory coverage, simultaneous bands)

## What It Does

Post-hoc wrapper (`scripts/conformal_wrapper.py`) that turns each deployable point predictor
into uncertainty bands. No retraining. Nonconformity = per-step position error.

- **Per-step band** r_t: each step covered w.p. ≥ 1−α (not the whole future).
- **Simultaneous band** r_t = q·σ_t: σ_t fitted on half of the calibration scenes, q = conformal
  quantile of max_t(err_t/σ_t) on the other half → whole-future guarantee (what a planner needs).
- Each as one global radius or per-class radii (= **Mondrian** conformal; standard method).
- **Regime B**: calibrate/evaluate on halves of the test **scenes**, 100 random splits.
- **Regime A**: calibrate on val, evaluate on test, scene-bootstrap CI.

## ⚠️ Oct 2026 corrections (important for the paper)

The old regime B split test **agents** 50/50 → correlated agents (and Trajectron++'s
overlapping per-timestep windows of the same agent) leaked across calibration/evaluation →
coverage looked nominal (89–91%). With scene-level splits:

1. Regime B averages a little **under** target and **varies hugely** between scene splits
   (5–95% range roughly 70–98%). Scene clustering is the binding issue on 22 scenes.
2. Regime A's val→test drop sits at the 8th–22nd percentile of no-shift scene splits →
   **not distinguishable from scene sampling.** Withdrawn claims: "naive conformal fails under
   val→test shift", "stronger predictors survive the shift better".
3. Per-step bands keep only ~72–75% of whole trajectories inside (target 90%). Simultaneous
   bands recover ~86–87%, at ~2× the radius (ST-GAT 11.0 m → 21.9 m).
4. (2026-10-27) Trajectron++ was re-scored on the canonical windows: its band is 13.8 m
   (mid-table), not the ~10 m "tightest band" of the dense-protocol dump.
5. ✅ **Survives:** per-class radii fix per-class unfairness, averaged over 100 scene splits
   (global radius: Car ~78–81%, Pedestrian ~94–96% → per-class: both ~87–89%).
6. ✅ **Survives:** better forecaster → tighter band (r = 0.95 across 8 models).

## Next step

Bands of 11–16 m are too wide to plan with. Tighter trajectory-level bands are the next
research step (method notes are kept locally, outside the public repo).

## Key Files

| File | Purpose |
|------|---------|
| `scripts/conformal_wrapper.py` | Wrapper: both bands, both regimes, scene folds, CIs |
| `scripts/dd_bootstrap.py` | Scene-level bootstrap helpers |
| `scripts/conformal_alpha_sensitivity.py` | 80/90/95% sweep (scene-fold means) |
| `scripts/dump_predictions.py` | Neural-model dumps (now with `scene_id`) |
| `outputs/benchmark_v2/conformal_metrics.json` | Full results |
| `tests/test_conformal.py` | Radius math, scene folds never split a scene, trajectory coverage, simultaneous guarantee |

Related: [[bootstrap-ci]], [[fairness-audit]], [[planner]]
