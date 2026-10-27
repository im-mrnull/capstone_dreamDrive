# Closed-Loop Test v2 — cross-validated, fixed instrument, fixed planner

## Status: 📌 Pre-registered (2026-10-27) — results pending

The first closed-loop test ([[closed-loop-test]], 22 scenes) could not separate the predictors.
It also showed three problems:
- a ~6% false-collision floor;
- 90% bands that paralysed the planner;
- too few scenes.

This protocol fixes all three. It was committed **before** any fold model drove any of the
primary scenes. The result JSON records the commit as `rule_commit`.

## 1. Instrument v2 (collision measurement)

All of these were calibrated on the recorded human driver's replay only, so they involve no
predictor.

- **Ego-pose glitches:** the motion reconstruction sometimes jumps the ego more than 1 m off
  its 9-frame median path. Episodes containing such a frame are excluded for every package.
- **Ego orientation:** taken from the logged pose yaw plus the measured axis offset, both for
  the human replay and for the route's orientation along its length. This is the frame the
  annotations were made in; the path tangent is noise at walking pace.
- **Exact oriented rectangles** replace the circle approximations for ground-truth contacts.
- **Ego footprint 3.4 × 1.7 m, centred 0.3 m ahead of the sensor.** This is the region no
  annotated agent edge enters around the sensor at a robust percentile, slow frames included.
  The planner still plans with its larger circle footprint.
- **Collision tolerance:** a contact counts beyond 0.25 m of overlap.
- **Result:** the human replay shows at-fault contacts in 0/72 val, 0/152 test and 6/524
  former-train episodes (1.1%), down from 0%, 5.7% and 10.3%. Replaying the human's timing on
  the smoothed route gives 8/524.

## 2. Planner v2 (band / planner interaction)

A 30-configuration grid ran on the 32 development scenes (frozen val + test) with the 8
original single-trajectory models at seed 0
([`closed_loop_dev.py`](../../scripts/closed_loop_dev.py),
`outputs/benchmark_v2/closed_loop_dev_grid.json`). Its parameters:
- hard-constraint horizon 0.5 / 1 / 3 s;
- soft penalty beyond it, weight 0 / 5;
- fallback: least penetration or straight emergency brake;
- band scale 0 / 0.5 / 1.

The criterion was fixed before the run: the highest mean episode score across the 8 models.
**Chosen: band ×0, hard constraint 1.0 s, soft weight 5, least-penetration fallback.**

On dev, every band setting scored lower:
- the best ×1 configuration: 0.813;
- the best ×0.5: 0.862;
- ×0: 0.894.

Harsh braking fell from 37–50% to about 9%.

## 3. Cross-validation

- 5 folds over all 105 scenes ([`make_cv_folds.py`](../../scripts/make_cv_folds.py)): md5 rank,
  dealt round-robin.
- Per fold: 21 test, 10 val (model selection + conformal calibration), 74 train.
- Models: CV, MLP, ST-GAT, map-free transformer, LED (mean-denoised). One seed per fold,
  trained with the exact benchmark-v2 settings.
- Every scene is driven by the models of the fold in which it is a test scene.
- **Primary set: the 73 scenes of the frozen train split.** No closed-loop result was ever
  looked at on them.
- Development set (32 scenes) and all 105 scenes are reported separately.
- References (not eligible): human driver replay, oracle (true future), ignoring agents.

## 4. Metrics and decision rule

The episodes, personas and episode score are unchanged from [[closed-loop-test]]: 0 after an
at-fault collision, otherwise (5·progress + 2·no-harsh-braking + 2·no-near-miss) / 9.

**Pick:** on the primary set, the highest mean score. Packages whose paired 95% scene-bootstrap
CI vs the leader includes 0 are tied. Ties are then broken by the lowest rate of at-fault
collisions with pedestrians or two-wheelers (paired), then by latency.

**Pre-registered hypotheses** (primary set, one-sided paired scene bootstrap, Holm-corrected
across all 16):
- **H1:** ST-GAT, the transformer and LED each score higher than CV.
- **H1b:** the same three each have fewer at-fault collision episodes than CV.
- **H2:** each of the 5 packages has fewer at-fault collision episodes than ignoring agents.
- **H3:** for each package, the chosen band setting (×0) scores higher than the full 90% band
  (×1), with the same planner.

## 5. Amendment (2026-10-27, before any primary result): instrument sensitivity

Every instrument setting above was calibrated on the human replay. To show the conclusions
don't hinge on those exact values, the same pre-registered analysis (primary set, same
hypotheses, same decision rule) is re-run with one setting varied at a time
(`python scripts/closed_loop_cv.py sensitivity` → `closed_loop_cv_sensitivity.json`):

| Setting | Variants (primary in bold) |
|---|---|
| Collision tolerance | 0, **0.25**, 0.5 m |
| Ego footprint | **3.4 × 1.7 m, centred +0.3 m (measured)**; 4.5 × 1.7 m, centred +0.5 m (nominal car) |
| Pose-glitch threshold | 0.5, **1.0**, 2.0 m |
| Agent footprint sizes | −20%, **nominal**, +20% |

The headline stays the primary analysis. For each hypothesis the report states in how many of
the 7 variants it remains supported. A hypothesis is called **robust** only if it is supported
in the primary analysis and in every variant.

**Caveats:**
- the agents don't react;
- footprints are nominal;
- there is one seed per fold;
- the planner was tuned on the development scenes (excluded from the primary set).

Related: [[closed-loop-test]], [[safety-scorecard]], [[bootstrap-ci]]
