# Ego-motion quality: sensitivity of the reported results

> Status: ✅ done 2026-11-26 (development analysis; plan fixed at 00:10 IST, before any outcome on a clean-scene subset was computed).
> User decision 2026-11-26: "yes run the sensitivity analysis on clean scenes".
> Background: [[chaos-gap]] section "Ego-motion estimate reversed, not a flipped yaw", and [[ego-motion-compensation]].

## Why

IDD-3D has no official ego poses. `0_compensate_ego_motion.py` estimates them with RANSAC on the annotations, and in
many scenes the estimate is wrong in places: travel reversed, speed jumps of up to 24 m/s within 0.5 s, or many
fallback frames. A wrong ego estimate moves every road user's world position by the same error, so ground truth
and model inputs in those scenes carry an artefact. This page checks whether the reported conclusions hold on
the scenes whose ego estimate passes the quality gate.

## Scene sets (fixed)

- **Gate:** the **ego-only** part of `scripts/ego_quality.py`, the same rule on every split:
  - travel not reversed;
  - at most 10 RANSAC fallback frames;
  - the estimated ego speed changes by at most 5 m/s within 0.5 s.

  The road-user checks were defined for dev only and are not applied here.
- **Clean scenes:** 32 of 105.

  | Split | Clean | Scenes |
  |---|---|---|
  | Dev | 26 of 83 (23 train, 3 val) | from `outputs/data_quality/ego_quality_dev.json` |
  | Test | 6 of 22 | 103308_seq_20, 110338_seq_11, 110338_seq_12, 112945_seq_22, 112945_seq_24, 115526_seq_2 (from `ego_quality_test_egoonly.json`) |

  - Clean val scenes: 103808_seq_3 and 112945_seq_26 (chaos-gap half A), 103308_seq_5 (half B).
  - The full dev gate (with road users) passes 25 dev scenes; the difference is 103808_seq_20 (train).
- **Flagged scenes:** the other 73. They are reported next to the clean ones for contrast.
- **SHA-256:**
  - `ego_quality_dev.json` d092f6f0dca8bb3a6a254a03725a6c84323da763bf2b4bb1f8915c07313b3ed3
  - `ego_quality_test_egoonly.json` 7914f32e3c648f3085e17313ecc49e945da5bd6c37ea158522e4035a62ce8994
  - `scripts/ego_quality.py` 9c96515d27ca9711b6985da15519be374bd2ac104579faa727d30d20019945a4

## What is re-scored

Existing outputs only: saved predictions, episodes and per-scene sums. Nothing is retrained or re-simulated, and
the planner settings, bands and models stay exactly as reported. Each analysis first reproduces the reported
number on the original scene set; a mismatch stops it.

| # | Reported result | Source | Original scenes | Clean subset |
|---|---|---|---|---|
| S1 | Open-loop ADE/FDE per model, ranking, paired differences (3-seed means) | `bootstrap_ci.json`, `summary.json` (seed dumps) | test 22 | test 6 |
| S2 | Conformal coverage, regime A (calibrated on val, applied to test) | `conformal_metrics.json` | test 22 | test 6 (calibration unchanged) |
| S3 | Closed-loop test: package scores and the pre-registered pick | `closed_loop.json` (`episodes_test.json`) | test 22 | test 6 |
| S4 | Closed-loop CV: hypotheses H1, H1b, H2, H3, S1, S2, the pick, and the amendment-3 tests | `closed_loop_cv.json`, `closed_loop_cv_extra.json` (`episodes_cv.json`) | primary 73 train; dev 32; all 105 | primary 23; dev 9; all 32 |
| S5 | Planner headroom: cell scores and at-fault rates, the planner effect, the G2-matched gate | `headroom_matched.json` (per-scene sums) | dev 32 (val + test) | 9 |
| S6 | Multi-future scorecard, per metric | `multifuture_scorecard.json` (seed dumps) | test 22 | test 6 |
| S7 | Chaos gap, dev: zero-shot ADE and G₀ (zero-shot minus IDD-trained) | `phase2_seeds_table.json`, `ft_budget_diagnosis.json` (dumps) | IDD val 10 (halves A and B) | 3 (descriptive only) |

The bootstrap settings are the originals: scene resampling, the same replicate counts and seeds, and for seed
pooling the same procedure.

## Reading rule (fixed before any clean-subset number was seen)

For each conclusion:
- **Holds:** on the clean subset, the same verdict (supported or not, the pick, the ordering) and the same sign of
  the point estimate.
- **Not determinable on clean scenes:** the point estimate keeps its sign, but the verdict changes only because
  the CI widened (fewer scenes). This is a loss of power, not a reversal.
- **Changes:** the point estimate changes sign, or the clean-subset CI excludes the original point estimate.
  Reported as a finding about the data defect.

Further rules:
- The clean and flagged subsets are reported side by side, together with scene descriptors (road users per
  scene, ego speed, class mix). Clean scenes are not a random sample, so a clean-vs-flagged difference has two
  possible sources, the defect or the scene type, and is described, not attributed.
- **Scope:**
  - This checks evaluation only. The models were trained on data that includes flagged scenes; retraining on
    clean scenes is a separate experiment and is not part of this one.
  - The closed-loop simulator properties fixed only in the demo (crawl routes, sideways sliding, 0.25 m snap) are
    a separate issue and stay as reported.
- **Not included:** the confidential planner's runs.

## Chaos gap: addition to the pre-registration draft

For the single frozen-test run of the chaos-gap study, the primary endpoint (zero-shot against IDD-trained ADE,
per model, with scene-bootstrap CIs) is **also** reported on the 6 ego-clean test scenes listed above. This is a
secondary sensitivity; the primary analysis keeps all 22 scenes. The list is fixed by the hash above.

Possible conflict, stated openly: S1 shows the IDD-trained models' test ADE on these 6 scenes before that run.
These are the same benchmark-v2 rows whose 22-scene values were already public, and the AV2 zero-shot side is
not read. The addition is written before S1 runs.

## Results (2026-11-26; script `scripts/ego_sensitivity.py`, outputs `outputs/data_quality/sensitivity_*.json`)

Every part first reproduced the reported numbers on the original scenes from the saved files:
- S1 seed 0 exactly; the seed-pooled per-seed ADEs exactly. `bootstrap_ci.json` pooled 3 models, 6 have seed dumps
  now, so the pooled CIs agree only approximately.
- The 3-seed means equal `summary.json`.
- S2–S7 exactly.
- S4 amendment 3: see below.

### S0: the clean scenes are a different kind of scene

| | Test clean (6) | Test flagged (16) | Dev clean (26) | Dev flagged (56) |
|---|---|---|---|---|
| Road users per scene (median) | 134 | 71 | 164 | 47 |
| Vehicle / two-wheeler / pedestrian share | 0.35 / 0.34 / 0.26 | 0.47 / 0.41 / 0.11 | 0.38 / 0.32 / 0.28 | 0.51 / 0.40 / 0.10 |
| Moving share, world frame (carries the defect) | 0.57 | 0.77 | 0.60 | 0.85 |
| Median speed of moving road users, world frame (carries the defect) | 5.1 m/s | 5.2 m/s | 2.5 m/s | 5.0 m/s |

- Clean scenes are dense and rich in pedestrians; flagged scenes are sparse and dominated by vehicles.
- A plausible mechanism, not tested: RANSAC finds enough static matches in dense pedestrian scenes, but locks onto
  moving traffic when few static objects are present.
- **Consequence:** every clean-vs-flagged difference below has two possible sources, the defect or the scene type.
  This page does not separate them.

### S1: open-loop accuracy, test

| 3-seed mean ADE (m) | ST-GAT | Transformer | LED | AgentFormer | Trajectron++ | K-mode | MLP | CV |
|---|---|---|---|---|---|---|---|---|
| All 22 scenes (reported) | 3.41 | 3.57 | 3.75 | 3.75 | 3.91 | 4.15 | 3.97 | 4.36 |
| Clean 6 | 1.48 | 1.48 | 1.39 | 1.51 | 1.49 | 1.78 | 1.58 | 1.39 |
| Flagged 16 | 4.84 | 5.12 | 5.49 | 5.41 | 5.71 | 5.91 | 5.74 | 6.57 |

Model minus CV, seed 0, paired scene-bootstrap 95% CI:

| Scenes | Result |
|---|---|
| All 22 | 5 of 7 models beat CV, e.g. ST-GAT −1.03 [−1.96, −0.32] |
| Clean 6 | **No model beats CV**: ST-GAT −0.01 [−0.40, +0.25], LED +0.03, transformer +0.04, Trajectron++ +0.06, MLP +0.17. AgentFormer +0.23 [+0.02, +0.44] and K-mode +0.46 [+0.23, +0.69] are worse than CV |
| Flagged 16 | All 7 beat CV (−0.8 to −1.8 m) |

Reading:
- The reported "learned models beat constant velocity on the test split" rests on the flagged scenes.
- On the clean scenes the CIs exclude an advantage of the reported size.
- Whether that is the defect (CV extrapolates an ego jump; learned models trained on the same artefacts may damp it)
  or the scene type (dense, slow pedestrian scenes favour CV) is not determined here.
- A within-scene check could separate them: windows that touch an ego-jump frame vs windows that do not, inside the
  flagged scenes.

### S2: conformal coverage, regime A (bands calibrated on val, as reported)

| Step coverage, target 0.90 | All 22 | Clean 6 | Flagged 16 |
|---|---|---|---|
| Range over the 8 models | 0.73–0.81 | 0.90–0.97 | 0.58–0.69 |
| Example: ST-GAT | 0.79 [0.67, 0.89] | 0.97 [0.95, 0.99] | 0.67 [0.53, 0.79] |

- The reported under-coverage under the val→test shift is concentrated in the flagged test scenes.
- On the clean ones the val-calibrated bands reach the target. K-mode is the lowest at 0.898 [0.84, 0.94]; the others
  over-cover.
- 7 of the 10 calibration (val) scenes are themselves flagged.

### S3: closed-loop test (pre-registered decision rule)

- The pick is CV on all 22, the clean 6 and the flagged 16 scenes.
- 11 of 12 packages are tied with the leader on score on the clean scenes (10 of 12 on all 22). In every subset the
  pick comes from tie-break 2 (latency), as reported.
- **Instrument check:** the at-fault rule fires on the human's own logged drive and on the perfect-prediction oracle
  in the flagged scenes, never in the clean ones.

  | At-fault rate | All 22 | Clean 6 | Flagged 16 |
  |---|---|---|---|
  | Human log replay | 5.7% | 0% | 7.8% |
  | Oracle | 6.2% | 0% | 8.6% |

  This is consistent with phantom contacts created by the defect.
- **Ignoring agents vs CV, at-fault:** 15.9% vs 12.5% on all scenes; 4.2% vs 4.2% on the clean scenes (48 episodes
  per package and seed).

### S4: closed-loop CV (pre-registered hypotheses)

- **Primary 73 frozen-train scenes:** all 16 hypotheses (H1, H1b, H2, H3) have the same verdict on the 23 clean and on
  the 50 flagged scenes, and the pick is CV in all three.
  - H1 and H1b (no learned model beats CV) are not supported anywhere.
  - H2 (each predictor beats ignoring agents) and H3 (the chosen band setting) are supported everywhere.
- **At-fault rates, primary scenes:**

  | Scenes | Ignoring agents | Single-path predictors | Set packages (K-mode, LED) |
  |---|---|---|---|
  | All 73 | 16.4% | 2.1–4.6% | 13.4%, 7.4% |
  | Clean 23 | 24.5% | 0.5–1.6% | 16.8%, 9.2% |
  | Flagged 50 | 12.1% | 2.4–6.2% | 11.5%, 6.5% |

- "Prediction is needed for safety" and "sets as obstacles hurt" hold, and the first is stronger on the clean scenes.
- **Development sets (val + test):** the pick stays CV on the 9 clean scenes; on the 23 flagged ones it is the MLP.
- **Amendment 3** (MF transformer, AgentFormer, Trajectron++ and their sets): the amendment-3 episodes were never
  saved, so they were re-simulated (CPU, `episodes_cv_extra.json`). The re-simulation reproduces
  `closed_loop_cv_extra.json` exactly. No amendment-3 verdict changes on the clean or flagged primary scenes.

### The reading rule applied (`sensitivity_report.json`)

| Part | Clean subset | Flagged subset |
|---|---|---|
| S1 | changes 12, holds 6, not determinable 4 | changes 1, holds 17, not determinable 4 |
| S2 | changes 8 | holds 7, not determinable 1 |
| S3 | changes 5, holds 8 | changes 2, holds 11 |
| S4 | changes 11, holds 60 | changes 14, holds 57 |
| S5 | changes 1, holds 7, not determinable 3 | changes 3, holds 7, not determinable 1 |
| S6 | changes 2, holds 1 | changes 1, holds 2 |

**Post-hoc note on the rule (added after seeing the output; the rule itself is unchanged):** "changes" includes any
sign flip.
- **S4 and S5:** almost every "changes" is a sign flip of a near-zero estimate that is not significant either before
  or after (e.g. H1 ST-GAT score vs CV, −0.003 → +0.010).
  - Only one S4 verdict differs: on the 23 flagged development scenes, ST-GAT beats CV on score (Holm p 0.022).
  - On the primary scenes, every pre-registered verdict is the same in both subsets.
- **S1, S2 and S6:** the "changes" are substantive. On the clean scenes:
  - the CIs exclude the reported learned-vs-CV advantages;
  - coverage moves from below the target to at or above it;
  - the best multi-future model changes.

### S5: planner headroom (public cells)

- **Matched planners, dev scenes with eligible episodes** (25: 9 clean, 16 flagged):
  - **Planner effect** (proposal minus v2, per predictor): +0.042 to +0.050 → clean +0.039 to +0.059, flagged
    +0.040 to +0.044. This holds.
  - **Ideal set vs the best ideal single, under the proposal planner:** +0.008 [−0.001, +0.017] → clean
    +0.012 [−0.003, +0.029], flagged +0.003 [−0.005, +0.011]. Still within noise.
  - **G2-matched (public planners):** does not pass in any subset.
- **Confirmation, 62 primary scenes with eligible episodes** (23 clean): H4, the margin single beating the set on
  score, goes from −0.013 [−0.024, −0.002] to −0.014 [−0.034, +0.007] on the clean scenes. Same direction, CI now
  includes 0: not determinable on the clean scenes.

### S6: multi-future scorecard, test (K = 10)

| Metric | All 22 (reported) | Clean 6 | Flagged 16 |
|---|---|---|---|
| minADE, best | K-mode 2.01 | LED 0.85, Trajectron++ 0.89 (K-mode 1.24, the worst) | K-mode 2.58 |
| Energy score, best | K-mode 3.59 | Trajectron++ 1.40 (K-mode 1.65) | K-mode 5.02 |

- K-mode's lead on the reported scorecard comes from the flagged scenes.
- On the clean scenes the samplers (LED, Trajectron++) lead; the CIs overlap.

### S7: chaos gap, dev (descriptive; only 3 clean val scenes)

G₀ = ADE(AV2 zero-shot) − ADE(IDD-trained, from scratch, k = 72), option 1, both halves pooled.

| Model | All 10 | Clean 3 | Flagged 7 |
|---|---|---|---|
| MLP | +0.75 | +0.73 | +0.76 |
| ST-GAT | −0.53 | −0.05 | −1.15 |
| Transformer | −0.30 | +0.14 | −0.88 |
| AutoBot | −0.65 | +0.02 | −1.52 |
| K-mode (MFP) | −0.67 | −0.71 | −0.62 |
| AgentFormer | −0.05 | −0.26 | +0.23 |
| LED | +0.08 | +0.00 | +0.19 |

- The negative gaps ("the US model is better") of ST-GAT, the transformer and AutoBot come from the flagged val
  scenes. This includes `103308_seq_27` (AutoBot's half-B case).
- The MLP's positive gap is the same on both subsets.
- This supports reporting the frozen-test chaos-gap endpoint on the clean test scenes too (added above, before S1 ran).
