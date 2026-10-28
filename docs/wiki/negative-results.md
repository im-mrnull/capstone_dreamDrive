# Negative Results (Honest, Documented)

## Status: ✅ All Five Fully Done

---

## 1. Hetero Trajectron++ (Unicycle Dynamics)

**Hypothesis:** Modeling wheeled classes with unicycle dynamics (nuScenes-style) would improve
predictions by incorporating heading information.

**Result:** ❌ Degrades deployable most-likely ADE from 3.91 → **4.55** (+0.64m, canonical
windows, 3 seeds; was 2.58 → 2.84 on the dense protocol).
- Loss concentrated on moving vehicles (riders +1.0 to +1.3m, cars +1.0m; pedestrians −0.15m)
- Oracle slightly worse too (2.08 → 2.22)
- **Root cause:** Heading estimated from velocity direction (no true yaw in data). Unicycle
  integrates heading noise into deterministic rollout. Sampling absorbs it, most-likely can't.
- **Prerequisite for fix:** True yaw from raw dataset or perception pipeline.

**Artifacts:** `tpp_hetero_grid/`, `trajectronpp_hetero_canonical_seed{0,1,2}.json`

---

## 2. EWTA Annealing (K-Mode Decoder)

**Hypothesis:** Evolving winner-takes-all (annealing winner set 10→1) would improve mode
selection over plain WTA.

**Result:** ❌ Plain WTA won outright.
- Plain WTA: val top-1 ADE **2.46**
- Hard EWTA: 2.60
- Soft EWTA: 2.76

Not promoted to 3-seed run.

**Artifacts:** `mfp_ewta_grid/`

---

## 3. Content-Aware Mode Scorer

**Hypothesis:** Scoring modes by their decoded trajectory content (endpoint, waypoints, path
length, heading) instead of just context would improve top-1 selection.

**Result:** ❌ Wash on val (2.51 vs 2.46 baseline). Not promoted to 3-seed.
- Unit test confirms scorer reads trajectory (perturbing one mode's path moves only that logit)
- **Root cause:** Decoder is deterministic → trajectory features are a fixed function of the
  per-mode context state the scorer already uses. Redundant, not new information.
- **Implication:** Bottleneck is WTA assignment target, not scorer input. Next attack should
  target objective/structure (jointly-trained mixture density, or anchor/goal modes).

**Artifacts:** `mfp_content_grid/`, `tests/test_mode_scorer.py`

---

## 4. Multi-Future Sets Under an Obstacle Planner (closed loop, confirmatory)

**Hypothesis (pre-registered, S1 / S2 in [[closed-loop-cv]]):** driving against all 10
futures of a multi-future set is safer than driving against the set's single trajectory.

**Result:** ❌ The opposite, and significantly so, on the 73 primary scenes:

| Set | Score (vs its single) | At-fault (vs its single) | Harsh braking |
|---|---|---|---|
| K-mode set | 0.626 vs 0.791 | 13.4% vs 4.6% | 63% |
| LED set | 0.682 vs 0.793 | 7.4% vs 3.4% | 56% |
| Multi-future transformer set (amendment 3) | 0.438 vs 0.790 | 19.3% vs 3.6% | 93% |
| Trajectron++ set (amendment 3) | 0.752 vs 0.799 | 4.4% vs 3.4% | 44% |
| AgentFormer set (amendment 3) | 0.793 vs 0.800 | 3.4% vs 2.9% | 25% |

None of the five sets beats its single trajectory. AgentFormer's set, which blocks the
planner least, is the only one level with it.

- **Root cause:** ten futures leave no feasible manoeuvre. The least-penetration fallback then
  drives into agents, and the car brakes hard far more often.
- **Implication:** multi-future prediction is only as useful as the planner that consumes it.
  How a planner should consume multiple futures is the open question (separate work).

## 5. Learned Predictors vs Constant Velocity (closed loop, confirmatory)

**Hypothesis (H1 / H1b in [[closed-loop-cv]]):** ST-GAT, the map-free transformer and LED
each score higher than, and collide less than, constant velocity.

**Result:** ❌ Not supported for any of them, in the primary analysis or in any of the 7
instrument variants.
- All 5 single-trajectory packages score 0.793–0.802.
- The oracle scores only 0.821, so on these scenes the planner, not prediction accuracy, sets
  the limit.

