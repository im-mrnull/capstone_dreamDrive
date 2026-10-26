# Negative Results (Honest, Documented)

## Status: ✅ All Three Fully Done

---

## 1. Hetero Trajectron++ (Unicycle Dynamics)

**Hypothesis:** Modeling wheeled classes with unicycle dynamics (nuScenes-style) would improve
predictions by incorporating heading information.

**Result:** ❌ Degrades deployable most-likely ADE from 2.58 → **2.84** (+0.26m).
- Loss concentrated on moving vehicles (riders +0.6m, cars +0.5m)
- Oracle coverage is a wash (1.37 vs 1.37)
- **Root cause:** Heading estimated from velocity direction (no true yaw in data). Unicycle
  integrates heading noise into deterministic rollout. Sampling absorbs it, most-likely can't.
- **Prerequisite for fix:** True yaw from raw dataset or perception pipeline.

**Artifacts:** `tpp_hetero_grid/`, `trajectronpp_hetero_seed{0,1,2}.json`

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
