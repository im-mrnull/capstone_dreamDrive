# Regime map: the draw-count addendum

**Status:** committed before any primary draw 30–59 is driven · the pre-registration,
[[headroom-test]] section 14, is unchanged

**The question:** whether multi-future prediction gives a measurable closed-loop safety
advantage under the different recognition delays and commitment intervals of section 14. The
experiment is not set up to show that it does.

## 1. Dev tuning

Draws 60–64, 32 dev scenes, 880 episodes per cell (`outputs/benchmark_v2/regime_tune.json`).

**The proposal planner's margins**, in metres (lowest dev at-fault rate, then highest score),
given as single (mean) / single (most likely) / set:

| Recognition delay Δ | R 0.2 s | R 0.5 s | R 1.0 s |
|---|---|---|---|
| 0 s | 0 / 0 / 0.5 | 0.75 / 0.75 / 0.75 | 0.5 / 0.5 / 0.25 |
| 1 s | 0 / 0 / 0 | 0.25 / 0.25 / 0 | 0.25 / 0.25 / 0.5 |
| 2 s | 2.0 / 2.0 / 0 | 0.75 / 0.75 / 0 | 0.5 / 0.5 / 0.5 |

- At Δ 2 s, R 0.2 s the top of the grid (2.0 m) won for both singles. The extension (2.5 and
  3.0 m) was run and did not beat it.
- The confidential planner's margins, and each regime's dev selection of the best single and
  the best set, are recorded with the confidential planner.

## 2. Calibration pass 2

Results are in `outputs/benchmark_v2/regime_sample_size.json`. The construction is as
pre-registered in section 14.6:
- dev: the map's own dev cells;
- evaluation: the pass 1 structure, with each regime's base rate and deviation spread scaled as
  on dev.

**Results:**
- **Nulls:** every upper bound is ≤ 0.06. P1, P2, P3 and the reading are all calibrated.
- **Power at 30 draws**, under the pre-registered alternative (an additive trend from 0 to
  1 percentage point): P1 47%, P2 1.7%, P1 and P2 together 1.4%.
- **P3** (interaction, 1 point where both factors are slow): 25%.
- No draw count reaches 80% for both P1 and P2.
- Pass 2's power is close to pass 1's unequal-noise scenario (section 14.6: P1 53%, P2 4.6% at
  30 draws).

## 3. The draw count: 30 (primary draws 30–59)

The rule sets 30 draws, because no draw count reaches 80% power for both P1 and P2.

**Fixed before the run:**
- **Joint power.** Under the pre-registered 1-percentage-point additive-trend alternative, the
  joint P1/P2 power at 30 draws is only 1.4%. Larger true effects would be detected more often.
- **If neither P1 nor P2 is significant,** the reading is "no headroom" if the corner headroom
  (best oracle − best single) is below 1 point, and "inconclusive" otherwise. "No
  regime-dependent benefit" cannot be read, because power was not reached. A non-significant P1
  and P2 cannot establish that there is no regime-dependent benefit.
- **P2 has 1.7% power** against a 1-point advantage in the corner. A non-significant P2 does
  not establish that multi-future prediction fails in the adverse corner.
- **P3 has 25% power.** If P1 and P2 are significant and P3 is not, the reading is
  "interaction inconclusive"; "additive" cannot be read. A non-significant P3 cannot establish
  that the two factors do not interact.
- **A significant result** is still evidence, because the nulls are calibrated. At this power,
  though, its estimate tends to overstate the true effect. Estimates are reported with their
  CIs.
- **The whole map is reported descriptively,** as pre-registered: the 9 cells with CIs, the
  per-regime `max_vs_max` comparisons and the headroom. The descriptive map is not statistical
  evidence: exploratory patterns are kept apart from confirmed findings.
- **The draw count is not changed** after any primary outcome has been seen.

**Related:** [[headroom-test]], [[negative-results]]
