# Oracle-vs-Deployable Gap Decomposition (Trajectron++)

## Status: ✅ Fully Done (re-scored on the canonical windows, 2026-10-27)

## The Gap

Most-likely (deployable): ADE 3.91 vs Oracle best-of-20: ADE 2.08 → **gap = 1.83m**

## Decomposition

Best-of-K curve (K=1,2,5,10,20): **4.51 → 3.54 → 2.79 → 2.38 → 2.08**

| Readout | Test ADE | Meaning |
|---------|---------|---------|
| Random single sample (mean-sample) | 4.55 | Blind draw |
| Most-likely (deployable) | **3.91** | argmax latent + GMM mode |
| Oracle best-of-20 | 2.08 | Min over samples (hindsight) |

- **Calibration value = 0.64m** (random → most-likely): mode selection is informative
- **Coverage value = 1.83m** (most-likely → oracle): cost of committing to one future
- Curve still falling at K=20 → genuine multimodality, not sampling noise

Earlier numbers (2.58 / 1.37 / gap 1.21 m) were from the dense node-timestep protocol —
see [[benchmark-v2]] correction.

## Key Files

- Plot: `outputs/benchmark_v2/gap_decomposition.png`
- Data: `summary.json` → `trajectronpp_gap_decomposition` (from `trajectronpp_canonical_seed*.json`)
- Regenerate: `python scripts/plot_gap_decomposition.py`
