# Oracle-vs-Deployable Gap Decomposition (Trajectron++)

## Status: ✅ Fully Done

## The Gap

Most-likely (deployable): ADE 2.58 vs Oracle best-of-20: ADE 1.37 → **gap = 1.21m**

## Decomposition

Best-of-K curve (K=1,2,5,10,20): **3.04 → 2.36 → 1.83 → 1.57 → 1.37**

| Readout | Test ADE | Meaning |
|---------|---------|---------|
| Random single sample (best-of-1) | 3.04 | Blind draw |
| Most-likely (deployable) | **2.58** | argmax latent + GMM mode |
| Oracle best-of-20 | 1.37 | Min over samples (hindsight) |

- **Calibration value = 0.47m** (random → most-likely): mode selection is informative
- **Coverage value = 1.21m** (most-likely → oracle): cost of committing to one future
- Curve still falling at K=20 → genuine multimodality, not sampling noise

## Key Files

- Plot: `outputs/benchmark_v2/gap_decomposition.png`
- Data: `summary.json` → `trajectronpp_gap_decomposition`
- Regenerate: `python scripts/plot_gap_decomposition.py`
