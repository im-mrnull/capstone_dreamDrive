# Conformal Prediction Wrapper

## Status: ✅ Fully Done — All 6 Deployable Models

## What It Does

Post-hoc wrapper that turns each deployable point predictor into a per-timestep uncertainty
band with a distribution-free coverage guarantee. No retraining needed.

- Nonconformity score = per-step position error
- Frozen splits ensure statistically clean calibration
- Covers all 6 deployable models: CV, MLP, K-mode top-1, ST-GAT, transformer, Trajectron++

## Four Key Findings (α=0.1, 90% target)

### 1. Method is valid when calibration matches deployment
Split-conformal within test set hits 89.0–90.9% for **every** model. ✅

### 2. Naïve conformal fails under val→test shift (important negative)
Calibrated on val, deployed on test → coverage collapses (73.8–84.8%).
The exchangeability assumption breaks with distribution shift.

### 3. Stronger predictors survive the shift better
| Model | Regime B (valid) | Regime A (shifted) | Band radius |
|-------|---------|----------|-------------|
| Trajectron++ | 89.9% | 84.8% | 10.0m |
| ST-GAT | 90.2% | 79.3% | 10.8m |
| CV | 89.0% | 73.8% | 15.9m |

Accuracy→efficiency correlation: r = 0.93.

### 4. Class-conditional radii fix per-class unfairness
Global radius: Car 78% coverage, Pedestrian 96% (unfair both ways).
Per-class radii: Car → 91%, Pedestrian → 89% (band halved: 11.7→5.6m).

## Alpha Sensitivity (robust across targets)

Tested at 80%/90%/95% — all findings hold. Not an artifact of the 90% choice.

## Key Files

| File | Purpose |
|------|---------|
| `scripts/conformal_wrapper.py` | Main wrapper (12KB) |
| `scripts/conformal_alpha_sensitivity.py` | Multi-alpha sweep |
| `scripts/dump_predictions.py` | Generate prediction dumps for neural models |
| `outputs/benchmark_v2/conformal_metrics.json` | Full results (78KB) |
| `outputs/benchmark_v2/conformal_sensitivity.json` | Alpha sensitivity |
| `outputs/benchmark_v2/conformal_coverage*.png` | Per-model coverage plots |
| `outputs/benchmark_v2/uncertainty_efficiency.png` | Accuracy→band plot |
| `outputs/benchmark_v2/robustness_gap.png` | Shift robustness plot |
| `tests/test_conformal.py` | 4 tests: radius math, coverage, multi-alpha, class-conditional |
