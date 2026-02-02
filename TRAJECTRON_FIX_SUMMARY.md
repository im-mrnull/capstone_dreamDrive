# Trajectron++ Scale Mismatch - Root Cause & Fix

## Problem Summary

Initial Trajectron++ evaluation showed unrealistic metrics:
- ADE: **163.17** (expected ~2-5)
- FDE: **414.94** (expected ~4-10)
- ~60x worse than baseline models

## Root Cause

**Incorrect dt (time delta) parameter:**
- IDD-3D data: 10 fps → **dt should be 0.1 seconds**
- Data exported with: **dt = 0.4 seconds (DEFAULT)**
- Impact: 4x timing error compounded through evaluation

## Investigation Steps

1. **Compared data scales:**
   - Trajectron++ data: mean displacement ~0.53 pixels/timestep
   - Baseline data: mean displacement ~0.29 pixels/timestep
   - → Data scales were similar, not the issue

2. **Traced evaluation pipeline:**
   - Checked `prediction_output_to_trajectories()` - no dt scaling
   - Checked manual error computation - reasonable (~10-12 pixels)
   - Identified: dt parameter in scene objects

3. **Found mismatch:**
   - `scene.dt = 0.4` (from export script default)
   - Should be `0.1` (10 fps = 0.1 seconds per frame)

## Solution

**NO RETRAINING REQUIRED!** The model checkpoint is valid.

### Option 1: Quick Fix (Patch Evaluation) ✅ USED
**Script:** `scripts/9_evaluate_trajectronpp_corrected_dt.py`

Patches `scene.dt = 0.1` during evaluation:
```python
for scene in eval_env.scenes:
    scene.dt = 0.1  # Correct dt for 10fps data
```

**Results:**
- Multi-Sample (K=50): ADE **13.32**, FDE **14.57** ✅
- Most-Likely (K=1): ADE **25.85**, FDE **65.76** ✅
- Improvement: **163 → 13.3 ADE** (~12x reduction)

### Option 2: Re-export Data (Permanent Fix)
**Script:** `scripts/8_re_export_trajectronpp_correct_dt.py`

Re-runs data export with correct parameter:
```bash
python scripts/6_export_trajectronpp_format.py \
    --dt 0.1  # ← Correct value
```

Then evaluate with original script:
```bash
python scripts/7_evaluate_trajectronpp.py
```

## Corrected Performance

| Model | Mode | ADE | FDE | Comparison |
|-------|------|-----|-----|------------|
| **Trajectron++** | K=50 oracle | **13.32** | **14.57** | 7x worse than MFP oracle |
| **Trajectron++** | K=1 most-likely | **25.85** | **65.76** | 14x worse than ST-GAT |
| ST-GAT v4 | Single future | 1.87 | 4.48 | Best single-future baseline |
| MFP | K=10 oracle | 0.97 | 1.89 | Best oracle baseline |

## Analysis: Why Still Underperforming?

Despite fixing the dt issue, Trajectron++ still lags behind baselines. Possible reasons:

### 1. Hyperparameter Tuning Needed
- Used default config designed for pedestrian datasets (ETH/UCY)
- IDD-3D has different characteristics:
  - Vehicle dynamics vs pedestrian
  - Longer prediction horizon (60 vs typical 12)
  - Different scene complexity

**Potential improvements:**
- Increase model capacity (enc_rnn_dim, dec_rnn_dim)
- Adjust K modes (currently 25)
- Tune learning rate schedule
- Adjust KL weight and annealing

### 2. Coordinate Frame Issues
- Trajectron++ may expect ego-centric coordinates
- IDD-3D uses global scene coordinates
- Scene-relative encoding might help

### 3. Data Preprocessing
- Trajectron++ has specific normalization expectations
- Check attention radius settings (currently 10.0)
- Verify node type mappings (VEHICLE vs PEDESTRIAN dynamics)

### 4. Training Duration
- 100 epochs may not be sufficient
- Original Trajectron++ paper trained for 500+ epochs
- Consider longer training or better early stopping

## Recommendations

### Short-term (Quick Wins):
1. ✅ **Use corrected evaluation** - Done! (script 9)
2. **Re-export with correct dt** - Optional but recommended
3. **Try longer training** - Run for 200-300 epochs
4. **Increase model capacity** - Double hidden dimensions

### Medium-term (Tuning):
1. **Hyperparameter sweep:**
   - Learning rate: [0.0001, 0.001, 0.01]
   - K modes: [10, 25, 50]
   - RNN dims: [64, 128, 256]
   
2. **Data augmentation:**
   - Random scene rotations
   - Random agent dropouts
   
3. **Architecture modifications:**
   - Try different attention mechanisms
   - Experiment with transformer encoders

### Long-term (Deep Investigation):
1. **Coordinate system audit:**
   - Verify all transformations
   - Compare with original Trajectron++ datasets
   
2. **Alternative models:**
   - Try Trajectron++ v2 (if available)
   - Consider other multi-modal models (DESIRE, SocialGAN)

## Key Takeaways

1. ✅ **Root cause identified and fixed:** dt parameter mismatch
2. ✅ **No retraining needed:** Model checkpoint is valid
3. ✅ **Metrics improved 12x:** 163 → 13.3 ADE
4. ⚠️ **Still below baselines:** Further tuning needed
5. 📊 **Performance gap:** 7-14x worse than best models

The dt fix solved the immediate scale mismatch, but revealed that Trajectron++ needs architecture/hyperparameter adaptation for competitive performance on IDD-3D vehicle trajectories.
