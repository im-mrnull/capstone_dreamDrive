# IDD-3D Trajectory Prediction - Quick Start

**Goal:** Convert raw IDD-3D annotations → multi-agent trajectory prediction models

---

## Pipeline Overview

```
Raw IDD-3D JSONs 
  → (1) Per-scene positions CSV 
  → (2) Consolidated trajectories 
  → (3) Single-agent windows NPZ 
  → (4) Multi-agent per-scene tensors 
  → (5) ST-GAT format NPZ (used by MFP also)
  → Train models
```

---

## Step-by-Step Pipeline

### 1. Parse raw labels → positions CSV
```bash
python scripts/1_parse_idd3d_positions.py --dataset-root idd-3d_dataset
```
- **Output:** `outputs/parsed_positions/<scene>.csv` (x, y positions per frame)

### 2. Consolidate → single trajectories CSV
```bash
python scripts/2_consolidate_trajectories.py
```
- **Output:** `outputs/trajectories/trajectories.csv` (per-agent tracks)

### 3. Create sliding windows (obs=20, pred=60, stride=3)
```bash
python scripts/3_make_windows_obs20_pred60.py
```
- **Output:** `outputs/windows_obs20_pred60.npz` (N=4821 windows, centered on last obs)
---

## Models & Results

### Model 1: Constant Velocity (CV)

**Basic (vel_k=1):**
```bash
python scripts/model1_baseline_cv_eval.py --vel-k 1 \
  --save-metrics-json outputs/baselines/cv_metrics_obs20_pred60_velk1.json \
  --save-pred-npz outputs/baselines/cv_pred_obs20_pred60_velk1.npz
```
- **Results:** ADE **3.42** | FDE **7.64**

**Smoothed (vel_k=5):**
```bash
python scripts/model1_baseline_cv_eval.py --vel-k 5 \
  --save-metrics-json outputs/baselines/cv_metrics_obs20_pred60_velk5.json \
  --save-pred-npz outputs/baselines/cv_pred_obs20_pred60_velk5.npz
```
- **Results:** ADE **2.67** | FDE **6.30**

---

### Model 2: Single-Agent MLP

```bash
python scripts/model2_baseline_mlp_train_eval.py --input outputs/windows_obs20_pred60.npz
```
- **Val Results:** ADE **2.70** | FDE **6.37**
- Simple 2-layer MLP baseline

---

### Model 3: ST-GAT (Multi-Agent, Single Future)

**Best run (MHA + AMP, early-stopped):**
```bash
./.venv312/Scripts/python.exe scripts/model3_stgat_train_eval.py \
  --data outputs/stgat_dataset_obs20_pred60.npz \
  --device cuda --pool mha --amp \
  --epochs 300 --patience 0 --batch-size 128 \
  --save-metrics-json outputs/baselines/stgat_metrics_obs20_pred60_cuda_mha_amp_noes_v5.json \
  --save-ckpt outputs/baselines/stgat_ckpt_obs20_pred60_cuda_mha_amp_noes_v5.pt \
  --save-pred-npz outputs/baselines/stgat_valpred_obs20_pred60_cuda_mha_amp_noes_v5.npz
```
- **Best Val Results:** ADE **1.87** | FDE **4.48**
- LSTM encoder + graph attention + LSTM decoder

---

### Model 4: MFP (Multi-Future Prediction)

**Architecture:** Predicts K candidate futures per agent

**Training:** Best-of-K (variety) loss + learned mode-scoring head for meaningful top-1

**Best run (K=10, learned top-1):**
```bash
./.venv312/Scripts/python.exe scripts/model4_mfp_train_eval.py \
  --data outputs/stgat_dataset_obs20_pred60.npz \
  --device cuda --k 10 --batch-size 64 \
  --epochs 300 --patience 0 \
  --mode-cls-weight 0.1 --top1-by argmax --log-every 1 \
  --save-metrics-json outputs/baselines/mfp_metrics_obs20_pred60_cuda_k10_noes_argmax_log1_v2.json \
  --save-ckpt outputs/baselines/mfp_ckpt_obs20_pred60_cuda_k10_noes_argmax_log1_v2.pt \
  --save-pred-npz outputs/baselines/mfp_pred_val_obs20_pred60_cuda_k10_noes_argmax_log1_v2.npz
```

**Results (K=10):**
- **Oracle best-of-K (epoch 74):** ADE **1.05** | FDE **1.92**
- **Learned top-1 (epoch 9):** ADE **2.18** | FDE **4.86**

**Key insights:**
- Adding mode-scoring head (`--mode-cls-weight 0.1`) makes top-1 meaningful by learning to predict which mode is most likely, dramatically improving single-trajectory prediction (7.33 → 2.18 ADE).
- **Oracle best-of-K:** Pick the best mode after seeing GT (upper bound) - not available at test time
- **Top-1:** Single deterministic prediction (selecting 1 trajectory from K modes)
  - **Purpose: Makes MFP comparable to ST-GAT (which outputs only 1 trajectory)**
  - Old approach: fixed mode index → poor results (ADE 7.33)
  - New approach: learned mode-scoring head → competitive results (ADE 2.18)
  - **Learned top-1:** Uses a neural network head (3-layer MLP) that scores all K modes and selects the highest-scoring one via `argmax(logits)`. Trained with auxiliary cross-entropy loss to predict which mode is closest to ground truth.
  - MFP top-1 vs ST-GAT: Both produce single outputs, but MFP chooses from K diverse predictions

---

## Complete Performance Summary (All 10 Models/Variants)

| # | Model | Config | Val ADE | Val FDE | Notes |
|---|-------|--------|---------|---------|-------|
| 1 | **CV (vel_k=1)** | No learning | 3.42 | 7.64 | Last-step velocity |
| 2 | **CV (vel_k=5)** | No learning | 2.67 | 6.30 | Smoothed velocity |
| 3 | **MLP** | Single-agent, CPU | 2.70 | 6.37 | 2-layer neural baseline |
| 4 | **ST-GAT v1** | Multi-agent, CPU, 25 epochs | 2.37 | 5.80 | Initial implementation |
| 5 | **ST-GAT v2** | Multi-agent, CUDA, 120 epochs | 2.31 | 5.54 | GPU acceleration |
| 6 | **ST-GAT v3** | Simple pool + AMP, early-stop | 1.89 | 4.53 | Optimized training |
| 7 | **ST-GAT v4** | MHA pool + AMP, no early-stop | **1.87** | **4.48** | **Best single-future** |
| 8 | **MFP (oracle K=10)** | Best-of-10, no mode head | **0.97** | **1.89** | **Upper bound** (fixed mode 0 top-1: 7.33/19.83) |
| 9 | **MFP (K=10)** | Best-of-10 oracle + learned top-1 | **1.05** / 2.18 | **1.92** / 4.86 | **Best oracle** / Meaningful top-1 |
| 10 | **Trajectron++ (K=50)** | Multi-agent CVAE, oracle best-of-50 | **13.32** | **14.57** | ✅ Corrected (was 163 w/ wrong dt) |
| 11 | **Trajectron++ (K=1)** | Most-likely mode prediction | **25.85** | **65.76** | ✅ Corrected (was 156 w/ wrong dt) |

**Note on Trajectron++:** Initial results were 60x too high due to incorrect dt=0.4 (should be 0.1 for 10fps data). After correction, metrics improved dramatically but still lag behind other baselines. This may indicate need for hyperparameter tuning or architecture adaptation for vehicle trajectory prediction.

---

### Key Insights

1. **Best Oracle Performance:** MFP oracle (K=10, fixed mode (no learned mode)) → ADE **0.97** / FDE **1.89**  
2. **Best Single-Future:** ST-GAT MHA v4 → ADE **1.87** / FDE **4.48**  
3. **Best Learned Top-1:** MFP learned mode → ADE **2.18** / FDE **4.86**

4. Using **top-1** so that it **makes MFP comparable to ST-GAT** (which outputs only 1 trajectory becuase it **single future**)
5. **Learned top-1:** Uses a neural network head (3-layer MLP) that scores all K modes and selects the highest-scoring one via **`argmax(logits)`**. Trained with **auxiliary cross-entropy loss to predict which mode is closest to ground truth.**

**Major Improvement:** Adding learned mode-scoring head to MFP improved top-1 from **7.33 → 2.18 ADE** while maintaining competitive oracle performance.

---

## Key Concepts

### Window Centering
All windows are **centered on the last observed point**: `obs[:, -1, :] = [0, 0]`

### Scene-Level Split
Train/val split by **scene** (not by window) to avoid data leakage

---

## Tuning Tips

**For better oracle best-of-K:** Set `--mode-cls-weight 0` (pure variety loss)

**For better top-1:** Increase `--mode-cls-weight` to 0.2 or 0.5

---

## Environment

- **GPU:** RTX 4090
- **Python:** 3.12 (`.venv312` for CUDA compatibility)
- **Key libs:** PyTorch, NumPy, Pandas
