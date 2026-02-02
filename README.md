# IDD-3D Data Processing Pipeline (PW26_CGM_04)

This workspace contains a complete preprocessing pipeline to convert raw IDD-3D annotations into trajectory windows suitable for trajectory prediction models.

## Dataset

- Dataset root expected: `idd-3d_dataset/`
- Typical structure used:
  - `idd-3d_dataset/train_val/<scene_id>/label/*.json`

## Pipeline Outputs (What We Generated)

### 1) Parsed per-frame positions (per scene)

**Goal:** extract per-object ground-plane positions over time.

- Script: `scripts/1_parse_idd3d_positions.py`
- Output folder: `outputs/parsed_positions/`
- Output format (one CSV per scene):
  - Columns: `scene_id, frame_id, obj_id, obj_type, x, y`
  - `(x, y)` comes from `psr.position.x` and `psr.position.y` in the per-frame label JSON.

Notes:
- The parser includes a lenient JSON loader to handle rare label files containing valid JSON followed by extra trailing bytes (it warns and ignores trailing content instead of crashing).

Run:
- `python scripts/1_parse_idd3d_positions.py --dataset-root idd-3d_dataset`

### 2) Consolidated trajectories (single CSV)

**Goal:** group per-scene positions into per-agent tracks.

- Script: `scripts/2_consolidate_trajectories.py`
- Output file: `outputs/trajectories/trajectories.csv`
- Output columns:
  - `scene_id, obj_id, obj_type, trajectory`
- `trajectory` is JSON list of `[frame_id, x, y]` sorted by `frame_id`.

Run:
- `python scripts/2_consolidate_trajectories.py`

### 3) Window generation for training (NPZ)

**Goal:** create sliding windows for model training.

- Script: `scripts/3_make_windows_obs20_pred60.py`
- Output file: `outputs/windows_obs20_pred60.npz`

Windowing parameters:
- `obs_len = 20`
- `pred_len = 60`
- `stride = 3`

Window format:
- `obs`: `(N, 20, 2)`
- `fut`: `(N, 60, 2)`
- `scene_id`: `(N,)`

Centering:
- Each window is centered on the last observed point:
  - `obs_centered = obs - obs[-1]`
  - `fut_centered = fut - obs[-1]`

Generated dataset stats:
- `N = 4821` windows
- Shapes:
  - `obs (4821, 20, 2)`
  - `fut (4821, 60, 2)`
  - `scene_id (4821,)`
- Centering sanity check: `max(abs(obs[:, -1, :])) == 0`

Run:
- `python scripts/3_make_windows_obs20_pred60.py`

## Baseline Results (CV)

We established a Constant Velocity (CV) baseline using the centered window data.

- Script: `scripts/model1_baseline_cv_eval.py`
- Outputs folder: `outputs/baselines/`

### CV baseline (last-step velocity, vel_k=1)

- Metrics: `outputs/baselines/cv_metrics_obs20_pred60.json`
- Predictions: `outputs/baselines/cv_pred_obs20_pred60.npz`
- Results (N=4821):
  - ADE = **3.4204**
  - FDE = **7.6426**

Run:
- `python scripts/model1_baseline_cv_eval.py --vel-k 1`

### CV baseline (smoothed velocity, vel_k=5)

- Metrics: `outputs/baselines/cv_metrics_obs20_pred60_velk5.json`
- Predictions: `outputs/baselines/cv_pred_obs20_pred60_velk5.npz`
- Results (N=4821):
  - ADE = **2.6722**
  - FDE = **6.3009**

Run:
- `python scripts/model1_baseline_cv_eval.py --vel-k 5 \
    --save-metrics-json outputs/baselines/cv_metrics_obs20_pred60_velk5.json \
    --save-pred-npz outputs/baselines/cv_pred_obs20_pred60_velk5.npz`

## Single-Agent Neural Baseline (MLP)

We benchmarked a simple single-agent neural baseline: a 2-layer MLP that maps the flattened observed positions (20×2) directly to future positions (60×2).

- Script: `scripts/model2_baseline_mlp_train_eval.py`
- Split: **scene-level** train/val split (20% of scenes held out)
- Metrics are reported on the **val** set (to avoid leakage across scenes)
- Outputs:
  - Metrics: `outputs/baselines/mlp_metrics_obs20_pred60.json`
  - Best checkpoint: `outputs/baselines/mlp_best_obs20_pred60.npz`
  - Val predictions: `outputs/baselines/mlp_pred_val_obs20_pred60.npz`

Results (seed=42, val_frac=0.2, hidden=256, depth=2):
- Train (N=3993): ADE = **2.5850**, FDE = **5.9669**
- Val   (N=828):  ADE = **2.6952**, FDE = **6.3653**

Run:
- `python scripts/model2_baseline_mlp_train_eval.py --input outputs/windows_obs20_pred60.npz`

## Multi-Agent Export (Trajectron++ Prep)

Trajectron++ requires **multi-agent, per-scene** data (multiple agents present at the same timesteps). Our `windows_obs20_pred60.npz` is **single-agent windows**, so we export a per-scene tensor format as an intermediate step.

- Script: `scripts/4_export_multiagent_trajpp_format.py`
- Output folder: `outputs/multiagent_trajpp_v1/`
  - `index.json`: list of all exported scenes + sizes
  - `schema.json`: array definitions
  - `scenes/<scene_id>.npz`: per-scene tensors

Per-scene NPZ arrays:
- `pos`: `(T, A, 2)` float32 (NaN where missing)
- `mask`: `(T, A)` bool
- `vel`: `(T, A, 2)` float32 (frame-to-frame delta; NaN where unavailable)
- `vel_mask`: `(T, A)` bool
- `frame_id`: `(T,)` int32/object
- `agent_id`: `(A,)` int32
- `agent_type`: `(A,)` object

Run:
- `python scripts/4_export_multiagent_trajpp_format.py --parsed-positions-dir outputs/parsed_positions --out-dir outputs/multiagent_trajpp_v1`

## Multi-Agent Sequences (ST-GAT Prep)

Many ST-GAT implementations use an ETH/UCY-style dataset format with variable numbers of agents per sequence and a `seq_start_end` index.

- Script: `scripts/5_export_stgat_format.py`
- Output: `outputs/stgat_dataset_obs20_pred60.npz`
- Stats: `outputs/stgat_dataset_obs20_pred60_stats.json`

This export uses the same `obs_len=20`, `pred_len=60`, `stride=3`, and keeps only agents that are present for *all* timesteps in each (obs+pred) window.

Run:
- `python scripts/5_export_stgat_format.py --scenes-dir outputs/multiagent_trajpp_v1/scenes --out outputs/stgat_dataset_obs20_pred60.npz`

## ST-GAT (Implemented)

We implemented a lightweight ST-GAT-inspired multi-agent model (PyTorch): LSTM encoder per agent + graph-attention pooling across agents + LSTM decoder.

- Script: `scripts/model3_stgat_train_eval.py`
- Train/val split: **scene-level** (20% of scenes held out)
- Outputs:
  - Metrics: `outputs/baselines/stgat_metrics_obs20_pred60.json`
  - Checkpoint: `outputs/baselines/stgat_ckpt_obs20_pred60.pt`
  - Val predictions: `outputs/baselines/stgat_pred_val_obs20_pred60.npz`

Results (25 epochs, seed=42):
- Val ADE = **2.3720**
- Val FDE = **5.8032**

GPU run (RTX 4090, CUDA, 120 epochs, batch=64):
- Metrics: `outputs/baselines/stgat_metrics_obs20_pred60_cuda.json`
- Val ADE = **2.3086**
- Val FDE = **5.5391**

Improved GPU run (RTX 4090, CUDA, simple pooling + AMP + LR plateau, early-stopped):
- Metrics: `outputs/baselines/stgat_metrics_obs20_pred60_cuda_simple_amp_v4.json`
- Checkpoint: `outputs/baselines/stgat_ckpt_obs20_pred60_cuda_simple_amp_v4.pt`
- Val predictions: `outputs/baselines/stgat_valpred_obs20_pred60_cuda_simple_amp_v4.npz`
- Best Val ADE = **1.8882** (best_epoch=46)
- Best Val FDE = **4.5308**

Multi-head attention run (RTX 4090, CUDA, `--pool mha` + AMP + LR plateau, **no early stopping**):
- Metrics: `outputs/baselines/stgat_metrics_obs20_pred60_cuda_mha_amp_noes_v5.json`
- Checkpoint: `outputs/baselines/stgat_ckpt_obs20_pred60_cuda_mha_amp_noes_v5.pt`
- Val predictions: `outputs/baselines/stgat_valpred_obs20_pred60_cuda_mha_amp_noes_v5.npz`
- Best Val ADE = **1.8674** (best_epoch=91)
- Best Val FDE = **4.4765**

Run:
- `python scripts/model3_stgat_train_eval.py --data outputs/stgat_dataset_obs20_pred60.npz`

Run on GPU (Python 3.12 venv):
- `./.venv312/Scripts/python.exe scripts/model3_stgat_train_eval.py --data outputs/stgat_dataset_obs20_pred60.npz --device cuda --epochs 120 --batch-size 64 \
  --save-metrics-json outputs/baselines/stgat_metrics_obs20_pred60_cuda.json \
  --save-ckpt outputs/baselines/stgat_ckpt_obs20_pred60_cuda.pt \
  --save-pred-npz outputs/baselines/stgat_pred_val_obs20_pred60_cuda.npz`

Improved run command (matches `*_cuda_simple_amp_v4.*` artifacts):
- `./.venv312/Scripts/python.exe scripts/model3_stgat_train_eval.py --data outputs/stgat_dataset_obs20_pred60.npz --device cuda --pool simple --amp --loss mse --epochs 300 --patience 40 --lr 0.001 --weight-decay 1e-5 --batch-size 128 --hidden-dim 256 --emb-dim 64 --attn-dim 128 --dropout 0.1 --lr-plateau --lr-plateau-factor 0.5 --lr-plateau-patience 10 --min-lr 1e-5 --tf-start 1.0 --tf-end 1.0 --tf-decay-epochs 1 --seed 0 \
  --save-metrics-json outputs/baselines/stgat_metrics_obs20_pred60_cuda_simple_amp_v4.json \
  --save-ckpt outputs/baselines/stgat_ckpt_obs20_pred60_cuda_simple_amp_v4.pt \
  --save-pred-npz outputs/baselines/stgat_valpred_obs20_pred60_cuda_simple_amp_v4.npz`

MHA run command with early stopping disabled (matches `*_cuda_mha_amp_noes_v5.*` artifacts):
- `./.venv312/Scripts/python.exe scripts/model3_stgat_train_eval.py --data outputs/stgat_dataset_obs20_pred60.npz --device cuda --pool mha --amp --loss mse --epochs 300 --patience 0 --lr 0.001 --weight-decay 1e-5 --batch-size 128 --hidden-dim 256 --emb-dim 64 --num-heads 4 --gat-layers 2 --dropout 0.1 --lr-plateau --lr-plateau-factor 0.5 --lr-plateau-patience 10 --min-lr 1e-5 --tf-start 1.0 --tf-end 1.0 --tf-decay-epochs 1 --seed 0 \
  --save-metrics-json outputs/baselines/stgat_metrics_obs20_pred60_cuda_mha_amp_noes_v5.json \
  --save-ckpt outputs/baselines/stgat_ckpt_obs20_pred60_cuda_mha_amp_noes_v5.pt \
  --save-pred-npz outputs/baselines/stgat_valpred_obs20_pred60_cuda_mha_amp_noes_v5.npz`

## MFP (Multi-Future Prediction)

Yes — our MFP baseline uses the **same ST-GAT-format dataset** produced by the ST-GAT export step:

- Input: `outputs/stgat_dataset_obs20_pred60.npz`
- Arrays used: `obs_traj_rel`, `pred_traj_rel`, plus `seq_start_end` to pack variable agent counts per sequence.

### What MFP predicts

MFP predicts **K candidate futures** per agent:

- `pred_rel_k`: `(B, M, K, 2, pred_len)`

Training uses **best-of-K (“variety”) loss**:

- for each agent in each batch, pick the mode with the smallest MSE to GT, and backprop only through that mode.

This typically produces strong **oracle best-of-K** metrics.

### Why “top-1” was originally bad

If we choose “top-1” as a fixed mode index (e.g. mode 0), it is essentially arbitrary, because variety loss does not enforce any consistent ordering of modes.

### Learned top-1 (mode scoring head)

To get a meaningful top-1, we added a **mode-scoring head** that predicts per-mode logits and selects:

- `top1_k = argmax(mode_logits)`

We train that head with an auxiliary cross-entropy loss to the oracle-best mode per agent:

- `loss = variety_loss + mode_cls_weight * CE(mode_logits, best_mode)`

This greatly improves top-1, but can slightly increase oracle best-of-K (it is a tradeoff).

### Run

Script:
- `scripts/model4_mfp_train_eval.py`

Example run (CUDA, K=10, no early stopping, top-1 via argmax logits, print every epoch):

- `./.venv312/Scripts/python.exe scripts/model4_mfp_train_eval.py \
  --data outputs/stgat_dataset_obs20_pred60.npz \
  --device cuda \
  --k 10 \
  --epochs 300 --patience 0 \
  --batch-size 64 \
  --mode-cls-weight 0.1 \
  --top1-by argmax \
  --log-every 1 \
  --save-metrics-json outputs/baselines/mfp_metrics_obs20_pred60_cuda_k10_noes_argmax_log1_v2.json \
  --save-ckpt outputs/baselines/mfp_ckpt_obs20_pred60_cuda_k10_noes_argmax_log1_v2.pt \
  --save-pred-npz outputs/baselines/mfp_pred_val_obs20_pred60_cuda_k10_noes_argmax_log1_v2.npz`

### Recent results (K=10)

Old “top-1” (fixed mode 0) example:
- Metrics: `outputs/baselines/4.1_mfp_metrics_obs20_pred60_cuda_k10_noes_top1.json`
- Oracle best-of-K: ADE **0.9709**, FDE **1.8919**
- Top-1 (mode 0): ADE **7.3259**, FDE **19.8294**

Learned top-1 (argmax logits) example:
- Metrics: `outputs/baselines/4.2_mfp_metrics_obs20_pred60_cuda_k10_noes_argmax_log1_v2.json`
- Oracle best-of-K: ADE **1.0539**, FDE **1.9235**
- Top-1 (argmax): ADE **2.1823**, FDE **4.8625**

Notes:
- To maximize oracle best-of-K, set `--mode-cls-weight 0` (but top-1 is less meaningful).
- To maximize top-1, increase `--mode-cls-weight` (may slightly reduce oracle performance).

## Notes on IDs

- `obj_type` is a category label (many agents share the same type).
- `obj_id` is the per-scene instance identifier (different `obj_id` usually means different physical agents).
- Some datasets can have rare ID switches; we do not automatically merge tracks.
