# SAM 3D Objects — Guidance Evaluation

This README covers the full workflow for running and evaluating Stage 1 guidance on [Open3DHOI](https://github.com/leolyliu/Open3DHOI).
For the original SAM 3D Objects model documentation, see [README_ORIGINAL.md](README_ORIGINAL.md).
For technical details on how guidance works, see [GUIDANCE.md](GUIDANCE.md).

---

## 1. Setup

Follow the original setup in [doc/setup.md](doc/setup.md) to install the environment and download checkpoints.

### Data — Open3DHOI

```bash
# Download Open3DHOI dataset and place under data/
# Expected structure:
# data/Open3DHOI/data/{category}/{instance}/
#   image.jpg
#   obj_mask.png
#   depth.npy          # not all instances have this — depth/normal guidance skips those
#   gt_mesh.obj
```

### Snellius

```bash
# Outputs are symlinked to scratch for storage
ln -s /gpfs/scratch1/shared/scur0847_sweep         outputs/sweep
ln -s /gpfs/scratch1/shared/scur0847_sam3d_baseline outputs/baseline_all
```

---

## 2. Baseline (no guidance)

Run inference on all 1709 Open3DHOI instances without any guidance signal.

```bash
# Single job — runs all instances sequentially
sbatch jobs/05_baseline_all.job

# Check results
tail -f logs/sam_baseline_all_<JOBID>.out
```

Output: `outputs/baseline_all/{category}/{instance}/pred_mesh.obj`

---

## 3. Eval Baseline

Compute Chamfer Distance, F-score, and IoU for baseline results.

```bash
sbatch jobs/06_eval_baseline_all.job
```

Output: `outputs/baseline_all/results_multi_init.csv`

---

## 4. Guidance Sweep

Run inference with guidance signals on all instances. Uses sharded jobs for parallelism.

### Instance lists

| File | Description |
|------|-------------|
| `misc/all_1709.txt` | Full 1709-instance set |
| `misc/eval_855.txt` | Stratified 50% subset (main eval) |
| `misc/pose_428.txt` | Stratified 25% subset (pose eval) |
| `misc/tune_85.txt` | Stratified 5% subset (scale tuning) |

### Available configs

| Config | Guidance | Scale |
|--------|----------|-------|
| `ss_1.0` | Silhouette | 1.0 |
| `depth_1.0` | Depth | 1.0 |
| `normal_1.0` | Normal | 1.0 |
| `pose_0.01` | Pose | 0.01 |

### Running a sweep

```bash
# Sharded sweep — adjust SHARD, NUM_SHARDS, CONFIGS, INSTANCES as needed

# Silhouette (4 shards, ~21h total)
for i in 0 1 2 3; do
  sbatch --time=24:00:00 \
    --export=ALL,SHARD=$i,NUM_SHARDS=4,CONFIGS="ss_1.0",INSTANCES="misc/all_1709.txt" \
    jobs/09_sweep_final.job
done

# Depth (2 shards, ~12h total)
for i in 0 1; do
  sbatch --time=12:00:00 \
    --export=ALL,SHARD=$i,NUM_SHARDS=2,CONFIGS="depth_1.0",INSTANCES="misc/all_1709.txt" \
    jobs/09_sweep_final.job
done

# Normal (2 shards, ~12h total)
for i in 0 1; do
  sbatch --time=12:00:00 \
    --export=ALL,SHARD=$i,NUM_SHARDS=2,CONFIGS="normal_1.0",INSTANCES="misc/all_1709.txt" \
    jobs/09_sweep_final.job
done

# Pose (6 shards, ~24h total — evaluated on stratified 50%)
for i in 0 1 2 3 4 5; do
  sbatch --time=24:00:00 \
    --export=ALL,SHARD=$i,NUM_SHARDS=6,CONFIGS="pose_0.01",INSTANCES="misc/eval_855.txt" \
    jobs/09_sweep_final.job
done
```

Each instance output is saved to `outputs/sweep/{config}/{category}/{instance}/`:
- `pred_mesh.obj` — reconstructed mesh
- `final_step.pt` — occupancy grid from last denoising step (used for IoU)
- `pose_params.pt` — predicted pose parameters
- `grad_stats.pt` — gradient norms per timestep (for analysis)

> **Note:** instances without `depth.npy` are automatically skipped for `depth_1.0` and `normal_1.0` (109 instances in Open3DHOI).

---

## 5. Eval Sweep

Evaluate all guidance configs against baseline. Runs baseline + sweep eval in one job:

```bash
sbatch jobs/11_eval_all.job
```

Or separately:

```bash
# Sweep only (ss/depth/normal on full dataset, pose on stratified 50%)
sbatch jobs/08_eval_sweep.job
```

Output: `outputs/sweep/results.csv`

### Results table format

```
config,n,cd_mean,f_score_mean,iou_mean
baseline,...
ss_1.0,...
depth_1.0,...
normal_1.0,...
pose_0.01,...
```

---

## 6. Grad Norm Visualization

Plot gradient norm vs diffusion timestep per guidance family:

```bash
sbatch jobs/10_viz_grad_norm.job
```

Output: `outputs/viz_grad_norm/grad_norm_vs_timestep.png`

---

## Timing reference (H100)

| Config | Time/instance | Notes |
|--------|-------------|-------|
| Baseline | ~45s | |
| Silhouette | ~174s | |
| Depth / Normal | ~59s | skips instances without depth.npy |
| Pose | ~600s | 2 backward passes |
