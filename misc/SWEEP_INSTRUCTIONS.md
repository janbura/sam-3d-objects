# Combo Guidance Sweep — Instructions

## Who runs what

| Person   | Instance file          | Instances |
|----------|------------------------|-----------|
| User 0   | `misc/sweep_user0.txt` | 570       |
| User 1   | `misc/sweep_user1.txt` | 570       |
| User 2   | `misc/sweep_user2.txt` | 569       |

## Time estimate

- 570 instances ÷ 4 shards = 143 instances per job
- Slowest config (`pose_ss_depth_normal`): 143 × 527s ≈ **21h wall clock**
- All 12 jobs run in parallel — total wait is ~21h

## Prerequisites

Before running, make sure you have:

1. **Conda environment** `sam3d-objects` set up with all dependencies
2. **Model checkpoints** at `checkpoints/hf/`
3. **Dataset** at `data/Open3DHOI/data/` — each instance needs `image.jpg`, `obj_mask.png`, and `depth.npy`

> If you don't have the dataset yet, copy or symlink it from the shared location:
> ```bash
> mkdir -p ~/sam-3d-objects/data/Open3DHOI
> ln -s /gpfs/scratch1/shared/<source_data_path> ~/sam-3d-objects/data/Open3DHOI/data
> ```

## Setup (one-time, before submitting)

Symlink your `outputs/sweep` to the shared scratch so all results land in one place automatically — no rsync needed at the end:

```bash
mkdir -p ~/sam-3d-objects/outputs

# Only run if outputs/sweep does not exist yet
ln -s /gpfs/scratch1/shared/scur0847_sweep ~/sam-3d-objects/outputs/sweep
```

> Note: the shared scratch already has write permissions open for you. New files you create will be owned by your account.

## How to submit (replace `userX` with your assigned number: 0, 1, or 2)

```bash
cd ~/sam-3d-objects

for cfg in pose_ss pose_depth_normal pose_ss_depth_normal; do
  for i in 0 1 2 3; do
    sbatch --export=ALL,CONFIGS=$cfg,SHARD=$i,NUM_SHARDS=4,INSTANCES=misc/sweep_userX.txt \
      jobs/13_sweep_combos.job
  done
done
```

This submits **12 jobs** (3 configs × 4 shards). Check they queued:

```bash
squeue -u $USER
```

## Notes

- Jobs have a 24h time limit — each shard takes ~21h for the slowest config, so do not reduce the time limit
- The sweep skips instances that already have outputs, so it is safe to resubmit if a job fails
- `pose_depth_normal` and `pose_ss_depth_normal` require `depth.npy` — instances without it are skipped automatically
- Logs land in `~/sam-3d-objects/logs/sam_sweep_combo_<jobid>.out`
