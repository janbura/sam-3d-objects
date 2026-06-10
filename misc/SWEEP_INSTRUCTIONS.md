# Combo Guidance Sweep — Instructions

We are running 3 composite guidance configs on the full Open3DHOI dataset (1709 instances),
split across 3 people to finish within the deadline.

## Who runs what

| Person  | Instance file          | Instances |
|---------|------------------------|-----------|
| Shidqie | `misc/sweep_user0.txt` | 570       |
| Jason   | `misc/sweep_user1.txt` | 570       |
| Lucas   | `misc/sweep_user2.txt` | 569       |

## Configs being run

| Config                 | What it combines              | Avg time/instance |
|------------------------|-------------------------------|-------------------|
| `pose_ss`              | Pose + Silhouette             | ~442s             |
| `pose_depth_normal`    | Pose + Depth + Normal         | ~369s             |
| `pose_ss_depth_normal` | Pose + Sil + Depth + Normal   | ~527s             |

## Time estimate

- 570 instances ÷ 4 shards = **143 instances per job**
- Slowest config (`pose_ss_depth_normal`): 143 × 527s ≈ **21h wall clock**
- All 12 jobs run in parallel — total wait is ~21h

## How to submit (replace `userX` with your assigned number)

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

## After your jobs finish

Rsync your results to Shidqie's sweep directory:

```bash
for cfg in pose_ss pose_depth_normal pose_ss_depth_normal; do
  rsync -av ~/sam-3d-objects/outputs/sweep/$cfg/ \
    scur0847@snellius.surf.nl:/gpfs/scratch1/shared/scur0847_sweep/$cfg/
done
```

Shidqie will then run the final evaluation once all three people are done.
