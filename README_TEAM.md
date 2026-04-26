# SAM 3D — Team Setup Guide for Snellius

**Status:** ✅ Validated end-to-end on Snellius 2026-04-26 (NVIDIA A100, CUDA 12.1, PyTorch 2.5.1)

This guide gets SAM 3D inference running on the [Snellius supercomputer](https://www.surf.nl/en/services/snellius-the-national-supercomputer) using SLURM batch jobs. It's a thin wrapper around the upstream [doc/setup.md](doc/setup.md) — the differences are partition choices, scratch-vs-home placement, and a few hard-won bug fixes.

If you're not on Snellius, follow `doc/setup.md` directly; this README probably doesn't apply.

---

## Quick start


```bash
# 1. Clone this fork on Snellius (NOT upstream)
git clone git@github.com:janbura/sam-3d-objects.git
cd sam-3d-objects
git remote add upstream https://github.com/facebookresearch/sam-3d-objects.git

# 2. Make personal config
cp paths.env.example paths.env  # or write your own; see below
echo "hf_YOUR_TOKEN_HERE" > ~/.hf_token && chmod 600 ~/.hf_token

# 3. Make working directories
mkdir -p logs /scratch-shared/$USER/conda_envs /scratch-shared/$USER/sam3d_checkpoints

# 4. Run the four jobs in order — wait for each to succeed before the next
sbatch jobs/01_setup_cpu.job             # ~30-60 min on rome
sbatch jobs/02_download_checkpoints.job  # ~1-30 min on rome (HF bandwidth dependent)
sbatch jobs/03_install_inference_gpu.job # ~5-20 min on gpu_mig
sbatch jobs/04_run_inference.job         # ~7 min on gpu_a100
```

Total: roughly 1–2 hours of wall time, mostly SLURM queue + waiting. Active hands-on: ~10 min.

---

## Prerequisites

Each user needs:

1. **A Snellius account with a working home directory.** Test with `cd ~ && ls -la`. If you get `Permission denied`, the home directory has wrong ownership and only [SURF support](mailto:servicedesk@surf.nl) can fix it. Don't proceed until this works.
2. **A GitHub account with collaborator access** to this fork. Ask the fork owner.
3. **An SSH key registered with GitHub from Snellius.** `ssh-keygen -t ed25519 -C "snellius-github" -f ~/.ssh/id_ed25519` (no passphrase), then add `~/.ssh/id_ed25519.pub` at https://github.com/settings/keys. Test with `ssh -T git@github.com`.
4. **A HuggingFace account.** Free at https://huggingface.co/.
5. **Acceptance of the SAM 3D license** at https://huggingface.co/facebook/sam-3d-objects (the model is gated). Click "Agree and access repository."
6. **A personal HF read token** at https://huggingface.co/settings/tokens. Read access only. Save in `~/.hf_token` on Snellius with `chmod 600`. **Never share, never commit, never paste in commands** — CLI errors echo argv and tokens leak that way.

---

## Storage layout

The conda env (~25 GB) and checkpoints (~12 GB) are too big for `$HOME`. Use scratch:

| What | Where | Why |
|---|---|---|
| Repo clone | `$HOME/sam-3d-objects` | Small, version-controlled |
| Conda env | `/scratch-shared/$USER/conda_envs/sam3d-objects` | Big, regenerable |
| Checkpoints | `/scratch-shared/$USER/sam3d_checkpoints` | Big, slow to re-download |
| Job logs | `$HOME/sam-3d-objects/logs/` | Small, useful to keep |
| HF token | `$HOME/.hf_token` (mode 600) | Per-user secret |

The job files reference these paths via `paths.env`, which you create per-user (it's gitignored). Template:

```bash
# paths.env — per-user, DO NOT COMMIT
export SAM3D_REPO="$HOME/sam-3d-objects"
export SAM3D_CONDA_ENVS="/scratch-shared/$USER/conda_envs"
export SAM3D_ENV_NAME="sam3d-objects"
export SAM3D_CHECKPOINTS="/scratch-shared/$USER/sam3d_checkpoints"
```

If your team has a shared project space, point `SAM3D_CHECKPOINTS` there to avoid downloading the same 12 GB six times.

---

## SLURM partition strategy

Snellius bills per-partition. We use the cheapest partition that can do each job:

| Job | Partition | VRAM | Why |
|---|---|---|---|
| `01_setup_cpu` | `rome` (CPU) | n/a | No GPU needed for env build + non-CUDA pip installs |
| `02_download_checkpoints` | `rome` (CPU) | n/a | Pure I/O |
| `03_install_inference_gpu` | `gpu_mig` | ~20 GB | Compile needs CUDA, but full A100 is overkill |
| `04_run_inference` | `gpu_a100` | 40 GB | **Required: ≥32 GB VRAM per upstream `doc/setup.md`** |

`gpu_mig` only has 4 nodes total — if all are busy, switch to `gpu_a100`. If `gpu_a100` is contested, `gpu_h100` (84 nodes, 80 GB VRAM) is also fine and often less busy.

---

## What each job does

### `01_setup_cpu.job` (rome, ~30–60 min)

- Loads `Anaconda3/2023.07-2`
- Creates conda env at `${SAM3D_CONDA_ENVS}/${SAM3D_ENV_NAME}` from `environments/default.yml`
- Runs `pip install -e '.[dev]'` then `pip install -e '.[p3d]'` with the upstream `PIP_EXTRA_INDEX_URL` (NVIDIA NGC + PyTorch CUDA 12.1)
- Verifies torch loads with `2.5.1+cu121`

The env-creation step is idempotent — if the env directory already exists, it skips. So a re-run after a failed pip install only takes a few minutes.

### `02_download_checkpoints.job` (rome, ~1–30 min)

- Activates the env from job 1
- Reads HF token from `~/.hf_token`
- Runs `hf download facebook/sam-3d-objects` to `${SAM3D_CHECKPOINTS}/hf-download`
- Renames to `${SAM3D_CHECKPOINTS}/hf` and symlinks it into `${SAM3D_REPO}/checkpoints/hf` so `demo.py` finds it without modification

You'll see ~12 GB of `.ckpt` files. Note: two files (`ss_encoder.safetensors` and `ss_encoder.yaml`) are 0 bytes — this is intentional placeholder content from the upstream HF repo, not a download bug. The inference pipeline doesn't reference them.

### `03_install_inference_gpu.job` (gpu_mig, ~5–20 min)

- Activates env on a MIG slice (so CUDA is visible for any compile steps)
- Runs `pip install -e '.[inference]'` with `PIP_FIND_LINKS` pointing at kaolin wheels for `torch-2.5.1_cu121` (note: NOT `torch-2.3.0`, which an outdated guide may suggest)
- Runs `./patching/hydra` (a small upstream-supplied patch script)
- Smoke-tests imports: `pytorch3d`, `kaolin`, `hydra`, `flash_attn`

If `flash_attn` was already pulled in by `[dev]` deps, you'll skip the slow source compile. Otherwise this job can take an additional 10–20 min.

### `04_run_inference.job` (gpu_a100, ~7 min)

- Activates env on a full A100 (40 GB VRAM)
- Runs `python demo.py` twice — both runs use seed 42 on the same bundled kidsroom test image
- Compares the two `splat.ply` outputs by md5sum to check determinism
- Produces `outputs/run_a/splat.ply` and `outputs/run_b/splat.ply` (~55 MB each)

First run is slower (~4 min) due to one-time DINOv2 model download (~1.1 GB) and `torch.compile` JIT pass. Second run is ~3 min.

**Note on determinism:** in our validation runs, the two same-seed outputs had different md5sums. This is likely due to `torch.compile` non-determinism or cuDNN benchmark mode rather than a true RNG issue, and is being investigated. For ablation studies, plan to use a tolerance-based comparison rather than bit-exactness.

---

## How to view the output

`splat.ply` is a Gaussian splat in the standard `.ply` format with extra attributes (50+ per vertex). Most generic mesh viewers will show it as an unrendered point cloud or fail outright. Best options:

- **[SuperSplat](https://playcanvas.com/supersplat/editor)** (web, no install) — drag-and-drop, renders Gaussian splats correctly. Easiest.
- **MeshLab** — `brew install --cask meshlab` (Mac). Treats it as a point cloud, useful for technical inspection.
- **Blender 4.2+** has Gaussian splat support via add-ons, but it's more involved.

For an end-to-end smoke test, opening `outputs/run_a/splat.ply` in SuperSplat should show a recognizable kidsroom scene with bed, chairs, etc.

---

## Common pitfalls (read before submitting jobs)

These are issues we hit during validation. The job files in this repo already account for them, but if you adapt the job files for your own use, watch out:

1. **Don't use `set -euo pipefail` — use `set -eo pipefail`.** The `-u` flag (unbound variables) collides with conda's `binutils` activation script, which references `$ADDR2LINE` unbound. Job 1 fails silently with a one-line error otherwise.

2. **Don't source `paths.env` via `$(dirname "$0")`.** SLURM copies job scripts to `/var/spool/slurm/slurmd/<jobid>/` before running, so `$0` doesn't resolve to your repo path. Use absolute paths, e.g. `source "$HOME/sam-3d-objects/paths.env"`.

3. **Activate conda env by full prefix path, not by name.** The env was created with `conda env create -p <prefix>`, so it's not in conda's named env list. `conda activate sam3d-objects` will silently fail; use `conda activate "${SAM3D_CONDA_ENVS}/${SAM3D_ENV_NAME}"`.

4. **`module load 2023` before `Anaconda3/2023.07-2`.** Snellius's lmod requires the toolchain year first.

5. **Never pass HF tokens on the command line.** CLI error messages echo argv. Always read from a file: `export HF_TOKEN=$(cat ~/.hf_token)`.

---

## Validated environment

Confirmed working as of 2026-04-26:

| Component | Version |
|---|---|
| OS | RHEL 8 (Snellius login + compute) |
| Module | `2023` toolchain |
| Conda | `Anaconda3/2023.07-2` |
| Python | 3.11 (from conda env) |
| PyTorch | 2.5.1+cu121 |
| CUDA toolkit | 12.1 (bundled in conda env via conda-forge) |
| GPU driver | 590.48.01 (CUDA Version 13.1, backwards-compat with 12.1) |
| pytorch3d | 0.7.8 |
| kaolin | 0.17.0 |
| hydra-core | 1.3.2 |
| flash_attn | (latest matching torch 2.5.1) |

GPU tested on: `NVIDIA A100-SXM4-40GB` (full) and `NVIDIA A100-SXM4-40GB MIG 3g.20gb` (slice).

---

## Open caveats

- **Same-seed inference is not bit-deterministic** in this validated configuration. Use tolerance-based comparison for ablation experiments.
- **`pipeline.yaml` triggers extra HF downloads on first inference run** (MoGe depth model + DINOv2 weights, ~1.7 GB total cached to `~/.cache/torch/hub/`). Make sure your HF token is in place before the first `04_run_inference.job`.
- **Inference uses fp16 by default** (`dtype: float16` in `checkpoints/hf/pipeline.yaml`). Required to fit in 40 GB VRAM.

---

## Contributing back

This fork is for our team's project work; if you have improvements to the setup itself worth proposing upstream, open a PR against [`facebookresearch/sam-3d-objects`](https://github.com/facebookresearch/sam-3d-objects).

For team-internal work: branch off `main`, push to this fork, open a PR against `main`. Don't push to `upstream`.

---

## Acknowledgements

Setup path validated by Jan Burakowski (UvA) on 2026-04-26 against [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) by Meta Superintelligence Labs.