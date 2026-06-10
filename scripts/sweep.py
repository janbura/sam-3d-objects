"""
Scale sweep for Stage 1 guidance over a fixed set of samples.

Loads the model once, then runs every (sample, config) combination.
Outputs land in outputs/sweep/<config_name>/<cat>/<inst>/.

Usage:
    python sweep.py                              # hardcoded SAMPLES × CONFIGS
    python sweep.py --instances worst_30.txt     # file-driven instances × CONFIGS
    python sweep.py --instances worst_30.txt --configs ss_1.0 ss_5.0 ss5_pose005
    python sweep.py --tag hf --seed 42
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import torch
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, "notebook"))
from inference import Inference, load_image, load_mask
from main import build_guidance

# ── Hardcoded samples (used when --instances not provided) ───────────────────

DATA = "data/Open3DHOI/data"

SAMPLES = [
    {"name": "wrench", "dir": f"{DATA}/wrench/wrench"},
    {"name": "chair",  "dir": f"{DATA}/chair/2398495"},
]

# ── Sweep configs ─────────────────────────────────────────────────────────────

CONFIGS = {
    # baseline — no guidance
    "baseline":      {"ss": None,  "pose": None, "depth": None, "normal": None},
    # shape guidance only (sweet spot ~2-3 on dev wrench; ss>=10 destructive)
    "ss_0.1":        {"ss": 0.1,   "pose": None, "depth": None, "normal": None},
    "ss_0.5":        {"ss": 0.5,   "pose": None, "depth": None, "normal": None},
    "ss_1.0":        {"ss": 1.0,   "pose": None, "depth": None, "normal": None},
    "ss_2.0":        {"ss": 2.0,   "pose": None, "depth": None, "normal": None},
    "ss_3.0":        {"ss": 3.0,   "pose": None, "depth": None, "normal": None},
    "ss_5.0":        {"ss": 5.0,   "pose": None, "depth": None, "normal": None},
    "ss_10.0":       {"ss": 10.0,  "pose": None, "depth": None, "normal": None},
    # pose guidance only (no effect on normalized+ICP CD — eval via silhouette IoU)
    # smaller range than ss/depth/normal: pose latents are low-dim, same scale = larger per-element step
    "pose_0.01":     {"ss": None,  "pose": 0.01, "depth": None, "normal": None},
    "pose_0.05":     {"ss": None,  "pose": 0.05, "depth": None, "normal": None},
    "pose_0.1":      {"ss": None,  "pose": 0.1,  "depth": None, "normal": None},
    "pose_0.2":      {"ss": None,  "pose": 0.2,  "depth": None, "normal": None},
    # depth guidance only
    "depth_1.0":     {"ss": None,  "pose": None, "depth": 1.0,  "normal": None},
    "depth_2.0":     {"ss": None,  "pose": None, "depth": 2.0,  "normal": None},
    "depth_5.0":     {"ss": None,  "pose": None, "depth": 5.0,  "normal": None},
    "depth_10.0":    {"ss": None,  "pose": None, "depth": 10.0, "normal": None},
    # normal guidance only
    "normal_1.0":    {"ss": None,  "pose": None, "depth": None, "normal": 1.0},
    "normal_2.0":    {"ss": None,  "pose": None, "depth": None, "normal": 2.0},
    "normal_5.0":    {"ss": None,  "pose": None, "depth": None, "normal": 5.0},
    "normal_10.0":   {"ss": None,  "pose": None, "depth": None, "normal": 10.0},
    # combos
    "ss5_pose005":          {"ss": 5.0,  "pose": 0.05, "depth": None, "normal": None},
    "ss5_depth5":           {"ss": 5.0,  "pose": None, "depth": 5.0,  "normal": None},
    # pose-anchored combos (best single-guidance scales)
    "pose_ss":              {"ss": 1.0,  "pose": 0.01, "depth": None, "normal": None},
    "pose_depth_normal":    {"ss": None, "pose": 0.01, "depth": 1.0,  "normal": 1.0},
    "pose_ss_depth_normal": {"ss": 1.0,  "pose": 0.01, "depth": 1.0,  "normal": 1.0},
}

DEFAULT_CONFIGS = [
    "depth_1.0", "depth_5.0", "depth_10.0",
    "normal_1.0", "normal_5.0", "normal_10.0",
    "ss_1.0", "ss_5.0", "ss_10.0",
    "pose_0.01", "pose_0.05", "pose_0.2",
]

# ─────────────────────────────────────────────────────────────────────────────


def load_instances(path):
    """Read cat/inst lines from a text file, return list of sample dicts."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("/", 1)
            if len(parts) == 2:
                cat, inst = parts
                samples.append({
                    "name": f"{cat}/{inst}",
                    "cat":  cat,
                    "inst": inst,
                    "dir":  os.path.join(DATA, cat, inst),
                })
    return samples


def run_sweep(tag: str, seed: int, samples, config_names: list, save_steps: bool = False):
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    print(f"Loading model from {config_path} ...")
    inference = Inference(config_path, compile=False)
    print("Model loaded.\n")

    configs = [(name, CONFIGS[name]) for name in config_names]
    total = len(samples) * len(configs)
    done = 0

    # family -> list of elapsed seconds (skipped runs excluded)
    family_times = defaultdict(list)

    for cfg_name, cfg in configs:
        family = cfg_name.split("_")[0]
        for sample in samples:
            done += 1
            cat  = sample.get("cat",  sample["name"])
            inst = sample.get("inst", sample["name"])
            out_dir = os.path.join("outputs", "sweep", cfg_name, cat, inst)

            pred_mesh_path   = os.path.join(out_dir, "pred_mesh.obj")
            pose_params_path = os.path.join(out_dir, "pose_params.pt")
            final_step_path  = os.path.join(out_dir, "final_step.pt")

            if (os.path.exists(pred_mesh_path) and os.path.exists(pose_params_path)
                    and os.path.exists(final_step_path)):
                print(f"[{done}/{total}] SKIP {cfg_name} / {sample['name']} (exists)")
                continue

            print(f"[{done}/{total}] RUN  {cfg_name} / {sample['name']}")

            inst_dir = sample["dir"]
            image = load_image(os.path.join(inst_dir, "image.jpg"))
            mask  = load_mask(os.path.join(inst_dir, "obj_mask.png"))
            depth_map = os.path.join(inst_dir, "depth.npy") if (cfg["depth"] or cfg["normal"]) else None
            if depth_map and not os.path.exists(depth_map):
                print(f"[{done}/{total}] SKIP {cfg_name} / {sample['name']} (no depth.npy)")
                continue

            prefix = os.path.join("sweep", cfg_name, cat, inst) if save_steps else None

            guidance = build_guidance(
                mask_path=os.path.join(inst_dir, "obj_mask.png"),
                ss_guidance_scale=cfg["ss"],
                pose_guidance_scale=cfg["pose"],
                depth_guidance_scale=cfg["depth"],
                normal_guidance_scale=cfg["normal"],
                depth_map=depth_map,
                steps_prefix=prefix,
                w_centroid=1.0,
                w_size=1.0,
            )

            t0 = time.time()
            output = inference(image, mask, seed=seed, steps_prefix=prefix, guidance=guidance)
            elapsed = time.time() - t0
            family_times[family].append(elapsed)
            print(f"         → {elapsed:.1f}s")

            os.makedirs(out_dir, exist_ok=True)

            if "mesh" in output:
                mesh = output["mesh"][0]
                if mesh.success:
                    tm = trimesh.Trimesh(
                        vertices=mesh.vertices.cpu().numpy(),
                        faces=mesh.faces.cpu().numpy(),
                    )
                    tm.export(pred_mesh_path)

            pose_params = {
                k: output[k].cpu() if hasattr(output.get(k), "cpu") else output.get(k)
                for k in ("rotation", "translation", "scale", "intrinsics")
                if k in output
            }
            if pose_params:
                torch.save(pose_params, pose_params_path)

            if "ss_grid" in output:
                torch.save({"ss_grid": output["ss_grid"]}, final_step_path)

            # Grad norm stats
            if hasattr(guidance, "get_stats"):
                stats = guidance.get_stats()
                torch.save(stats, os.path.join(out_dir, "grad_stats.pt"))
            print(f"         → saved to {out_dir}")

    print("\nSweep complete.")

    print("\n=== Timing summary (actual runs only, skips excluded) ===")
    total_all = 0.0
    for fam in sorted(family_times):
        ts = family_times[fam]
        avg = sum(ts) / len(ts)
        total_fam = sum(ts)
        total_all += total_fam
        print(f"  {fam:<8}  n={len(ts):>4}  avg={avg:>6.1f}s  total={total_fam/3600:.2f}h")
    print(f"  {'TOTAL':<8}  {'':>10}  {'':>14}  {total_all/3600:.2f}h")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",       default="hf")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--instances", default=None,
                        help="text file with cat/instance lines; uses hardcoded SAMPLES if omitted")
    parser.add_argument("--configs",    nargs="+", default=DEFAULT_CONFIGS,
                        help=f"config names to run (default: {DEFAULT_CONFIGS}); "
                             f"available: {sorted(CONFIGS)}")
    parser.add_argument("--save-steps", action="store_true",
                        help="save per-diffusion-step .pt files (slow; off by default)")
    parser.add_argument("--shard",      type=int, default=0,
                        help="0-indexed shard index for this job")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="total number of parallel shards (default 1 = no sharding)")
    args = parser.parse_args()

    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        parser.error(f"Unknown config names: {unknown}. Available: {sorted(CONFIGS)}")

    if args.shard >= args.num_shards:
        parser.error(f"--shard {args.shard} out of range for --num-shards {args.num_shards}")

    samples = load_instances(args.instances) if args.instances else SAMPLES
    if args.num_shards > 1:
        samples = samples[args.shard::args.num_shards]
        print(f"Shard {args.shard}/{args.num_shards}: {len(samples)} instances")
    print(f"Instances: {len(samples)}  Configs: {args.configs}  save_steps={args.save_steps}\n")

    run_sweep(args.tag, args.seed, samples, args.configs, save_steps=args.save_steps)


if __name__ == "__main__":
    main()
