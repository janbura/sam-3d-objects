"""
Visualize gradient norm vs diffusion timestep per guidance family.
Aggregates grad_stats.pt across instances, plots mean ± std per timestep.

Usage:
    python scripts/viz_grad_norm.py
    python scripts/viz_grad_norm.py --configs depth_1.0 normal_1.0 ss_1.0 pose_0.01
    python scripts/viz_grad_norm.py --max-instances 200
"""
import argparse
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(ROOT, "outputs", "sweep")
OUT   = os.path.join(ROOT, "outputs", "viz_grad_norm")

# class name → display label + color
CLASS_META = {
    "ShapeGuidance":  ("Silhouette",  "tab:blue"),
    "DepthGuidance":  ("Depth",       "tab:orange"),
    "NormalGuidance": ("Normal",      "tab:green"),
    "PoseGuidance":   ("Pose",        "tab:red"),
}

DEFAULT_CONFIGS = ["depth_1.0", "normal_1.0", "ss_1.0", "pose_0.01"]


def load_stats(sweep_dir, configs, max_instances):
    # class_name -> {t_step -> [grad_norm, ...]}
    data = defaultdict(lambda: defaultdict(list))

    for cfg in configs:
        cfg_dir = os.path.join(sweep_dir, cfg)
        if not os.path.isdir(cfg_dir):
            print(f"  [skip] {cfg} not found")
            continue
        files = []
        for cat in os.listdir(cfg_dir):
            cat_dir = os.path.join(cfg_dir, cat)
            if not os.path.isdir(cat_dir):
                continue
            for inst in os.listdir(cat_dir):
                p = os.path.join(cat_dir, inst, "grad_stats.pt")
                if os.path.exists(p):
                    files.append(p)
        if max_instances:
            files = files[:max_instances]
        print(f"  {cfg}: {len(files)} instances with grad_stats")

        for path in files:
            try:
                stats = torch.load(path, weights_only=False, map_location="cpu")
            except Exception:
                continue
            for cls_name, step_stats in stats.items():
                for entry in step_stats:
                    t = entry[0]
                    norm = entry[1]  # trans norm for pose, grad norm for others
                    data[cls_name][round(t, 4)].append(norm)

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir",     default=SWEEP)
    parser.add_argument("--configs",       nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--log-scale",     action="store_true", default=True)
    parser.add_argument("--out",           default=OUT)
    args = parser.parse_args()

    print("Loading grad stats...")
    data = load_stats(args.sweep_dir, args.configs, args.max_instances)

    if not data:
        print("No grad_stats found.")
        return

    os.makedirs(args.out, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for cls_name, t_dict in sorted(data.items()):
        label, color = CLASS_META.get(cls_name, (cls_name, None))
        ts    = sorted(t_dict.keys())
        means = [np.mean(t_dict[t]) for t in ts]
        stds  = [np.std(t_dict[t])  for t in ts]
        ts    = np.array(ts)
        means = np.array(means)
        stds  = np.array(stds)

        kwargs = dict(color=color) if color else {}
        ax.plot(ts, means, label=label, linewidth=2, **kwargs)
        ax.fill_between(ts, means - stds, means + stds, alpha=0.15, **kwargs)

    ax.set_xlabel("Flow time $t$  (0 = noise, 1 = data)", fontsize=12)
    ax.set_ylabel("Gradient norm", fontsize=12)
    ax.set_title("Guidance gradient norm over denoising", fontsize=13)
    if args.log_scale:
        ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(args.out, "grad_norm_vs_timestep.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")

    # Also save per-family summary
    print("\nPeak timestep per family:")
    for cls_name, t_dict in sorted(data.items()):
        label, _ = CLASS_META.get(cls_name, (cls_name, None))
        ts    = sorted(t_dict.keys())
        means = [np.mean(t_dict[t]) for t in ts]
        peak_t = ts[np.argmax(means)]
        print(f"  {label}: peak at t={peak_t:.3f}  max_mean={max(means):.4f}")


if __name__ == "__main__":
    main()
