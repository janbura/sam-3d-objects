"""
Visualize silhouette overlay comparison across pose guidance scales.

Grid: rows=instances, cols=[Image, GT, baseline, pose_0.01, pose_0.05, pose_0.2]
+ color legend strip at the bottom.

Usage:
    python scripts/viz_pose_comparison.py
    python scripts/viz_pose_comparison.py --instances "teddy bear/HICO_train2015_00003818"
    python scripts/viz_pose_comparison.py --n 5 --out outputs/viz_pose_cmp.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from guidance import _extract_mesh, _render_soft_silhouette

DATA_ROOT = os.path.join(ROOT, "data", "Open3DHOI", "data")
BASELINE  = os.path.join(ROOT, "outputs", "baseline_all")
SWEEP     = os.path.join(ROOT, "outputs", "sweep")

# (display label, scale annotation, output dir)
CONFIGS = [
    ("Baseline",   r"$\eta=0$",    BASELINE),
    ("pose 0.01",  r"$\eta=0.01$", os.path.join(SWEEP, "pose_0.01")),
    ("pose 0.05",  r"$\eta=0.05$", os.path.join(SWEEP, "pose_0.05")),
    ("pose 0.20",  r"$\eta=0.20$", os.path.join(SWEEP, "pose_0.2")),
]

# Default sweet-spot instances (pose_0.01 best, higher scales degrade)
DEFAULT_INSTANCES = [
    "teddy bear/HICO_train2015_00003818",
    "sports ball/73ed398151cda1a4",
    "cake/COCO_val2014_000000403975",
]

RENDER_SIZE = 256


def render_sil(out_dir, device="cpu"):
    final_step_path = os.path.join(out_dir, "final_step.pt")
    pose_path       = os.path.join(out_dir, "pose_params.pt")
    if not os.path.exists(final_step_path) or not os.path.exists(pose_path):
        return None, None
    data = torch.load(final_step_path, weights_only=False, map_location="cpu")
    pose = torch.load(pose_path,       weights_only=False, map_location="cpu")
    result = _extract_mesh(data["ss_grid"], device)
    if result is None:
        return None, None
    verts, faces = result
    sil = _render_soft_silhouette(
        verts.to(device), faces.to(device),
        pose["rotation"], pose["translation"], pose["scale"], pose["intrinsics"],
        RENDER_SIZE, device,
    )
    sil_np = sil.squeeze().cpu().numpy()
    bin_np = (sil_np > 0.1).astype(np.float32)
    return sil_np, bin_np


def compute_iou(pred_bin, gt_bin):
    inter = (pred_bin * gt_bin).sum()
    union = ((pred_bin + gt_bin) > 0).sum()
    return float(inter) / float(union + 1e-8)


def make_overlay(pred_bin, gt_bin):
    H, W = gt_bin.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    overlap  = (pred_bin > 0) & (gt_bin > 0)
    pred_only = (pred_bin > 0) & ~overlap
    gt_only   = (gt_bin  > 0) & ~overlap
    rgb[overlap]   = [1.0, 1.0, 0.0]   # yellow  = overlap (TP)
    rgb[pred_only] = [1.0, 0.0, 0.0]   # red     = pred only (FP)
    rgb[gt_only]   = [0.0, 0.85, 0.0]  # green   = GT only (FN)
    return rgb


def add_legend(fig):
    legend_elements = [
        mpatches.Patch(facecolor=(1.0, 1.0, 0.0), edgecolor="gray", label="Overlap (TP)"),
        mpatches.Patch(facecolor=(1.0, 0.0, 0.0), edgecolor="gray", label="Pred only (FP)"),
        mpatches.Patch(facecolor=(0.0, 0.85, 0.0), edgecolor="gray", label="GT only (FN)"),
        mpatches.Patch(facecolor=(0.0, 0.0, 0.0), edgecolor="gray", label="Background"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
        title="Silhouette overlay",
        title_fontsize=9,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="*", default=None,
                        help="cat/name pairs; defaults to sweet-spot set")
    parser.add_argument("--n",   type=int, default=None,
                        help="take first N from default list (ignored if --instances given)")
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "viz_pose_cmp.png"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.instances:
        instances = args.instances
    else:
        instances = DEFAULT_INSTANCES[:args.n] if args.n else DEFAULT_INSTANCES
        print(f"Using default sweet-spot instances: {len(instances)}")

    n_rows = len(instances)
    n_cols = 2 + len(CONFIGS)  # Image + GT + one per config

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 3.0 * n_rows + 0.6),  # +0.6 for legend
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Column headers: two-line (name + scale)
    col_headers = [("Image", ""), ("GT mask", "")]
    for label, scale, _ in CONFIGS:
        col_headers.append((label, scale))

    for j, (title, scale) in enumerate(col_headers):
        header = f"{title}\n{scale}" if scale else title
        axes[0, j].set_title(header, fontsize=9, fontweight="bold", linespacing=1.4)

    for i, inst in enumerate(instances):
        cat, name = inst.split("/", 1)
        short_name = cat.replace(" ", "\n")

        # --- Image ---
        img_path = os.path.join(DATA_ROOT, cat, name, "image.jpg")
        if os.path.exists(img_path):
            axes[i, 0].imshow(np.array(Image.open(img_path)))
        axes[i, 0].axis("off")
        axes[i, 0].set_ylabel(short_name, fontsize=8, rotation=0,
                               labelpad=50, va="center", ha="right")

        # --- GT mask ---
        mask_path = os.path.join(DATA_ROOT, cat, name, "obj_mask.png")
        gt_bin = None
        if os.path.exists(mask_path):
            gt = np.array(Image.open(mask_path).convert("L"))
            gt_bin = (gt > 0).astype(np.float32)
            axes[i, 1].imshow(gt_bin, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].axis("off")

        # --- Each config ---
        for j, (label, scale, base) in enumerate(CONFIGS):
            ax = axes[i, 2 + j]
            out_dir = os.path.join(base, cat, name)

            if not os.path.exists(os.path.join(out_dir, "final_step.pt")):
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
                ax.axis("off")
                continue

            _, pred_bin = render_sil(out_dir, device=args.device)
            if pred_bin is None or gt_bin is None:
                ax.axis("off")
                continue

            if pred_bin.shape != gt_bin.shape:
                pred_bin = np.array(
                    Image.fromarray(pred_bin).resize(
                        (gt_bin.shape[1], gt_bin.shape[0]), Image.NEAREST
                    )
                )

            iou = compute_iou(pred_bin, gt_bin)
            overlay = make_overlay(pred_bin, gt_bin)
            ax.imshow(overlay)
            # IoU as overlay text (bottom-centre) — keeps set_title for column header
            ax.text(0.5, 0.03, f"IoU = {iou:.3f}",
                    transform=ax.transAxes, fontsize=8, color="white",
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0))
            ax.axis("off")

    add_legend(fig)
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave room for legend
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
