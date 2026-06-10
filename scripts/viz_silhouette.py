"""
Visualize image / GT mask / rendered silhouette for a single instance.
Helps debug IoU values.

Usage:
    python scripts/viz_silhouette.py --cat book --inst COCO_train2014_000000416549_3
    python scripts/viz_silhouette.py --cat chair --inst 2398495 --config ss_1.0
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
import trimesh
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from guidance import _extract_mesh, _render_soft_silhouette

DATA     = os.path.join(ROOT, "data/Open3DHOI/data")
BASELINE = "/gpfs/scratch1/shared/scur0847_sam3d_baseline"
SWEEP    = "/gpfs/scratch1/shared/scur0847_sweep"
OUT_DIR  = os.path.join(ROOT, "outputs/viz_sil")
SIZE     = 256


def render_sil(pred_dir, device, use_final_step=False):
    pose_path       = os.path.join(pred_dir, "pose_params.pt")
    final_step_path = os.path.join(pred_dir, "final_step.pt")
    mesh_path       = os.path.join(pred_dir, "pred_mesh.obj")
    if not os.path.exists(pose_path):
        return None, None
    pose = torch.load(pose_path, weights_only=False, map_location="cpu")

    if use_final_step and os.path.exists(final_step_path):
        data   = torch.load(final_step_path, weights_only=False, map_location="cpu")
        result = _extract_mesh(data["ss_grid"], device)
        if result is None:
            print("  [final_step] _extract_mesh returned None (empty grid)")
            return None, pose
        verts, faces = result
    elif os.path.exists(mesh_path):
        m     = trimesh.load(mesh_path, force="mesh")
        verts = torch.tensor(np.array(m.vertices), dtype=torch.float32)
        faces = torch.tensor(np.array(m.faces),    dtype=torch.int64)
    else:
        return None, None

    sil = _render_soft_silhouette(
        verts, faces,
        pose["rotation"], pose["translation"], pose["scale"], pose["intrinsics"],
        SIZE, device,
    ).detach().cpu().numpy()
    return sil, pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat",    required=True)
    parser.add_argument("--inst",   required=True)
    parser.add_argument("--config", default=None, help="sweep config name; omit for baseline")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    inst_dir  = os.path.join(DATA, args.cat, args.inst)
    img_path  = os.path.join(inst_dir, "image.jpg")
    mask_path = os.path.join(inst_dir, "obj_mask.png")

    if args.config:
        pred_dir = os.path.join(SWEEP, args.config, args.cat, args.inst)
        label    = args.config
    else:
        pred_dir = os.path.join(BASELINE, args.cat, args.inst)
        label    = "baseline"

    print(f"Instance : {args.cat}/{args.inst}")
    print(f"Config   : {label}")
    print(f"Pred dir : {pred_dir}")

    img  = np.array(Image.open(img_path).resize((SIZE, SIZE)))
    gt   = np.array(Image.open(mask_path).resize((SIZE, SIZE), Image.NEAREST))
    gt_b = (gt > 127).astype(np.float32)

    def iou_np(a, b):
        inter = (a * b).sum()
        union = (a + b - a * b).sum().clip(min=1)
        return inter / union

    def report(sil, tag):
        pred_b = (sil > 0.1).astype(np.float32)
        nonzero = sil[sil > 0]
        print(f"\n--- {tag} ---")
        print(f"sil: min={sil.min():.4f}  max={sil.max():.4f}  mean={sil.mean():.4f}")
        print(f"non-zero: {len(nonzero)}/{sil.size} ({100*len(nonzero)/sil.size:.2f}%)")
        print(f"IoU@0.5 : {iou_np((sil>0.5).astype(np.float32), gt_b):.4f}")
        print(f"IoU@0.1 : {iou_np(pred_b, gt_b):.4f}")
        return pred_b

    # pred_mesh.obj silhouette
    sil_mesh, _ = render_sil(pred_dir, args.device, use_final_step=False)
    # final_step.pt (FlexiCubes solid mesh) silhouette
    sil_fc, pose = render_sil(pred_dir, args.device, use_final_step=True)

    if sil_mesh is None and sil_fc is None:
        print("ERROR: missing both pred_mesh.obj and final_step.pt")
        return

    pred_b_mesh = report(sil_mesh, "pred_mesh.obj") if sil_mesh is not None else None
    pred_b_fc   = report(sil_fc,   "final_step (FlexiCubes)") if sil_fc is not None else None

    # Overlay using FlexiCubes if available, else mesh
    pred_b_best = pred_b_fc if pred_b_fc is not None else pred_b_mesh
    overlay = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    overlay[..., 1] = gt_b
    overlay[..., 0] = pred_b_best
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    axes[0].imshow(img);   axes[0].set_title("Image")
    axes[1].imshow(gt_b,   cmap="gray"); axes[1].set_title("GT mask")
    if sil_mesh is not None:
        axes[2].imshow(sil_mesh, cmap="hot")
        axes[2].set_title(f"pred_mesh.obj\nIoU@0.1={iou_np(pred_b_mesh, gt_b):.3f}")
    if sil_fc is not None:
        axes[3].imshow(sil_fc, cmap="hot")
        axes[3].set_title(f"FlexiCubes (final_step)\nIoU@0.1={iou_np(pred_b_fc, gt_b):.3f}")
        axes[4].imshow(pred_b_fc, cmap="gray")
        axes[4].set_title(f"FC bin >0.1\nIoU={iou_np(pred_b_fc, gt_b):.3f}")
    axes[5].imshow(overlay); axes[5].set_title(f"Overlay (FC)\ngreen=GT red=pred")
    for ax in axes:
        ax.axis("off")
    plt.suptitle(f"{label} / {args.cat}/{args.inst}")
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{label}_{args.cat}_{args.inst}.png")
    plt.savefig(out, dpi=120)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
