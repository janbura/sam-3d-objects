"""
Evaluate all baseline_all predictions vs GT meshes.
Computes Chamfer distance, F-score (4 thresholds), and mask IoU (hard).
Saves results sorted by CD descending (worst first).

Usage:
    python eval_baseline_all.py
    python eval_baseline_all.py --device cpu
    python eval_baseline_all.py --out outputs/baseline_all/results.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import trimesh
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, "notebook"))
from eval_single import chamfer, f_score, load_gt_points, normalize, seed_everything
from evaluation.alignment import align_icp
from guidance import _extract_mesh, _render_soft_silhouette

DATA_ROOT    = "data/Open3DHOI/data"
PRED_ROOT    = "outputs/baseline_all"
RENDER_SIZE  = 256
FSCORE_THRESHOLDS = (0.005, 0.01, 0.02, 0.05)


def find_pairs(pred_root):
    for cat in sorted(os.listdir(pred_root)):
        cat_dir = os.path.join(pred_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        for inst in sorted(os.listdir(cat_dir)):
            pred_mesh = os.path.join(cat_dir, inst, "pred_mesh.obj")
            gt_mesh   = os.path.join(DATA_ROOT, cat, inst, "object_mesh.obj")
            if os.path.exists(pred_mesh) and os.path.exists(gt_mesh):
                yield cat, inst, os.path.join(cat_dir, inst), gt_mesh


def compute_iou(pred_dir, gt_mask_path, device):
    pose_path       = os.path.join(pred_dir, "pose_params.pt")
    final_step_path = os.path.join(pred_dir, "final_step.pt")
    if not os.path.exists(pose_path):
        return None
    pose = torch.load(pose_path, weights_only=False, map_location="cpu")

    if os.path.exists(final_step_path):
        data   = torch.load(final_step_path, weights_only=False, map_location="cpu")
        result = _extract_mesh(data["ss_grid"], device)
        if result is None:
            return None
        verts, faces = result
        verts = verts.to(device)
        faces = faces.to(device)
    else:
        m     = trimesh.load(os.path.join(pred_dir, "pred_mesh.obj"), force="mesh")
        verts = torch.tensor(np.array(m.vertices), dtype=torch.float32, device=device)
        faces = torch.tensor(np.array(m.faces),    dtype=torch.int64,   device=device)
    sil   = _render_soft_silhouette(
        verts, faces,
        pose["rotation"], pose["translation"], pose["scale"], pose["intrinsics"],
        RENDER_SIZE, device,
    )
    gt_np = np.array(Image.open(gt_mask_path).resize((RENDER_SIZE, RENDER_SIZE), Image.NEAREST))
    gt    = torch.tensor((gt_np > 127).astype(np.float32), device=device)
    inter = ((sil > 0.1).float() * gt).sum()
    union = ((sil > 0.1).float() + gt - (sil > 0.1).float() * gt).sum().clamp(min=1.0)
    return float(inter / union)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--pred-root", default=PRED_ROOT)
    parser.add_argument("--out",       default=None)
    args = parser.parse_args()

    pred_root = args.pred_root
    out_csv   = args.out or os.path.join(pred_root, "results_multi_init.csv")

    seed_everything()

    pairs = list(find_pairs(pred_root))
    print(f"Found {len(pairs)} prediction/GT pairs.\n")

    rows = []
    for i, (cat, inst, pred_dir, gt_path) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {cat}/{inst} ...", end=" ", flush=True)
        gt_mask = os.path.join(DATA_ROOT, cat, inst, "obj_mask.png")
        try:
            pred     = normalize(load_gt_points(os.path.join(pred_dir, "pred_mesh.obj"), device=args.device))
            gt       = normalize(load_gt_points(gt_path, device=args.device))
            pred_icp = align_icp(pred, gt, mode="grid")
            cd       = float(chamfer(pred_icp, gt))
            fs       = f_score(pred_icp, gt, thresholds=FSCORE_THRESHOLDS)
            iou      = compute_iou(pred_dir, gt_mask, args.device) if os.path.exists(gt_mask) else None
            iou_str  = f"  IoU={iou:.4f}" if iou is not None else ""
            print(f"CD={cd:.4f}  F@0.02={fs[0.02]:.4f}{iou_str}")
            row = {"category": cat, "instance": inst, "chamfer": cd}
            row.update({f"f@{tau}": fs[tau] for tau in FSCORE_THRESHOLDS})
            row["hard_iou"] = iou if iou is not None else ""
            rows.append(row)
        except Exception as e:
            print(f"ERROR: {e}")

    rows.sort(key=lambda r: r["chamfer"], reverse=True)

    fieldnames = (["category", "instance", "chamfer"]
                  + [f"f@{t}" for t in FSCORE_THRESHOLDS]
                  + ["hard_iou"])
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} results to {out_csv}")

    all_cds  = [r["chamfer"] for r in rows]
    all_f02  = [r["f@0.02"] for r in rows]
    all_ious = [r["hard_iou"] for r in rows if r["hard_iou"] != ""]

    print(f"\n{'='*60}")
    print(f"OVERALL (n={len(all_cds)})")
    print(f"  CD     mean={np.mean(all_cds):.4f}  median={np.median(all_cds):.4f}  p90={np.percentile(all_cds,90):.4f}")
    for tau in FSCORE_THRESHOLDS:
        vals = [r[f"f@{tau}"] for r in rows]
        print(f"  F@{tau}  mean={np.mean(vals):.4f}")
    if all_ious:
        print(f"  IoU    mean={np.mean(all_ious):.4f}  (n={len(all_ious)})")
    print(f"{'='*60}")

    # Table row (copy-paste)
    cd_s  = f"{np.mean(all_cds):.4f}"
    f02_s = f"{np.mean(all_f02):.4f}"
    iou_s = f"{np.mean(all_ious):.4f}" if all_ious else "--"
    print(f"\nTABLE ROW:")
    print(f"SAM 3D (no guidance) & {cd_s} & {f02_s} & {iou_s} \\\\")


if __name__ == "__main__":
    main()
