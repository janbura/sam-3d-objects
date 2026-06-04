"""
Aggregate eval over all sweep outputs.

Reads outputs/sweep/<config>/<cat>/<inst>/pred_mesh.obj, computes
Chamfer + F-score (shape quality) and mask IoU (placement quality)
against GT, and prints a summary table + range analysis.

Usage:
    python scripts/eval_sweep.py
    python scripts/eval_sweep.py --instances misc/tune_85.txt --device cuda
    python scripts/eval_sweep.py --configs ss_1.0 ss_5.0 --device cuda
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

DATA        = "data/Open3DHOI/data"
RENDER_SIZE = 256

FSCORE_THRESHOLDS = (0.005, 0.01, 0.02, 0.05)
FAMILY_ORDER      = {"ss": 1, "pose": 2, "depth": 3, "normal": 4}


def load_instances(path):
    instances = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cat, inst = line.split("/", 1)
            instances.append((cat, inst))
    return instances


def discover_instances(sweep_dir, configs):
    """Find all (cat, inst) pairs that appear in at least one config dir."""
    seen = set()
    for cfg in configs:
        cfg_dir = os.path.join(sweep_dir, cfg)
        if not os.path.isdir(cfg_dir):
            continue
        for cat in os.listdir(cfg_dir):
            cat_dir = os.path.join(cfg_dir, cat)
            if not os.path.isdir(cat_dir):
                continue
            for inst in os.listdir(cat_dir):
                if os.path.isdir(os.path.join(cat_dir, inst)):
                    seen.add((cat, inst))
    return sorted(seen)


def parse_config(name):
    if name == "baseline":
        return ("baseline", 0.0)
    fam, _, scale = name.partition("_")
    try:
        return (fam, float(scale))
    except ValueError:
        return (fam, 0.0)


def config_sort_key(name):
    fam, scale = parse_config(name)
    if fam == "baseline":
        return (0, 0.0)
    return (FAMILY_ORDER.get(fam, 9), scale)


def eval_cd(pred_dir, gt_obj_path, device):
    """24-rotation grid ICP Chamfer + F-score."""
    pred = normalize(load_gt_points(os.path.join(pred_dir, "pred_mesh.obj"), device=device))
    gt   = normalize(load_gt_points(gt_obj_path, device=device))
    pred_icp = align_icp(pred, gt, mode="grid")
    cd = float(chamfer(pred_icp, gt))
    fs = f_score(pred_icp, gt, thresholds=FSCORE_THRESHOLDS)
    return cd, fs


def eval_iou(pred_dir, gt_mask_path, device):
    """Mask IoU: render predicted mesh silhouette vs GT obj_mask.png.
    Uses final_step.pt (FlexiCubes solid mesh) when available, else pred_mesh.obj."""
    pose_path       = os.path.join(pred_dir, "pose_params.pt")
    final_step_path = os.path.join(pred_dir, "final_step.pt")
    if not os.path.exists(pose_path):
        return None, None

    pose = torch.load(pose_path, weights_only=False, map_location="cpu")

    if os.path.exists(final_step_path):
        data   = torch.load(final_step_path, weights_only=False, map_location="cpu")
        result = _extract_mesh(data["ss_grid"], device)
        if result is None:
            return None, None
        verts, faces = result
        verts = verts.to(device)
        faces = faces.to(device)
    else:
        m     = trimesh.load(os.path.join(pred_dir, "pred_mesh.obj"), force="mesh")
        verts = torch.tensor(np.array(m.vertices), dtype=torch.float32, device=device)
        faces = torch.tensor(np.array(m.faces),    dtype=torch.int64,   device=device)

    sil = _render_soft_silhouette(
        verts, faces,
        pose["rotation"], pose["translation"], pose["scale"], pose["intrinsics"],
        RENDER_SIZE, device,
    )

    gt_np = np.array(
        Image.open(gt_mask_path).resize((RENDER_SIZE, RENDER_SIZE), Image.NEAREST)
    )
    gt = torch.tensor((gt_np > 127).astype(np.float32), device=device)

    pred_bin = (sil > 0.1).float()
    inter    = (pred_bin * gt).sum()
    union    = (pred_bin + gt - pred_bin * gt).sum().clamp(min=1.0)
    hard_iou = float(inter / union)

    inter_s = (sil * gt).sum()
    union_s = (sil + gt - sil * gt).sum().clamp(min=1e-8)
    soft_iou = float(inter_s / union_s)

    return hard_iou, soft_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir",  default="outputs/sweep")
    parser.add_argument("--instances",  default=None,
                        help="text file with cat/inst lines; auto-discovers if omitted")
    parser.add_argument("--configs",    nargs="+", default=None)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    device    = args.device
    seed_everything()

    all_configs = sorted([
        d for d in os.listdir(sweep_dir)
        if os.path.isdir(os.path.join(sweep_dir, d))
    ])
    configs = [c for c in all_configs if c in args.configs] if args.configs else all_configs
    if not configs:
        print(f"No configs found in {sweep_dir}")
        sys.exit(1)

    instances = (load_instances(args.instances) if args.instances
                 else discover_instances(sweep_dir, configs))
    print(f"Configs: {len(configs)}  Instances: {len(instances)}\n")

    rows = []

    for cfg_name in configs:
        cfg_results = []
        for cat, inst in instances:
            pred_dir  = os.path.join(sweep_dir, cfg_name, cat, inst)
            pred_path = os.path.join(pred_dir, "pred_mesh.obj")
            gt_obj    = os.path.join(DATA, cat, inst, "object_mesh.obj")
            gt_mask   = os.path.join(DATA, cat, inst, "obj_mask.png")

            if not os.path.exists(pred_path):
                continue

            print(f"  {cfg_name} / {cat}/{inst} ...", end=" ", flush=True)
            try:
                cd, fs             = eval_cd(pred_dir, gt_obj, device)
                hard_iou, soft_iou = eval_iou(pred_dir, gt_mask, device)
                iou_str = f"  IoU={hard_iou:.4f}" if hard_iou is not None else ""
                print(f"CD={cd:.4f}  F@0.02={fs[0.02]:.4f}{iou_str}")

                row = {"config": cfg_name, "category": cat, "instance": inst, "chamfer": cd}
                row.update({f"f@{tau}": fs[tau] for tau in FSCORE_THRESHOLDS})
                row["hard_iou"] = hard_iou if hard_iou is not None else ""
                row["soft_iou"] = soft_iou if soft_iou is not None else ""
                rows.append(row)
                cfg_results.append((cd, fs, hard_iou))
            except Exception as e:
                print(f"ERROR: {e}")

        if cfg_results:
            mean_cd  = np.mean([r[0] for r in cfg_results])
            mean_f02 = np.mean([r[1][0.02] for r in cfg_results])
            ious     = [r[2] for r in cfg_results if r[2] is not None]
            iou_str  = f"  IoU={np.mean(ious):.4f}" if ious else ""
            print(f"  → {cfg_name}  CD={mean_cd:.4f}  F@0.02={mean_f02:.4f}{iou_str}\n")

    # Save CSV
    csv_path   = os.path.join(sweep_dir, "results.csv")
    fieldnames = (["config", "category", "instance", "chamfer"]
                  + [f"f@{t}" for t in FSCORE_THRESHOLDS]
                  + ["hard_iou", "soft_iou"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-sample results to {csv_path}")

    # Per-config means
    mean_cd  = {}
    mean_iou = {}
    for cfg_name in configs:
        cfg_rows = [r for r in rows if r["config"] == cfg_name]
        if cfg_rows:
            mean_cd[cfg_name]  = np.mean([r["chamfer"] for r in cfg_rows])
            ious = [r["hard_iou"] for r in cfg_rows if r["hard_iou"] != ""]
            mean_iou[cfg_name] = np.mean(ious) if ious else None

    base_cd  = mean_cd.get("baseline")
    base_iou = mean_iou.get("baseline")

    # Summary table (CD + all F-score thresholds + hard IoU)
    hdr = f"{'config':<20} {'CD↓':>8}"
    for tau in FSCORE_THRESHOLDS:
        hdr += f" {'F@'+str(tau)+'↑':>9}"
    hdr += f" {'IoU↑(hard)':>11}  n"
    print(f"\n{hdr}")
    print("-" * (len(hdr) + 4))
    for cfg_name in sorted(mean_cd, key=config_sort_key):
        cfg_rows = [r for r in rows if r["config"] == cfg_name]
        m_cd  = np.mean([r["chamfer"] for r in cfg_rows])
        line  = f"{cfg_name:<20} {m_cd:>8.4f}"
        for tau in FSCORE_THRESHOLDS:
            mf = np.mean([r[f"f@{tau}"] for r in cfg_rows])
            line += f" {mf:>9.4f}"
        ious  = [r["hard_iou"] for r in cfg_rows if r["hard_iou"] != ""]
        iou_s = f"{np.mean(ious):>11.4f}" if ious else "        n/a"
        line += f" {iou_s}  {len(cfg_rows)}"
        print(line)

    # Range analysis
    print(f"\n\n{'='*64}")
    print("RANGE ANALYSIS  (Δ = mean − baseline; CD: negative = good, IoU: positive = good)")
    if base_cd:
        print(f"baseline  CD={base_cd:.4f}  IoU={base_iou:.4f}" if base_iou else
              f"baseline  CD={base_cd:.4f}")
    print("=" * 64)

    fam_entries = defaultdict(list)
    for cfg_name, m in mean_cd.items():
        fam, scale = parse_config(cfg_name)
        if fam in FAMILY_ORDER:
            fam_entries[fam].append((scale, cfg_name, m, mean_iou.get(cfg_name)))

    for fam in sorted(fam_entries, key=lambda f: FAMILY_ORDER[f]):
        entries = sorted(fam_entries[fam])
        print(f"\n[{fam}]   scale      CD      ΔCD       IoU     ΔIoU    trend")
        print(f"  {'-'*58}")
        prev_cd = base_cd
        for scale, cfg_name, m_cd, m_iou in entries:
            dcd  = f"{m_cd - base_cd:+.4f}" if base_cd  is not None else "n/a"
            diou = (f"{m_iou - base_iou:+.4f}" if (m_iou is not None and base_iou is not None)
                    else "  n/a")
            iou_s = f"{m_iou:.4f}" if m_iou is not None else "  n/a"
            arrow = ""
            if prev_cd is not None:
                arrow = "↓" if m_cd < prev_cd - 1e-4 else ("↑" if m_cd > prev_cd + 1e-4 else "~")
            print(f"  {scale:>8.3f}  {m_cd:.4f}  {dcd:>8}  {iou_s:>7}  {diou:>8}  {arrow}")
            prev_cd = m_cd

        best = min(entries, key=lambda e: e[2])
        verdict = (f"best CD: {best[1]} (CD={best[2]:.4f}, Δ={best[2]-base_cd:+.4f})"
                   if base_cd is not None else f"best CD: {best[1]} (CD={best[2]:.4f})")
        print(f"  -> {verdict}")

    print(f"\n{'='*64}")


if __name__ == "__main__":
    main()
