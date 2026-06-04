"""
Stratified sample from baseline results CSV, stratified by Chamfer distance.
Ensures tune_85 instances are included in the 50% eval set.

Usage:
    python scripts/make_stratified_sample.py
    python scripts/make_stratified_sample.py --frac 0.5 --out misc/eval_855.txt
    python scripts/make_stratified_sample.py --frac 0.25 --out misc/pose_428.txt
"""
import argparse
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_txt(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                items.append(line)
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",  default="outputs/baseline_all/results_multi_init.csv")
    parser.add_argument("--frac", type=float, default=0.5)
    parser.add_argument("--bins", type=int,   default=10)
    parser.add_argument("--seed", type=int,   default=42)
    parser.add_argument("--must-include", default="misc/tune_85.txt",
                        help="instances that must appear in the sample")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    must = set(load_txt(args.must_include)) if args.must_include else set()

    rows = []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            key = f"{row['category']}/{row['instance']}"
            rows.append((key, float(row["chamfer"])))

    print(f"Total instances: {len(rows)}")
    print(f"Must-include   : {len(must)}")

    cds   = np.array([r[1] for r in rows])
    edges = np.percentile(cds, np.linspace(0, 100, args.bins + 1))
    edges[0] -= 1e-9

    sampled = set(must)  # seed with required instances

    for i in range(args.bins):
        lo, hi   = edges[i], edges[i + 1]
        in_bin   = [r[0] for r in rows if lo < r[1] <= hi]
        n_target = max(1, round(len(in_bin) * args.frac))
        # already-included instances in this bin don't count toward quota
        already  = [k for k in in_bin if k in sampled]
        remaining = [k for k in in_bin if k not in sampled]
        n_need   = max(0, n_target - len(already))
        chosen   = rng.choice(remaining, size=min(n_need, len(remaining)), replace=False).tolist()
        sampled.update(chosen)

    sampled_list = sorted(sampled)
    print(f"Sampled {len(sampled_list)} instances ({100*len(sampled_list)/len(rows):.1f}%)")

    # Distribution check
    sampled_set = set(sampled_list)
    sampled_cds = [r[1] for r in rows if r[0] in sampled_set]
    print(f"CD mean   all={np.mean(cds):.4f}  sampled={np.mean(sampled_cds):.4f}")
    print(f"CD median all={np.median(cds):.4f}  sampled={np.median(sampled_cds):.4f}")
    print(f"CD p90    all={np.percentile(cds,90):.4f}  sampled={np.percentile(sampled_cds,90):.4f}")

    tune_covered = must & sampled_set
    print(f"tune_85 covered: {len(tune_covered)}/{len(must)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"# Stratified CD sample: frac={args.frac} bins={args.bins} seed={args.seed}\n")
        f.write(f"# must-include={args.must_include}  total={len(sampled_list)}\n")
        for key in sampled_list:
            f.write(key + "\n")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
