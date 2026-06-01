"""
Stratified sample from baseline CD results for guidance scale tuning.

Divides the CD distribution into N equal-frequency bins and samples K instances
per bin, giving a representative dev set that spans the full difficulty range.

Usage:
    python scripts/sample_stratified.py
    python scripts/sample_stratified.py --bins 5 --per-bin 17 --seed 42
    python scripts/sample_stratified.py --csv outputs/baseline_all/results_multi_init.csv --out misc/tune_85.txt
"""

import argparse
import csv
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default="outputs/baseline_all/results_multi_init.csv")
    parser.add_argument("--out",     default="misc/tune_85.txt")
    parser.add_argument("--bins",    type=int, default=5)
    parser.add_argument("--per-bin", type=int, default=17)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    rows.sort(key=lambda r: float(r["chamfer"]))
    n = len(rows)
    print(f"Loaded {n} instances from {args.csv}")

    rng = random.Random(args.seed)
    bin_size = n // args.bins
    sampled = []
    for b in range(args.bins):
        lo = b * bin_size
        hi = lo + bin_size if b < args.bins - 1 else n
        bucket = rows[lo:hi]
        k = min(args.per_bin, len(bucket))
        chosen = rng.sample(bucket, k)
        cd_vals = [float(r["chamfer"]) for r in chosen]
        print(f"  bin {b+1}/{args.bins}  CD [{float(bucket[0]['chamfer']):.4f}, {float(bucket[-1]['chamfer']):.4f}]"
              f"  n={len(bucket)}  sampled={k}  mean={sum(cd_vals)/len(cd_vals):.4f}")
        sampled.extend(chosen)

    sampled.sort(key=lambda r: float(r["chamfer"]), reverse=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"# Stratified CD sample: {args.bins} bins x {args.per_bin} = {len(sampled)} instances\n")
        f.write(f"# Source: {args.csv}  seed={args.seed}\n")
        for r in sampled:
            f.write(f"{r['category']}/{r['instance']}\n")

    print(f"\nWrote {len(sampled)} instances to {args.out}")


if __name__ == "__main__":
    main()
