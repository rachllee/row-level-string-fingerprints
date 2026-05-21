"""
Benchmark the Arrow-compute custom scanner across many sampled patterns,
plotting speedup vs FP rate and bits set in mask.
"""

import argparse
import csv
import os
import random

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from custom_scan import fp_scan, baseline_scan, pattern_mask, load_features

PARQUET = "title_strs_substr_fp16.parquet"


def extract_patterns(parquet, lengths, n_per_length=100, seed=42):
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t AS SELECT title FROM read_parquet('{parquet}')")
    rng = random.Random(seed)
    patterns = {}
    for length in lengths:
        print(f"  Sampling length-{length} patterns...")
        sql = f"""
            SELECT DISTINCT LOWER(SUBSTRING(title, pos, {length})) as sub
            FROM (
                SELECT title,
                       UNNEST(generate_series(1, LENGTH(title) - {length-1})) as pos
                FROM t WHERE LENGTH(title) >= {length} LIMIT 300000
            )
            WHERE LENGTH(LOWER(SUBSTRING(title, pos, {length}))) = {length}
        """
        result = [row[0] for row in con.execute(sql).fetchall()]
        clean = [
            s for s in result
            if s and len(s) == length
            and "'" not in s and "\\" not in s
            and all(32 <= ord(c) < 127 for c in s)
        ]
        patterns[length] = rng.sample(clean, min(n_per_length, len(clean)))
        print(f"    {len(patterns[length])} patterns")
    con.close()
    return patterns


def plot_results(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    lengths = sorted(set(r["pattern_length"] for r in results))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(lengths)))

    # 1. Speedup vs FP rate
    fig, ax = plt.subplots(figsize=(10, 6))
    for color, length in zip(cmap, lengths):
        subset = [r for r in results if r["pattern_length"] == length]
        fpr = np.array([r["fp_rate"] * 100 for r in subset])
        sus = np.array([r["speedup"] for r in subset])
        ax.scatter(fpr, sus, color=color, alpha=0.5, s=20, label=f"len={length}")

    ax.axhline(1, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("False positive rate — % of rows passing bitmask")
    ax.set_ylabel("Speedup vs baseline")
    ax.set_title("Arrow-compute row-level filtering: speedup vs FP rate")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "speedup_vs_fpr.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    plt.close(fig)

    # 2. Speedup vs bits set in mask
    fig, ax = plt.subplots(figsize=(9, 6))
    bits_vals = sorted(set(r["mask_bits"] for r in results))
    data_by_bits = {b: [r["speedup"] for r in results if r["mask_bits"] == b]
                    for b in bits_vals}
    ax.boxplot([data_by_bits[b] for b in bits_vals], labels=bits_vals)
    ax.axhline(1, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Bits set in fingerprint mask")
    ax.set_ylabel("Speedup vs baseline")
    ax.set_title("Arrow-compute row-level filtering: speedup vs mask bits")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(out_dir, "speedup_vs_bits.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    plt.close(fig)

    # 3. Speedup vs selectivity
    fig, ax = plt.subplots(figsize=(10, 6))
    for color, length in zip(cmap, lengths):
        subset = [r for r in results if r["pattern_length"] == length]
        sel = np.array([r["selectivity"] * 100 for r in subset])
        sus = np.array([r["speedup"] for r in subset])
        ax.scatter(sel, sus, color=color, alpha=0.5, s=20, label=f"len={length}")

    ax.axhline(1, color="gray", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Selectivity — % of rows matching ILIKE")
    ax.set_ylabel("Speedup vs baseline")
    ax.set_title("Arrow-compute row-level filtering: speedup vs selectivity")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "speedup_vs_selectivity.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",  type=int, default=100,
                        help="Patterns per length (default: 100)")
    parser.add_argument("--lengths",  type=int, nargs="+", default=[3, 4, 5, 6, 8, 10],
                        help="Pattern lengths to test")
    parser.add_argument("--reps",     type=int, default=3)
    parser.add_argument("--csv",      type=str, default="custom_scan_bench.csv")
    parser.add_argument("--out-dir",  type=str, default="custom_scan_plots")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    selected_features, ngram = load_features()
    pf = pq.ParquetFile(PARQUET)
    total_rows = pf.metadata.num_rows
    print(f"Parquet: {total_rows:,} rows, {pf.metadata.num_row_groups} row groups")
    print(f"Features (ngram={ngram}): {selected_features}\n")

    print("Sampling patterns...")
    patterns_by_length = extract_patterns(PARQUET, args.lengths, args.samples, args.seed)

    # Warm up the parquet file cache
    print("\nWarming cache...")
    baseline_scan(pf, "the")

    results = []
    total = sum(len(v) for v in patterns_by_length.values())
    done = 0

    print(f"\nBenchmarking {total} patterns (reps={args.reps})...")

    for length, patterns in sorted(patterns_by_length.items()):
        for pattern in patterns:
            if (done + 1) % 50 == 0 or done == 0:
                print(f"  {done+1}/{total}")
            done += 1

            mask = pattern_mask(pattern, selected_features, ngram)
            mask_bits = bin(mask).count("1")

            # Count actual matches and rows passing bitmask (use one baseline run)
            min_b, match_count, _, t_b0 = baseline_scan(pf, pattern)
            if match_count == 0:
                continue

            # Count rows passing bitmask via Arrow compute
            if mask == 0:
                fp_pass = total_rows
            else:
                mask_scalar = pa.scalar(mask, type=pa.uint16())
                fp_pass = 0
                for rg in range(pf.metadata.num_row_groups):
                    fp_col = pf.read_row_group(rg, columns=["fp16_chars"])["fp16_chars"]
                    hits = pc.equal(pc.bit_wise_and(fp_col, mask_scalar), mask_scalar)
                    fp_pass += pc.sum(hits).as_py()

            # Timed runs
            base_times = [baseline_scan(pf, pattern)[3] for _ in range(args.reps)]
            fp_times   = [fp_scan(pf, mask, pattern)[5] for _ in range(args.reps)]

            t_base = np.median(base_times)
            t_fp   = np.median(fp_times)

            results.append({
                "pattern":        pattern,
                "pattern_length": length,
                "mask_bits":      mask_bits,
                "match_count":    match_count,
                "fp_pass":        fp_pass,
                "selectivity":    match_count / total_rows,
                "fp_rate":        fp_pass / total_rows,
                "time_base_ms":   t_base,
                "time_fp_ms":     t_fp,
                "speedup":        t_base / t_fp if t_fp > 0 else 0,
            })

    print(f"\nCompleted {len(results)} patterns with matches")

    # Summary
    print("\n" + "=" * 55)
    print(f"{'Length':<8} {'N':>5}  {'Median speedup':>14}  {'Avg FP rate':>12}")
    print("-" * 55)
    for length in sorted(set(r["pattern_length"] for r in results)):
        subset = [r for r in results if r["pattern_length"] == length]
        sus = [r["speedup"] for r in subset]
        fpr = [r["fp_rate"] * 100 for r in subset]
        print(f"{length:<8} {len(subset):>5}  {np.median(sus):>13.2f}x  {np.mean(fpr):>11.1f}%")
    print("=" * 55)

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {args.csv}")

    plot_results(results, args.out_dir)


if __name__ == "__main__":
    main()
