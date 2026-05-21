"""
Benchmark substring queries using 16-bit character bitmask fingerprint.

For each sampled pattern p, compares:
  full scan:  WHERE title LIKE '%p%'
  fp filter:  WHERE (fp16_chars & mask) = mask AND title LIKE '%p%'

Patterns are real substrings extracted from the data, stratified by length.
"""

import argparse
import csv
import json
import os
import random
import time

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FEATURES_JSON = "substr_features.json"


def load_features():
    with open(FEATURES_JSON) as f:
        data = json.load(f)
    if "selected_features" in data:
        return data["selected_features"], data.get("ngram", 1)
    else:
        return data["selected_chars"], 1


def extract_ngrams(s: str, n: int) -> set:
    s = (s or "").lower()
    if len(s) < n:
        return set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def pattern_mask(p: str, selected_features: list, ngram: int) -> int:
    mask  = 0
    grams = extract_ngrams(p, ngram)
    for i, feat in enumerate(selected_features):
        if feat in grams:
            mask |= (1 << i)
    return mask


def time_query(con, sql, warmup=2, reps=5):
    for _ in range(warmup):
        con.execute(sql).fetchall()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        con.execute(sql).fetchall()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000  # ms


def extract_substrings(con, lengths, n_per_length=500, seed=42):
    """Extract real substrings of various lengths from the data."""
    rng = random.Random(seed)
    all_patterns = {}

    for length in lengths:
        print(f"  Extracting length-{length} substrings...")
        sql = f"""
            SELECT DISTINCT LOWER(SUBSTRING(title, pos, {length})) as sub
            FROM (
                SELECT title,
                       UNNEST(generate_series(1, LENGTH(title) - {length - 1})) as pos
                FROM t
                WHERE LENGTH(title) >= {length}
                LIMIT 500000
            )
            WHERE LENGTH(LOWER(SUBSTRING(title, pos, {length}))) = {length}
        """
        result = [row[0] for row in con.execute(sql).fetchall()]

        # Filter: no quotes/backslashes, only printable ASCII
        clean = [
            s for s in result
            if s and len(s) == length
            and "'" not in s
            and "\\" not in s
            and all(32 <= ord(c) < 127 for c in s)
        ]

        sampled = rng.sample(clean, min(n_per_length, len(clean)))
        all_patterns[length] = sampled
        print(f"    Found {len(clean):,} unique, sampled {len(sampled)}")

    return all_patterns


def plot_results(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Speedup vs selectivity
    fig, ax = plt.subplots(figsize=(10, 6))
    lengths = sorted(set(r["pattern_length"] for r in results))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(lengths)))

    for color, length in zip(cmap, lengths):
        subset = [r for r in results if r["pattern_length"] == length]
        sels = np.array([r["selectivity"] * 100 for r in subset])
        sus  = np.array([r["speedup"] for r in subset])
        ax.scatter(sels, sus, color=color, alpha=0.4, s=15,
                   label=f"len={length}")

        # Binned average
        if len(sels) > 5:
            lo, hi = max(sels.min(), 1e-6), sels.max()
            if hi > lo:
                bins = np.logspace(np.log10(lo), np.log10(hi), 15)
                for i in range(len(bins) - 1):
                    mask = (sels >= bins[i]) & (sels < bins[i + 1])
                    if mask.sum() >= 3:
                        ax.plot(np.sqrt(bins[i] * bins[i+1]),
                                np.mean(sus[mask]), 'o', color=color, ms=6)

    ax.axhline(1, color="gray", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Selectivity (% of rows matching)")
    ax.set_ylabel("Speedup vs full scan")
    ax.set_title("Substring Fingerprint Speedup vs Selectivity")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "speedup_vs_selectivity.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    plt.close(fig)

    # 2. Speedup vs pattern length (box plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    data_by_length = {
        length: [r["speedup"] for r in results if r["pattern_length"] == length]
        for length in lengths
    }
    ax.boxplot([data_by_length[l] for l in lengths], labels=lengths)
    ax.axhline(1, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Pattern length (chars)")
    ax.set_ylabel("Speedup vs full scan")
    ax.set_title("Substring Fingerprint Speedup by Pattern Length")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(out_dir, "speedup_by_length.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    plt.close(fig)

    # 3. Speedup vs false positive rate
    fig, ax = plt.subplots(figsize=(10, 6))
    for color, length in zip(cmap, lengths):
        subset = [r for r in results if r["pattern_length"] == length]
        fpr  = np.array([r["fp_rate"] * 100 for r in subset])
        sus  = np.array([r["speedup"] for r in subset])
        ax.scatter(fpr, sus, color=color, alpha=0.4, s=15, label=f"len={length}")

    ax.axhline(1, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("False positive rate (% of rows passing bitmask but not LIKE)")
    ax.set_ylabel("Speedup vs full scan")
    ax.set_title("Substring Fingerprint Speedup vs False Positive Rate")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "speedup_vs_fpr.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Benchmark substring fp16 fingerprint")
    parser.add_argument("--samples", type=int, default=300,
                        help="Samples per pattern length (default: 300)")
    parser.add_argument("--lengths", type=int, nargs="+", default=[3, 4, 5, 6, 8, 10],
                        help="Pattern lengths to test (default: 3 4 5 6 8 10)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--csv", type=str, default="substr_fp16.csv")
    parser.add_argument("--out-dir", type=str, default="substr_plots")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    parquet = "title_strs_substr_fp16.parquet"
    if not os.path.exists(parquet):
        raise FileNotFoundError(f"Missing {parquet}. Run build_substr.py first.")
    if not os.path.exists(FEATURES_JSON):
        raise FileNotFoundError(f"Missing {FEATURES_JSON}. Run build_substr.py first.")

    selected_features, ngram = load_features()
    print(f"Loaded {len(selected_features)} features (ngram={ngram}): {selected_features}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=1")
    con.execute("PRAGMA enable_object_cache=true")

    print("Loading table...")
    con.execute(f"CREATE TABLE t AS SELECT * FROM read_parquet('{parquet}')")

    print("Warming cache...")
    con.execute("SELECT COUNT(*), MIN(title), MAX(title) FROM t").fetchall()

    total_rows = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    print(f"Total rows: {total_rows:,}")

    print("\nExtracting substring patterns...")
    patterns_by_length = extract_substrings(con, args.lengths, args.samples, args.seed)

    results = []
    total = sum(len(v) for v in patterns_by_length.values())
    done = 0

    print(f"\nBenchmarking {total} queries (warmup={args.warmup}, reps={args.reps})...")

    for length, patterns in sorted(patterns_by_length.items()):
        for pattern in patterns:
            if (done + 1) % 100 == 0 or done == 0:
                print(f"  {done+1}/{total} ({100*(done+1)/total:.0f}%)")
            done += 1

            pe = pattern.replace("'", "''")
            mask = pattern_mask(pattern, selected_features, ngram)
            mask_bits = bin(mask).count('1')

            q_full = f"SELECT COUNT(*) FROM t WHERE title ILIKE '%{pe}%'"
            q_fp   = (f"SELECT COUNT(*) FROM t "
                      f"WHERE (fp16_chars & {mask}) = {mask} "
                      f"AND title ILIKE '%{pe}%'")

            match_count = con.execute(q_full).fetchone()[0]
            if match_count == 0:
                continue

            # Count rows passing bitmask (to compute FP rate)
            fp_count = con.execute(
                f"SELECT COUNT(*) FROM t WHERE (fp16_chars & {mask}) = {mask}"
            ).fetchone()[0]

            t_full = time_query(con, q_full, args.warmup, args.reps)
            t_fp   = time_query(con, q_fp,   args.warmup, args.reps)

            results.append({
                "pattern":        pattern,
                "pattern_length": length,
                "mask":           mask,
                "mask_bits":      mask_bits,
                "match_count":    match_count,
                "fp_count":       fp_count,
                "selectivity":    match_count / total_rows,
                "fp_rate":        (fp_count - match_count) / total_rows,
                "time_full_ms":   t_full,
                "time_fp_ms":     t_fp,
                "speedup":        t_full / t_fp if t_fp > 0 else 0,
            })

    print(f"\nCompleted {len(results)} queries with matches")

    if results:
        print("\n" + "=" * 60)
        print("SUMMARY by pattern length")
        print("=" * 60)
        print(f"{'Length':<8} {'N':>5}  {'Geo mean':>10}  {'Median':>8}  {'FP rate':>10}")
        print("-" * 60)
        for length in sorted(set(r["pattern_length"] for r in results)):
            subset = [r for r in results if r["pattern_length"] == length]
            sus = [r["speedup"] for r in subset if r["speedup"] > 0]
            fpr = [r["fp_rate"] * 100 for r in subset]
            gm  = np.exp(np.mean(np.log(sus))) if sus else 0
            print(f"{length:<8} {len(subset):>5}  {gm:>9.2f}x  "
                  f"{np.median(sus):>7.2f}x  {np.mean(fpr):>9.2f}%")
        print("=" * 60)

        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote {args.csv}")

        plot_results(results, args.out_dir)

    con.close()


if __name__ == "__main__":
    main()
