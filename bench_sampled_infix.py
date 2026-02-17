"""
Benchmark sampled prefix+suffix (infix) combinations.

Since the full combination space of 3-char prefixes × 3-char suffixes is too large,
we sample random pairs and measure speedup vs. selectivity.
"""

import argparse
import csv
import os
import random
import time

import duckdb
import numpy as np
import pandas as pd


PREFIX_BYTES = 8


def norm(s):
    return (s or "").lower()


def key_u64_from_normed(s_norm, nbytes=PREFIX_BYTES):
    b = s_norm.encode("utf-8", errors="ignore")[:nbytes]
    b = b + b"\x00" * (nbytes - len(b))
    return int.from_bytes(b, "big", signed=False)


def next_prefix_normed(s_norm: str) -> str:
    b = bytearray(s_norm.encode("utf-8", errors="ignore"))
    if not b:
        return "\uffff"
    b[-1] = min(255, b[-1] + 1)
    return bytes(b).decode("utf-8", errors="ignore")


def bucket_range(query: str, boundaries: np.ndarray, bits: int, suffix: bool):
    s = norm(query)
    if suffix:
        s = s[::-1]
    lo = key_u64_from_normed(s)
    hi = key_u64_from_normed(next_prefix_normed(s))
    if boundaries is None:
        shift = 64 - bits
        jlo = int(np.right_shift(lo, shift))
        jhi = int(np.right_shift(hi, shift))
        return min(jlo, jhi), max(jlo, jhi)
    jlo = np.searchsorted(boundaries, lo, side="right") - 1
    jhi = np.searchsorted(boundaries, hi, side="right") - 1
    jlo = int(np.clip(jlo, 0, len(boundaries) - 1))
    jhi = int(np.clip(jhi, 0, len(boundaries) - 1))
    return min(jlo, jhi), max(jlo, jhi)


def time_query(con, sql, warmup=1, reps=5):
    for _ in range(warmup):
        con.execute(sql).fetchall()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        con.execute(sql).fetchall()
        times.append(time.perf_counter() - t0)
    return times


def extract_distinct_ngrams(con, n=3, suffix=False):
    """Extract all distinct n-character prefixes or suffixes from the data."""
    if suffix:
        sql = f"""
            SELECT DISTINCT LOWER(SUBSTRING(title, LENGTH(title) - {n-1}, {n})) as ngram
            FROM t
            WHERE LENGTH(title) >= {n}
        """
    else:
        sql = f"""
            SELECT DISTINCT LOWER(SUBSTRING(title, 1, {n})) as ngram
            FROM t
            WHERE LENGTH(title) >= {n}
        """

    result = con.execute(sql).fetchall()
    return [r[0] for r in result if r[0] and len(r[0]) == n]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sampled prefix+suffix combinations"
    )
    parser.add_argument("--prefix-bits", type=int, default=8, help="Prefix fingerprint bits")
    parser.add_argument("--suffix-bits", type=int, default=8, help="Suffix fingerprint bits")
    parser.add_argument("--n", type=int, default=3, help="N-gram length (default: 3)")
    parser.add_argument("--samples", type=int, default=500, help="Number of random pairs to sample")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per query")
    parser.add_argument("--reps", type=int, default=5, help="Timed runs per query")
    parser.add_argument("--csv", type=str, default="", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    prefix_bits = args.prefix_bits
    suffix_bits = args.suffix_bits

    parquet = f"title_strs_infix_p{prefix_bits}_s{suffix_bits}.parquet"
    prefix_boundaries_npy = f"q{prefix_bits}_prefix_boundaries.npy"
    suffix_boundaries_npy = f"q{suffix_bits}_suffix_boundaries.npy"
    prefix_col = f"q{prefix_bits}_prefix"
    suffix_col = f"q{suffix_bits}_suffix"

    if not os.path.exists(parquet):
        raise FileNotFoundError(
            f"Missing {parquet}. Run build_infix.py --prefix-bits {prefix_bits} --suffix-bits {suffix_bits} first."
        )

    prefix_boundaries = np.load(prefix_boundaries_npy) if os.path.exists(prefix_boundaries_npy) else None
    suffix_boundaries = np.load(suffix_boundaries_npy) if os.path.exists(suffix_boundaries_npy) else None

    con = duckdb.connect()
    con.execute("PRAGMA threads=1")
    con.execute("PRAGMA enable_object_cache=true")
    con.execute(f"CREATE TABLE t AS SELECT * FROM read_parquet('{parquet}')")

    total_rows = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    print(f"Total rows: {total_rows:,}")
    print(f"Prefix bits: {prefix_bits}, Suffix bits: {suffix_bits}")
    print(f"N-gram length: {args.n}")

    # Extract distinct n-grams
    print(f"\nExtracting distinct {args.n}-char prefixes and suffixes...")
    prefixes = extract_distinct_ngrams(con, n=args.n, suffix=False)
    suffixes = extract_distinct_ngrams(con, n=args.n, suffix=True)
    print(f"Found {len(prefixes):,} distinct prefixes, {len(suffixes):,} distinct suffixes")
    print(f"Total possible combinations: {len(prefixes) * len(suffixes):,}")

    # Sample random pairs
    num_samples = min(args.samples, len(prefixes) * len(suffixes))
    print(f"Sampling {num_samples} random (prefix, suffix) pairs...")

    # Generate unique random pairs
    sampled_pairs = set()
    while len(sampled_pairs) < num_samples:
        p = random.choice(prefixes)
        s = random.choice(suffixes)
        sampled_pairs.add((p, s))

    sampled_pairs = list(sampled_pairs)
    print(f"Generated {len(sampled_pairs)} unique pairs")

    # Benchmark each pair
    results = []
    print(f"\nBenchmarking {len(sampled_pairs)} queries (warmup={args.warmup}, reps={args.reps})...")

    for i, (prefix_q, suffix_q) in enumerate(sampled_pairs):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Progress: {i+1}/{len(sampled_pairs)} ({100*(i+1)/len(sampled_pairs):.1f}%)")

        # Full query (no fingerprint)
        q_full = (
            f"SELECT COUNT(*) FROM t "
            f"WHERE title ILIKE '{prefix_q}%' AND title ILIKE '%{suffix_q}'"
        )

        # Get match count and selectivity
        match_count = con.execute(q_full).fetchone()[0]
        selectivity = match_count / total_rows if total_rows > 0 else 0

        # Skip if no matches
        if match_count == 0:
            continue

        # Build fingerprint-filtered query
        p_lo, p_hi = bucket_range(prefix_q, prefix_boundaries, prefix_bits, suffix=False)
        s_lo, s_hi = bucket_range(suffix_q, suffix_boundaries, suffix_bits, suffix=True)

        q_fp = (
            f"SELECT COUNT(*) FROM t "
            f"WHERE {prefix_col} BETWEEN {p_lo} AND {p_hi} "
            f"AND {suffix_col} BETWEEN {s_lo} AND {s_hi} "
            f"AND title ILIKE '{prefix_q}%' AND title ILIKE '%{suffix_q}'"
        )

        # Time both queries
        full_times = time_query(con, q_full, warmup=args.warmup, reps=args.reps)
        fp_times = time_query(con, q_fp, warmup=args.warmup, reps=args.reps)

        full_med = float(np.median(full_times))
        fp_med = float(np.median(fp_times))
        speedup = full_med / fp_med if fp_med > 0 else 0

        # Calculate combined pruning rate
        rows_in_buckets = con.execute(
            f"SELECT COUNT(*) FROM t "
            f"WHERE {prefix_col} BETWEEN {p_lo} AND {p_hi} "
            f"AND {suffix_col} BETWEEN {s_lo} AND {s_hi}"
        ).fetchone()[0]
        prune_rate = 1.0 - (rows_in_buckets / total_rows) if total_rows > 0 else 0

        results.append({
            "prefix": prefix_q,
            "suffix": suffix_q,
            "match_count": match_count,
            "selectivity": selectivity,
            "prefix_bucket_span": p_hi - p_lo + 1,
            "suffix_bucket_span": s_hi - s_lo + 1,
            "rows_in_buckets": rows_in_buckets,
            "prune_rate": prune_rate,
            "time_full_ms": full_med * 1000,
            "time_fp_ms": fp_med * 1000,
            "speedup": speedup,
        })

    print(f"\nCompleted {len(results)} queries with matches")

    # Summary statistics
    if results:
        speedups = [r["speedup"] for r in results]
        selectivities = [r["selectivity"] for r in results]
        prune_rates = [r["prune_rate"] for r in results]

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Queries tested: {len(results)}")
        print(f"Speedup - Mean: {np.mean(speedups):.2f}x, Median: {np.median(speedups):.2f}x")
        print(f"Speedup - Min: {min(speedups):.2f}x, Max: {max(speedups):.2f}x")
        print(f"Speedup - Geometric mean: {np.exp(np.mean(np.log(speedups))):.2f}x")
        print(f"Prune rate - Mean: {np.mean(prune_rates)*100:.2f}%")
        print(f"Selectivity - Mean: {np.mean(selectivities)*100:.6f}%")

    # Write CSV
    if args.csv and results:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"\nWrote results to {args.csv}")

    con.close()


if __name__ == "__main__":
    main()
