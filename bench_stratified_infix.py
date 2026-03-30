"""
Benchmark real infix queries with stratified sampling across bucket spans.

This ensures we get queries across different bucket span ranges, not just
the most common low-span queries.
"""

import argparse
import csv
import os
import random
import time

import duckdb
import numpy as np


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


def extract_infix_pairs_with_spans(con, prefix_boundaries, suffix_boundaries,
                                   prefix_bits, suffix_bits, n=3):
    """
    Extract real (prefix, suffix) pairs and calculate their bucket spans.
    Returns list of (prefix, suffix, prefix_span, suffix_span, combined_span).
    """
    print(f"Extracting real {n}-char (prefix, suffix) pairs with bucket spans...")

    sql = f"""
        SELECT DISTINCT
            LOWER(SUBSTRING(title, 1, {n})) as prefix,
            LOWER(SUBSTRING(title, LENGTH(title) - {n-1}, {n})) as suffix
        FROM t
        WHERE LENGTH(title) >= {n}
    """

    result = con.execute(sql).fetchall()

    # Calculate bucket spans for each pair
    pairs_with_spans = []
    for prefix, suffix in result:
        # Filter out problematic characters
        if (not prefix or not suffix or
            len(prefix) != n or len(suffix) != n or
            "'" in prefix or "'" in suffix or
            "\\" in prefix or "\\" in suffix):
            continue

        # Calculate bucket spans
        p_lo, p_hi = bucket_range(prefix, prefix_boundaries, prefix_bits, suffix=False)
        s_lo, s_hi = bucket_range(suffix, suffix_boundaries, suffix_bits, suffix=True)

        prefix_span = p_hi - p_lo + 1
        suffix_span = s_hi - s_lo + 1
        combined_span = prefix_span * suffix_span

        pairs_with_spans.append((prefix, suffix, prefix_span, suffix_span, combined_span))

    print(f"Found {len(pairs_with_spans):,} unique (prefix, suffix) pairs with spans")
    return pairs_with_spans


def stratified_sample(pairs_with_spans, total_samples=4000):
    """
    Sample queries stratified by combined bucket span.
    Ensures good coverage across different span ranges.
    """
    # Define bucket span bins
    bins = [
        (1, 1, "span=1"),
        (2, 5, "span=2-5"),
        (6, 10, "span=6-10"),
        (11, 20, "span=11-20"),
        (21, 50, "span=21-50"),
        (51, float('inf'), "span>50")
    ]

    # Group pairs by span bin
    binned_pairs = {label: [] for _, _, label in bins}
    for pair in pairs_with_spans:
        prefix, suffix, p_span, s_span, c_span = pair
        for lo, hi, label in bins:
            if lo <= c_span <= hi:
                binned_pairs[label].append(pair)
                break

    # Calculate samples per bin (try to get equal representation)
    # but allocate more to bins with fewer queries
    samples_per_bin = {}
    total_available = sum(len(pairs) for pairs in binned_pairs.values())

    print(f"\nDistribution across bucket span bins:")
    for label in binned_pairs.keys():
        count = len(binned_pairs[label])
        print(f"  {label}: {count:,} pairs")

    # Allocate samples proportionally, but with minimum per bin
    min_per_bin = 50  # Minimum samples per non-empty bin
    remaining_samples = total_samples

    for label, pairs in binned_pairs.items():
        if len(pairs) > 0:
            samples_per_bin[label] = min(min_per_bin, len(pairs), remaining_samples)
            remaining_samples -= samples_per_bin[label]
        else:
            samples_per_bin[label] = 0

    # Distribute remaining samples proportionally
    if remaining_samples > 0:
        for label, pairs in binned_pairs.items():
            if len(pairs) > 0:
                proportion = len(pairs) / total_available
                extra = int(remaining_samples * proportion)
                samples_per_bin[label] += min(extra, len(pairs) - samples_per_bin[label])

    # Sample from each bin
    sampled = []
    print(f"\nStratified sampling:")
    for label, pairs in binned_pairs.items():
        n_samples = min(samples_per_bin[label], len(pairs))
        if n_samples > 0:
            sampled_pairs = random.sample(pairs, n_samples)
            sampled.extend(sampled_pairs)
            print(f"  {label}: sampled {n_samples} / {len(pairs)} pairs")

    print(f"\nTotal sampled: {len(sampled)} pairs")
    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark infix queries with stratified sampling across bucket spans"
    )
    parser.add_argument("--prefix-bits", type=int, default=8, help="Prefix fingerprint bits")
    parser.add_argument("--suffix-bits", type=int, default=8, help="Suffix fingerprint bits")
    parser.add_argument("--n", type=int, default=3, help="N-gram length (default: 3)")
    parser.add_argument("--samples", type=int, default=4000, help="Total number of samples")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs per query")
    parser.add_argument("--reps", type=int, default=5, help="Timed runs per query")
    parser.add_argument("--csv", type=str, default="", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--disable-fsst", action="store_true",
                        help="Disable FSST string compression for pure comparison performance")
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

    # Disable FSST compression if requested
    if args.disable_fsst:
        con.execute("SET enable_fsst_vectors=false")
        print("FSST compression disabled")

    con.execute(f"CREATE TABLE t AS SELECT * FROM read_parquet('{parquet}')")

    # Global cache warming
    print("Warming cache with full table scan...")
    con.execute("SELECT COUNT(*), MIN(title), MAX(title) FROM t").fetchall()

    total_rows = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    print(f"Total rows: {total_rows:,}")
    print(f"Prefix bits: {prefix_bits}, Suffix bits: {suffix_bits}")
    print(f"N-gram length: {args.n}")
    print(f"FSST compression: {'disabled' if args.disable_fsst else 'enabled'}")

    # Extract pairs with bucket spans
    pairs_with_spans = extract_infix_pairs_with_spans(
        con, prefix_boundaries, suffix_boundaries, prefix_bits, suffix_bits, args.n
    )

    # Stratified sampling across bucket spans
    sampled_pairs = stratified_sample(pairs_with_spans, args.samples)

    # Benchmark each pair
    results = []
    print(f"\nBenchmarking {len(sampled_pairs)} queries (warmup={args.warmup}, reps={args.reps})...")

    for i, (prefix_q, suffix_q, p_span, s_span, c_span) in enumerate(sampled_pairs):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Progress: {i+1}/{len(sampled_pairs)} ({100*(i+1)/len(sampled_pairs):.1f}%)")

        # Escape single quotes
        prefix_escaped = prefix_q.replace("'", "''")
        suffix_escaped = suffix_q.replace("'", "''")

        # Full query
        q_full = (
            f"SELECT COUNT(*) FROM t "
            f"WHERE title ILIKE '{prefix_escaped}%' AND title ILIKE '%{suffix_escaped}'"
        )

        # Get match count
        match_count = con.execute(q_full).fetchone()[0]
        selectivity = match_count / total_rows if total_rows > 0 else 0

        # Skip if no matches
        if match_count == 0:
            continue

        # Build fingerprint query
        p_lo, p_hi = bucket_range(prefix_q, prefix_boundaries, prefix_bits, suffix=False)
        s_lo, s_hi = bucket_range(suffix_q, suffix_boundaries, suffix_bits, suffix=True)

        q_fp = (
            f"SELECT COUNT(*) FROM t "
            f"WHERE {prefix_col} BETWEEN {p_lo} AND {p_hi} "
            f"AND {suffix_col} BETWEEN {s_lo} AND {s_hi} "
            f"AND title ILIKE '{prefix_escaped}%' AND title ILIKE '%{suffix_escaped}'"
        )

        # Time both queries
        full_times = time_query(con, q_full, warmup=args.warmup, reps=args.reps)
        fp_times = time_query(con, q_fp, warmup=args.warmup, reps=args.reps)

        full_median = np.median(full_times) * 1000  # ms
        fp_median = np.median(fp_times) * 1000  # ms
        speedup = full_median / fp_median if fp_median > 0 else 0

        results.append({
            "prefix": prefix_q,
            "suffix": suffix_q,
            "match_count": match_count,
            "selectivity": selectivity,
            "prefix_bucket_span": p_span,
            "suffix_bucket_span": s_span,
            "combined_bucket_span": c_span,
            "prune_rate": 1.0,
            "time_full_ms": full_median,
            "time_fp_ms": fp_median,
            "speedup": speedup,
        })

    print(f"\nCompleted {len(results)} queries with matches")

    # Summary stats
    if results:
        speedups = [r["speedup"] for r in results]
        selectivities = [r["selectivity"] for r in results]
        c_spans = [r["combined_bucket_span"] for r in results]

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Queries tested: {len(results)}")
        print(f"Speedup - Mean: {np.mean(speedups):.2f}x, Median: {np.median(speedups):.2f}x")
        print(f"Speedup - Min: {np.min(speedups):.2f}x, Max: {np.max(speedups):.2f}x")
        print(f"Speedup - Geometric mean: {np.exp(np.mean(np.log(speedups))):.2f}x")
        print(f"Selectivity - Mean: {np.mean(selectivities)*100:.6f}%")
        print(f"Combined bucket span - Mean: {np.mean(c_spans):.1f}, Median: {np.median(c_spans):.0f}")
        print(f"Combined bucket span - Min: {np.min(c_spans)}, Max: {np.max(c_spans)}")

    # Write to CSV
    if args.csv:
        output_path = args.csv
    else:
        output_path = f"stratified_infix_p{prefix_bits}_s{suffix_bits}.csv"

    if results:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote results to {output_path}")

    con.close()


if __name__ == "__main__":
    main()
