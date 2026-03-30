"""
Benchmark ALL distinct 3-character prefixes (or suffixes) from the data.

This provides an unbiased measurement of fingerprint speedup across the actual
data distribution, and allows plotting speedup vs. selectivity.
"""

import argparse
import csv
import os
import time
import random

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


def escape_sql_string(s):
    """Escape single quotes for SQL LIKE patterns."""
    return s.replace("'", "''")


def extract_distinct_ngrams(con, n=3, suffix=False, sample_limit=None):
    """Extract all distinct n-character prefixes or suffixes from the data."""
    if suffix:
        # Last n characters
        sql = f"""
            SELECT DISTINCT LOWER(SUBSTRING(title, LENGTH(title) - {n-1}, {n})) as ngram
            FROM t
            WHERE LENGTH(title) >= {n}
        """
    else:
        # First n characters
        sql = f"""
            SELECT DISTINCT LOWER(SUBSTRING(title, 1, {n})) as ngram
            FROM t
            WHERE LENGTH(title) >= {n}
        """

    result = con.execute(sql).fetchall()
    # Filter out ngrams with problematic characters for SQL
    ngrams = []
    for r in result:
        if r[0] and len(r[0]) == n:
            # Skip ngrams with characters that could cause SQL issues
            if "'" not in r[0] and "\\" not in r[0] and "%" not in r[0]:
                ngrams.append(r[0])

    if sample_limit and len(ngrams) > sample_limit:
        random.seed(42)  # Reproducible sampling
        ngrams = random.sample(ngrams, sample_limit)

    return sorted(ngrams)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all distinct 3-char prefixes/suffixes"
    )
    parser.add_argument("--bits", type=int, default=8, help="Fingerprint bit width")
    parser.add_argument("--suffix", action="store_true", help="Test suffixes instead of prefixes")
    parser.add_argument("--n", type=int, default=3, help="N-gram length (default: 3)")
    parser.add_argument("--sample", type=int, default=0, help="Sample this many n-grams (0 = all)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per query")
    parser.add_argument("--reps", type=int, default=5, help="Timed runs per query")
    parser.add_argument("--csv", type=str, default="", help="Output CSV path")
    parser.add_argument("--source", type=str, default="table", choices=["table", "view"],
                        help="Use table or view (default: table)")
    parser.add_argument("--disable-fsst", action="store_true",
                        help="Disable FSST string compression for pure comparison performance")
    args = parser.parse_args()

    bits = args.bits
    mode = "suffix" if args.suffix else "prefix"
    parquet = f"title_strs_{mode}_b{bits}.parquet"
    boundaries_npy = f"q{bits}_{mode}_boundaries.npy"
    code_col = f"q{bits}_{mode}"

    if not os.path.exists(parquet):
        raise FileNotFoundError(f"Missing {parquet}. Run build.py --bits {bits} first.")

    boundaries = np.load(boundaries_npy) if os.path.exists(boundaries_npy) else None

    con = duckdb.connect()
    con.execute("PRAGMA threads=1")
    con.execute("PRAGMA enable_object_cache=true")

    # Disable FSST compression if requested
    if args.disable_fsst:
        con.execute("SET enable_fsst_vectors=false")
        print("FSST compression disabled")

    # Load data
    if args.source == "table":
        con.execute(f"CREATE TABLE t AS SELECT * FROM read_parquet('{parquet}')")
    else:
        con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{parquet}')")

    # Global cache warming: Ensure all data is in memory
    print("Warming cache with full table scan...")
    con.execute("SELECT COUNT(*), MIN(title), MAX(title) FROM t").fetchall()

    total_rows = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    print(f"Total rows: {total_rows:,}")
    print(f"Mode: {mode}, Bits: {bits}, Source: {args.source}")
    print(f"N-gram length: {args.n}")
    print(f"FSST compression: {'disabled' if args.disable_fsst else 'enabled'}")

    # Extract distinct n-grams
    print(f"\nExtracting distinct {args.n}-char {mode}es...")
    sample_limit = args.sample if args.sample > 0 else None
    ngrams = extract_distinct_ngrams(con, n=args.n, suffix=args.suffix, sample_limit=sample_limit)
    print(f"Found {len(ngrams):,} distinct {args.n}-char {mode}es")

    if sample_limit:
        print(f"(Sampled {sample_limit} for benchmarking)")

    # Benchmark each n-gram
    results = []
    print(f"\nBenchmarking {len(ngrams)} queries (warmup={args.warmup}, reps={args.reps})...")

    for i, ngram in enumerate(ngrams):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Progress: {i+1}/{len(ngrams)} ({100*(i+1)/len(ngrams):.1f}%)")

        # Calculate selectivity (what fraction of rows match this query)
        if args.suffix:
            like_pattern = f"%{ngram}"
            q_full = f"SELECT COUNT(*) FROM t WHERE title ILIKE '{like_pattern}'"
        else:
            like_pattern = f"{ngram}%"
            q_full = f"SELECT COUNT(*) FROM t WHERE title ILIKE '{like_pattern}'"

        # Get match count and selectivity
        match_count = con.execute(q_full).fetchone()[0]
        selectivity = match_count / total_rows if total_rows > 0 else 0

        # Skip if no matches (query would be meaningless)
        if match_count == 0:
            continue

        # Build fingerprint-filtered query
        lo, hi = bucket_range(ngram, boundaries, bits, suffix=args.suffix)
        q_fp = (
            f"SELECT COUNT(*) FROM t "
            f"WHERE {code_col} BETWEEN {lo} AND {hi} "
            f"AND title ILIKE '{like_pattern}'"
        )

        # Time both queries
        full_times = time_query(con, q_full, warmup=args.warmup, reps=args.reps)
        fp_times = time_query(con, q_fp, warmup=args.warmup, reps=args.reps)

        full_med = float(np.median(full_times))
        fp_med = float(np.median(fp_times))
        speedup = full_med / fp_med if fp_med > 0 else 0

        # Calculate pruning rate
        rows_in_buckets = con.execute(
            f"SELECT COUNT(*) FROM t WHERE {code_col} BETWEEN {lo} AND {hi}"
        ).fetchone()[0]
        prune_rate = 1.0 - (rows_in_buckets / total_rows) if total_rows > 0 else 0

        results.append({
            "ngram": ngram,
            "match_count": match_count,
            "selectivity": selectivity,
            "bucket_lo": lo,
            "bucket_hi": hi,
            "bucket_span": hi - lo + 1,
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

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Queries tested: {len(results)}")
        print(f"Speedup - Mean: {np.mean(speedups):.2f}x, Median: {np.median(speedups):.2f}x")
        print(f"Speedup - Min: {min(speedups):.2f}x, Max: {max(speedups):.2f}x")
        print(f"Speedup - Geometric mean: {np.exp(np.mean(np.log(speedups))):.2f}x")
        print(f"Selectivity - Mean: {np.mean(selectivities)*100:.4f}%")
        print(f"Selectivity - Min: {min(selectivities)*100:.6f}%, Max: {max(selectivities)*100:.2f}%")

    # Write CSV
    if args.csv and results:
        csv_path = args.csv
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"\nWrote results to {csv_path}")

    con.close()


if __name__ == "__main__":
    main()
