"""
Benchmark combined prefix+suffix (infix) fingerprint queries.

Compares:
- Full scan: WHERE title LIKE 'prefix%' AND title LIKE '%suffix'
- FP filtered: WHERE q8_prefix BETWEEN ... AND q8_suffix BETWEEN ... AND title LIKE ...
"""

import argparse
import csv
import os
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
    """Get bucket range for a query string."""
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


def time_query(con, sql, warmup=1, reps=10):
    """Time a query with warmup runs."""
    for _ in range(warmup):
        con.execute(sql).fetchone()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        con.execute(sql).fetchone()
        times.append(time.perf_counter() - t0)
    return times


def summarize_times(times):
    p10 = float(np.percentile(times, 10))
    p90 = float(np.percentile(times, 90))
    med = float(np.median(times))
    iqr = float(np.percentile(times, 75) - np.percentile(times, 25))
    return med, p10, p90, iqr


def main():
    parser = argparse.ArgumentParser(description="Benchmark combined prefix+suffix queries")
    parser.add_argument("--prefix-bits", type=int, default=8, help="Bit width for prefix (default: 8)")
    parser.add_argument("--suffix-bits", type=int, default=8, help="Bit width for suffix (default: 8)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per query")
    parser.add_argument("--reps", type=int, default=10, help="Timed runs per query")
    parser.add_argument("--csv", type=str, default="", help="Output CSV path for per-run timings (compatible with summarize_bench.py)")
    args = parser.parse_args()

    prefix_bits = args.prefix_bits
    suffix_bits = args.suffix_bits

    parquet = f"title_strs_infix_p{prefix_bits}_s{suffix_bits}.parquet"
    prefix_boundaries_npy = f"q{prefix_bits}_prefix_boundaries.npy"
    suffix_boundaries_npy = f"q{suffix_bits}_suffix_boundaries.npy"

    if not os.path.exists(parquet):
        print(f"Error: {parquet} not found. Run build_infix.py first.")
        return

    # Load boundaries
    prefix_boundaries = np.load(prefix_boundaries_npy) if os.path.exists(prefix_boundaries_npy) else None
    suffix_boundaries = np.load(suffix_boundaries_npy) if os.path.exists(suffix_boundaries_npy) else None

    prefix_col = f"q{prefix_bits}_prefix"
    suffix_col = f"q{suffix_bits}_suffix"

    # Combined queries: (prefix, suffix) pairs
    queries = [
        ("the", "a"),       # titles starting with "the" and ending with "a"
        ("the", "e"),       # titles starting with "the" and ending with "e"
        ("a", "s"),         # titles starting with "a" and ending with "s"
        ("in", "on"),       # titles starting with "in" and ending with "on"
        ("re", "ed"),       # titles starting with "re" and ending with "ed"
        ("2", "1"),         # titles starting with "2" and ending with "1"
    ]

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA enable_object_cache=true")

    print(f"Parquet: {parquet}")
    print(f"Prefix column: {prefix_col} ({prefix_bits} bits)")
    print(f"Suffix column: {suffix_col} ({suffix_bits} bits)")
    print(f"Warmup: {args.warmup}, Reps: {args.reps}")
    print("=" * 70)

    # For CSV output (compatible with summarize_bench.py)
    # We use prefix_bits + suffix_bits as "bits" to represent total fingerprint size
    csv_rows = [] if args.csv else None
    total_bits = prefix_bits + suffix_bits

    # Test with both view and table
    for source_label in ["view", "table"]:
        con.execute("DROP VIEW IF EXISTS t")
        con.execute("DROP TABLE IF EXISTS t")

        if source_label == "table":
            con.execute(f"CREATE TABLE t AS SELECT * FROM read_parquet('{parquet}')")
        else:
            con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{parquet}')")

        print(f"\n[{source_label.upper()}]")
        print("-" * 70)

        speedups = []

        for prefix_q, suffix_q in queries:
            # Get bucket ranges
            p_lo, p_hi = bucket_range(prefix_q, prefix_boundaries, prefix_bits, suffix=False)
            s_lo, s_hi = bucket_range(suffix_q, suffix_boundaries, suffix_bits, suffix=True)

            # Full scan query
            q_full = (
                f"SELECT COUNT(*) FROM t "
                f"WHERE title LIKE '{prefix_q}%' AND title LIKE '%{suffix_q}'"
            )

            # Fingerprint-filtered query
            q_fp = (
                f"SELECT COUNT(*) FROM t "
                f"WHERE {prefix_col} BETWEEN {p_lo} AND {p_hi} "
                f"AND {suffix_col} BETWEEN {s_lo} AND {s_hi} "
                f"AND title LIKE '{prefix_q}%' AND title LIKE '%{suffix_q}'"
            )

            # Get the actual count
            count = con.execute(q_full).fetchone()[0]

            # Time both queries
            full_times = time_query(con, q_full, warmup=args.warmup, reps=args.reps)
            fp_times = time_query(con, q_fp, warmup=args.warmup, reps=args.reps)

            # Record to CSV if requested
            query_label = f"{prefix_q}%..%{suffix_q}"
            if csv_rows is not None:
                for i, t in enumerate(full_times):
                    csv_rows.append((total_bits, source_label, query_label, "full", i, t))
                for i, t in enumerate(fp_times):
                    csv_rows.append((total_bits, source_label, query_label, "fp_exact", i, t))

            full_med, full_p10, full_p90, full_iqr = summarize_times(full_times)
            fp_med, fp_p10, fp_p90, fp_iqr = summarize_times(fp_times)
            speedup = full_med / fp_med if fp_med > 0 else 0
            speedups.append(speedup)

            print(f"\nQuery: '{prefix_q}%...%{suffix_q}'  (matches: {count:,})")
            print(f"  Prefix buckets: [{p_lo}, {p_hi}]  Suffix buckets: [{s_lo}, {s_hi}]")
            print(
                f"  Full scan: median {full_med*1000:.2f} ms "
                f"(P10 {full_p10*1000:.2f}, P90 {full_p90*1000:.2f})"
            )
            print(
                f"  FP+exact:  median {fp_med*1000:.2f} ms "
                f"(P10 {fp_p10*1000:.2f}, P90 {fp_p90*1000:.2f})"
            )
            print(f"  Speedup: {speedup:.2f}x")

        # Summary
        if speedups:
            geo_mean = float(np.exp(np.mean(np.log(speedups))))
            print(f"\n[{source_label.upper()}] Geometric mean speedup: {geo_mean:.2f}x")

    # Write CSV
    if args.csv and csv_rows is not None:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bits", "source", "prefix", "query", "run", "time_s"])
            w.writerows(csv_rows)
        print(f"\nWrote per-run timings to {args.csv}")


if __name__ == "__main__":
    main()
