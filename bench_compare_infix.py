"""
Head-to-head benchmark: two-column BETWEEN vs single q16_infix IN/= approach.

For each query (prefix, suffix):
  - full scan:     WHERE title ILIKE 'prefix%' AND title ILIKE '%suffix'
  - two-column:    WHERE q8_prefix BETWEEN p_lo AND p_hi
                     AND q8_suffix BETWEEN s_lo AND s_hi
                     AND title ILIKE ...
  - combined-16:   WHERE q16_infix IN (p<<8|s, ...) AND title ILIKE ...
                   (single equality when span=1x1)

Uses the same stratified sample across bucket span ranges so results are comparable.
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
    print(f"Extracting real {n}-char (prefix, suffix) pairs with bucket spans...")

    sql = f"""
        SELECT DISTINCT
            LOWER(SUBSTRING(title, 1, {n})) as prefix,
            LOWER(SUBSTRING(title, LENGTH(title) - {n-1}, {n})) as suffix
        FROM two_col
        WHERE LENGTH(title) >= {n}
    """

    result = con.execute(sql).fetchall()

    pairs_with_spans = []
    for prefix, suffix in result:
        if (not prefix or not suffix or
                len(prefix) != n or len(suffix) != n or
                "'" in prefix or "'" in suffix or
                "\\" in prefix or "\\" in suffix):
            continue

        p_lo, p_hi = bucket_range(prefix, prefix_boundaries, prefix_bits, suffix=False)
        s_lo, s_hi = bucket_range(suffix, suffix_boundaries, suffix_bits, suffix=True)

        prefix_span = p_hi - p_lo + 1
        suffix_span = s_hi - s_lo + 1
        combined_span = prefix_span * suffix_span

        pairs_with_spans.append((prefix, suffix, p_lo, p_hi, s_lo, s_hi,
                                  prefix_span, suffix_span, combined_span))

    print(f"Found {len(pairs_with_spans):,} unique pairs")
    return pairs_with_spans


def stratified_sample(pairs_with_spans, total_samples=4000, seed=42):
    bins = [
        (1, 1,          "span=1"),
        (2, 5,          "span=2-5"),
        (6, 10,         "span=6-10"),
        (11, 20,        "span=11-20"),
        (21, 50,        "span=21-50"),
        (51, float('inf'), "span>50"),
    ]

    binned = {label: [] for _, _, label in bins}
    for pair in pairs_with_spans:
        c_span = pair[8]
        for lo, hi, label in bins:
            if lo <= c_span <= hi:
                binned[label].append(pair)
                break

    print("\nDistribution across bucket span bins:")
    for label in binned:
        print(f"  {label}: {len(binned[label]):,} pairs")

    min_per_bin = 50
    samples_per_bin = {}
    remaining = total_samples
    total_available = sum(len(v) for v in binned.values())

    for label, pairs in binned.items():
        if pairs:
            alloc = min(min_per_bin, len(pairs), remaining)
            samples_per_bin[label] = alloc
            remaining -= alloc
        else:
            samples_per_bin[label] = 0

    if remaining > 0:
        for label, pairs in binned.items():
            if pairs:
                extra = int(remaining * len(pairs) / total_available)
                samples_per_bin[label] += min(extra, len(pairs) - samples_per_bin[label])

    rng = random.Random(seed)
    sampled = []
    print("\nStratified sampling:")
    for label, pairs in binned.items():
        n_take = min(samples_per_bin[label], len(pairs))
        if n_take > 0:
            chosen = rng.sample(pairs, n_take)
            sampled.extend(chosen)
            print(f"  {label}: {n_take} / {len(pairs)}")

    print(f"\nTotal sampled: {len(sampled)}")
    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Compare two-column BETWEEN vs combined q16_infix IN/= fingerprint"
    )
    parser.add_argument("--prefix-bits", type=int, default=8)
    parser.add_argument("--suffix-bits", type=int, default=8)
    parser.add_argument("--n", type=int, default=3, help="N-gram length (default: 3)")
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--csv", type=str, default="", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-fsst", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    pb = args.prefix_bits
    sb = args.suffix_bits

    two_col_parquet  = f"title_strs_infix_p{pb}_s{sb}.parquet"
    combined_parquet = f"title_strs_infix16_p{pb}_s{sb}.parquet"
    prefix_boundaries_npy = f"q{pb}_prefix_boundaries.npy"
    suffix_boundaries_npy = f"q{sb}_suffix_boundaries.npy"
    prefix_col = f"q{pb}_prefix"
    suffix_col  = f"q{sb}_suffix"

    for path in [two_col_parquet, combined_parquet]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run build_infix.py first.")

    prefix_boundaries = np.load(prefix_boundaries_npy)
    suffix_boundaries = np.load(suffix_boundaries_npy)

    con = duckdb.connect()
    con.execute("PRAGMA threads=1")
    con.execute("PRAGMA enable_object_cache=true")
    if args.disable_fsst:
        con.execute("SET enable_fsst_vectors=false")
        print("FSST disabled")

    print("Loading tables...")
    con.execute(f"CREATE TABLE two_col  AS SELECT * FROM read_parquet('{two_col_parquet}')")
    con.execute(f"CREATE TABLE combined AS SELECT * FROM read_parquet('{combined_parquet}')")

    print("Warming cache...")
    con.execute("SELECT COUNT(*), MIN(title), MAX(title) FROM two_col").fetchall()
    con.execute("SELECT COUNT(*), MIN(title), MAX(title) FROM combined").fetchall()

    total_rows = con.execute("SELECT COUNT(*) FROM two_col").fetchone()[0]
    print(f"Total rows: {total_rows:,}")

    # Extract and stratify
    pairs = extract_infix_pairs_with_spans(
        con, prefix_boundaries, suffix_boundaries, pb, sb, args.n
    )
    sampled = stratified_sample(pairs, args.samples, args.seed)

    results = []
    print(f"\nBenchmarking {len(sampled)} queries "
          f"(warmup={args.warmup}, reps={args.reps})...")

    for i, (prefix_q, suffix_q, p_lo, p_hi, s_lo, s_hi,
            p_span, s_span, c_span) in enumerate(sampled):

        if (i + 1) % 200 == 0 or i == 0:
            print(f"  {i+1}/{len(sampled)} ({100*(i+1)/len(sampled):.0f}%)")

        pe = prefix_q.replace("'", "''")
        se = suffix_q.replace("'", "''")

        # 1. Full scan (on either table — same title column)
        q_full = (
            f"SELECT COUNT(*) FROM two_col "
            f"WHERE title ILIKE '{pe}%' AND title ILIKE '%{se}'"
        )
        match_count = con.execute(q_full).fetchone()[0]
        if match_count == 0:
            continue
        selectivity = match_count / total_rows

        # 2. Two-column BETWEEN
        q_two = (
            f"SELECT COUNT(*) FROM two_col "
            f"WHERE {prefix_col} BETWEEN {p_lo} AND {p_hi} "
            f"AND {suffix_col} BETWEEN {s_lo} AND {s_hi} "
            f"AND title ILIKE '{pe}%' AND title ILIKE '%{se}'"
        )

        # 3. Combined q16_infix — bit-shift to recover prefix/suffix ranges
        q_combined = (
            f"SELECT COUNT(*) FROM combined "
            f"WHERE (q16_infix >> {sb}) BETWEEN {p_lo} AND {p_hi} "
            f"AND (q16_infix & {(1 << sb) - 1}) BETWEEN {s_lo} AND {s_hi} "
            f"AND title ILIKE '{pe}%' AND title ILIKE '%{se}'"
        )

        full_times    = time_query(con, q_full,     args.warmup, args.reps)
        two_times     = time_query(con, q_two,      args.warmup, args.reps)
        combined_times = time_query(con, q_combined, args.warmup, args.reps)

        full_ms     = np.median(full_times) * 1000
        two_ms      = np.median(two_times) * 1000
        combined_ms = np.median(combined_times) * 1000

        results.append({
            "prefix": prefix_q,
            "suffix": suffix_q,
            "match_count": match_count,
            "selectivity": selectivity,
            "prefix_bucket_span": p_span,
            "suffix_bucket_span": s_span,
            "combined_bucket_span": c_span,
            "time_full_ms": full_ms,
            "time_two_col_ms": two_ms,
            "time_combined16_ms": combined_ms,
            "speedup_two_col": full_ms / two_ms if two_ms > 0 else 0,
            "speedup_combined16": full_ms / combined_ms if combined_ms > 0 else 0,
        })

    print(f"\nCompleted {len(results)} queries with matches")

    if results:
        su_two = [r["speedup_two_col"] for r in results]
        su_c16 = [r["speedup_combined16"] for r in results]

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<30} {'Two-column':>12} {'Combined-16':>12}")
        print("-" * 60)
        print(f"{'Geometric mean speedup':<30} {np.exp(np.mean(np.log(su_two))):>11.2f}x {np.exp(np.mean(np.log(su_c16))):>11.2f}x")
        print(f"{'Median speedup':<30} {np.median(su_two):>11.2f}x {np.median(su_c16):>11.2f}x")
        print(f"{'Mean speedup':<30} {np.mean(su_two):>11.2f}x {np.mean(su_c16):>11.2f}x")
        print(f"{'Min speedup':<30} {np.min(su_two):>11.2f}x {np.min(su_c16):>11.2f}x")
        print(f"{'Max speedup':<30} {np.max(su_two):>11.2f}x {np.max(su_c16):>11.2f}x")
        print("=" * 60)

        # Breakdown by span
        print("\nSpeedup by combined bucket span:")
        span_bins = [(1, 1, "span=1"), (2, 5, "span=2-5"), (6, 10, "span=6-10"),
                     (11, 20, "span=11-20"), (21, 50, "span=21-50"), (51, 9999, "span>50")]
        print(f"  {'Span':<12} {'N':>5}  {'Two-col':>9}  {'Comb-16':>9}")
        for lo, hi, label in span_bins:
            subset = [r for r in results if lo <= r["combined_bucket_span"] <= hi]
            if subset:
                gm_two = np.exp(np.mean(np.log([r["speedup_two_col"] for r in subset])))
                gm_c16 = np.exp(np.mean(np.log([r["speedup_combined16"] for r in subset])))
                print(f"  {label:<12} {len(subset):>5}  {gm_two:>8.2f}x  {gm_c16:>8.2f}x")

    out_path = args.csv or f"compare_infix_p{pb}_s{sb}.csv"
    if results:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote results to {out_path}")

    con.close()


if __name__ == "__main__":
    main()
