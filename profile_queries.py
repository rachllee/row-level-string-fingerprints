"""
Profile sampled queries in detail to understand performance characteristics.

Samples queries from different speedup zones and shows:
- Full scan vs fingerprint scan timing
- Rows in bucket vs actual matches
- Bucket efficiency and false positive ratio
"""

import pandas as pd
import duckdb
import time
import numpy as np
import argparse

def profile_query(con, query_pattern, bits=8):
    """
    Profile a single query in detail.
    Returns detailed metrics about the query execution.
    """
    code_col = f"q{bits}_prefix"

    # Escape single quotes for SQL
    escaped_pattern = query_pattern.replace("'", "''")

    # Find the fingerprint code by looking up a matching row
    # (can't recompute the hash - need to look it up from actual data)
    # Use ILIKE for case-insensitive matching (same as benchmark)
    fp_code_result = con.execute(f"""
        SELECT {code_col}
        FROM t
        WHERE title ILIKE '{escaped_pattern}%'
        LIMIT 1
    """).fetchone()

    if fp_code_result is None:
        # No matches - this shouldn't happen for queries in the benchmark
        return None

    prefix_hash = fp_code_result[0]

    # 1. Full scan timing (use ILIKE like the benchmark does)
    full_times = []
    for _ in range(5):
        start = time.perf_counter()
        full_result = con.execute(f"""
            SELECT COUNT(*)
            FROM t
            WHERE title ILIKE '{escaped_pattern}%'
        """).fetchone()[0]
        full_times.append(time.perf_counter() - start)
    full_time = np.median(full_times)

    # 2. Fingerprint scan timing (use ILIKE like the benchmark does)
    fp_times = []
    for _ in range(5):
        start = time.perf_counter()
        fp_result = con.execute(f"""
            SELECT COUNT(*)
            FROM t
            WHERE {code_col} = {prefix_hash}
              AND title ILIKE '{escaped_pattern}%'
        """).fetchone()[0]
        fp_times.append(time.perf_counter() - start)
    fp_time = np.median(fp_times)

    # 3. Get bucket size (rows matching fingerprint)
    bucket_size = con.execute(f"""
        SELECT COUNT(*)
        FROM t
        WHERE {code_col} = {prefix_hash}
    """).fetchone()[0]

    # 4. Get actual matches
    actual_matches = full_result

    # 5. Calculate metrics
    total_rows = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    selectivity = (actual_matches / total_rows) * 100
    prune_rate = ((total_rows - bucket_size) / total_rows) * 100
    bucket_efficiency = (actual_matches / bucket_size * 100) if bucket_size > 0 else 0
    false_positives = bucket_size - actual_matches
    fp_ratio = (false_positives / bucket_size * 100) if bucket_size > 0 else 0
    speedup = full_time / fp_time if fp_time > 0 else 0

    return {
        'query': query_pattern,
        'full_time_ms': full_time * 1000,
        'fp_time_ms': fp_time * 1000,
        'speedup': speedup,
        'total_rows': total_rows,
        'bucket_size': bucket_size,
        'actual_matches': actual_matches,
        'selectivity_pct': selectivity,
        'prune_rate_pct': prune_rate,
        'bucket_efficiency_pct': bucket_efficiency,
        'false_positives': false_positives,
        'fp_ratio_pct': fp_ratio
    }

def main():
    parser = argparse.ArgumentParser(description="Profile sampled queries in detail")
    parser.add_argument("--csv", required=True, help="Input CSV with benchmark results")
    parser.add_argument("--bits", type=int, default=8, help="Fingerprint bit width")
    parser.add_argument("--samples", type=int, default=3, help="Samples per zone")
    args = parser.parse_args()

    # Load benchmark results
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} queries from {args.csv}\n")

    # Convert selectivity to percentage for display
    df['selectivity_pct'] = df['selectivity'] * 100

    # Define speedup zones
    zones = [
        ("High speedup (>13x)", df[df['speedup'] > 13]),
        ("Uncanny valley (5-8x)", df[(df['speedup'] >= 5) & (df['speedup'] <= 8)]),
        ("Low speedup (2-4x)", df[(df['speedup'] >= 2) & (df['speedup'] <= 4)])
    ]

    # Connect to DuckDB and load data
    bits = args.bits
    parquet = f"title_strs_prefix_b{bits}.parquet"

    con = duckdb.connect()
    con.execute("PRAGMA threads=1")
    con.execute("SET enable_fsst_vectors=false")
    con.execute(f"CREATE TABLE t AS SELECT * FROM read_parquet('{parquet}')")

    print("Warming cache with full table scan...")
    con.execute("SELECT COUNT(*), MIN(title), MAX(title) FROM t").fetchall()

    # Sample and profile queries from each zone
    all_profiles = []

    for zone_name, zone_df in zones:
        if len(zone_df) == 0:
            print(f"\n{zone_name}: No queries in this range")
            continue

        print(f"\n{'='*70}")
        print(f"{zone_name}")
        print(f"{'='*70}")
        print(f"Zone size: {len(zone_df)} queries")
        print(f"Selectivity range: {zone_df['selectivity_pct'].min():.6f}% - {zone_df['selectivity_pct'].max():.6f}%")

        # Sample queries (or use all if fewer than samples)
        n_samples = min(args.samples, len(zone_df))
        sampled = zone_df.sample(n=n_samples, random_state=42)

        print(f"\nProfiling {n_samples} sampled queries:\n")

        for idx, row in sampled.iterrows():
            query = row['ngram']  # Column is named 'ngram' not 'query'
            print(f"Query: '{query}%'")
            print(f"  Original benchmark speedup: {row['speedup']:.2f}x")

            # Profile the query
            profile = profile_query(con, query, bits)
            if profile is None:
                print(f"  Skipping - no matches found")
                continue
            all_profiles.append({**profile, 'zone': zone_name})

            # Print detailed metrics
            print(f"  Full scan time:      {profile['full_time_ms']:8.2f} ms")
            print(f"  Fingerprint time:    {profile['fp_time_ms']:8.2f} ms")
            print(f"  Measured speedup:    {profile['speedup']:8.2f}x")
            print(f"  Total rows:          {profile['total_rows']:,}")
            print(f"  Bucket size:         {profile['bucket_size']:,}")
            print(f"  Actual matches:      {profile['actual_matches']:,}")
            print(f"  False positives:     {profile['false_positives']:,}")
            print(f"  Selectivity:         {profile['selectivity_pct']:.6f}%")
            print(f"  Prune rate:          {profile['prune_rate_pct']:.4f}%")
            print(f"  Bucket efficiency:   {profile['bucket_efficiency_pct']:.4f}%")
            print(f"  FP ratio:            {profile['fp_ratio_pct']:.4f}%")
            print()

    # Save detailed profiles
    profile_df = pd.DataFrame(all_profiles)
    output_csv = args.csv.replace('.csv', '_profiles.csv')
    profile_df.to_csv(output_csv, index=False)
    print(f"\n{'='*70}")
    print(f"Saved detailed profiles to {output_csv}")

    # Summary statistics by zone
    print(f"\n{'='*70}")
    print("SUMMARY BY ZONE")
    print(f"{'='*70}\n")

    for zone_name in profile_df['zone'].unique():
        zone_profiles = profile_df[profile_df['zone'] == zone_name]
        print(f"{zone_name}:")
        print(f"  Mean bucket efficiency:  {zone_profiles['bucket_efficiency_pct'].mean():.4f}%")
        print(f"  Mean FP ratio:           {zone_profiles['fp_ratio_pct'].mean():.4f}%")
        print(f"  Mean bucket size:        {zone_profiles['bucket_size'].mean():,.0f}")
        print(f"  Mean actual matches:     {zone_profiles['actual_matches'].mean():,.0f}")
        print()

    con.close()

if __name__ == "__main__":
    main()
