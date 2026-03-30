"""
Profile sampled infix queries in detail to understand performance characteristics.
"""

import pandas as pd
import duckdb
import time
import numpy as np
import argparse


def profile_infix_query(con, prefix, suffix, prefix_bits=8, suffix_bits=8):
    """
    Profile a single infix query in detail.
    Returns detailed metrics about the query execution.
    """
    prefix_col = f"q{prefix_bits}_prefix"
    suffix_col = f"q{suffix_bits}_suffix"

    # Escape single quotes for SQL
    prefix_escaped = prefix.replace("'", "''")
    suffix_escaped = suffix.replace("'", "''")

    # Find the fingerprint codes by looking up a matching row
    fp_code_result = con.execute(f"""
        SELECT {prefix_col}, {suffix_col}
        FROM t
        WHERE title ILIKE '{prefix_escaped}%' AND title ILIKE '%{suffix_escaped}'
        LIMIT 1
    """).fetchone()

    if fp_code_result is None:
        # No matches
        return None

    prefix_hash, suffix_hash = fp_code_result

    # 1. Full scan timing
    full_times = []
    for _ in range(5):
        start = time.perf_counter()
        full_result = con.execute(f"""
            SELECT COUNT(*)
            FROM t
            WHERE title ILIKE '{prefix_escaped}%' AND title ILIKE '%{suffix_escaped}'
        """).fetchone()[0]
        full_times.append(time.perf_counter() - start)
    full_time = np.median(full_times)

    # 2. Fingerprint scan timing
    fp_times = []
    for _ in range(5):
        start = time.perf_counter()
        fp_result = con.execute(f"""
            SELECT COUNT(*)
            FROM t
            WHERE {prefix_col} = {prefix_hash}
              AND {suffix_col} = {suffix_hash}
              AND title ILIKE '{prefix_escaped}%' AND title ILIKE '%{suffix_escaped}'
        """).fetchone()[0]
        fp_times.append(time.perf_counter() - start)
    fp_time = np.median(fp_times)

    # 3. Get bucket sizes
    prefix_bucket_size = con.execute(f"""
        SELECT COUNT(*)
        FROM t
        WHERE {prefix_col} = {prefix_hash}
    """).fetchone()[0]

    suffix_bucket_size = con.execute(f"""
        SELECT COUNT(*)
        FROM t
        WHERE {suffix_col} = {suffix_hash}
    """).fetchone()[0]

    combined_bucket_size = con.execute(f"""
        SELECT COUNT(*)
        FROM t
        WHERE {prefix_col} = {prefix_hash}
          AND {suffix_col} = {suffix_hash}
    """).fetchone()[0]

    # 4. Get actual matches
    actual_matches = full_result

    # 5. Calculate metrics
    total_rows = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    selectivity = (actual_matches / total_rows) * 100
    prune_rate = ((total_rows - combined_bucket_size) / total_rows) * 100
    bucket_efficiency = (actual_matches / combined_bucket_size * 100) if combined_bucket_size > 0 else 0
    false_positives = combined_bucket_size - actual_matches
    fp_ratio = (false_positives / combined_bucket_size * 100) if combined_bucket_size > 0 else 0
    speedup = full_time / fp_time if fp_time > 0 else 0

    return {
        'prefix': prefix,
        'suffix': suffix,
        'full_time_ms': full_time * 1000,
        'fp_time_ms': fp_time * 1000,
        'speedup': speedup,
        'total_rows': total_rows,
        'prefix_bucket_size': prefix_bucket_size,
        'suffix_bucket_size': suffix_bucket_size,
        'combined_bucket_size': combined_bucket_size,
        'actual_matches': actual_matches,
        'selectivity_pct': selectivity,
        'prune_rate_pct': prune_rate,
        'bucket_efficiency_pct': bucket_efficiency,
        'false_positives': false_positives,
        'fp_ratio_pct': fp_ratio
    }


def main():
    parser = argparse.ArgumentParser(description="Profile sampled infix queries in detail")
    parser.add_argument("--csv", required=True, help="Input CSV with benchmark results")
    parser.add_argument("--prefix-bits", type=int, default=8, help="Prefix fingerprint bit width")
    parser.add_argument("--suffix-bits", type=int, default=8, help="Suffix fingerprint bit width")
    parser.add_argument("--samples", type=int, default=3, help="Samples per zone")
    args = parser.parse_args()

    # Load benchmark results
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} queries from {args.csv}\n")

    # Convert selectivity to percentage
    df['selectivity_pct'] = df['selectivity'] * 100

    # Define speedup zones
    zones = [
        ("High speedup (>14x)", df[df['speedup'] > 14]),
        ("Medium speedup (12-14x)", df[(df['speedup'] >= 12) & (df['speedup'] <= 14)]),
        ("Low speedup (<12x)", df[df['speedup'] < 12])
    ]

    # Connect to DuckDB and load data
    prefix_bits = args.prefix_bits
    suffix_bits = args.suffix_bits
    parquet = f"title_strs_infix_p{prefix_bits}_s{suffix_bits}.parquet"

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

        # Sample queries
        n_samples = min(args.samples, len(zone_df))
        sampled = zone_df.sample(n=n_samples, random_state=42)

        print(f"\nProfiling {n_samples} sampled queries:\n")

        for idx, row in sampled.iterrows():
            prefix = row['prefix']
            suffix = row['suffix']
            print(f"Query: '{prefix}%...%{suffix}'")
            print(f"  Original benchmark speedup: {row['speedup']:.2f}x")

            # Profile the query
            profile = profile_infix_query(con, prefix, suffix, prefix_bits, suffix_bits)
            if profile is None:
                print(f"  Skipping - no matches found")
                continue
            all_profiles.append({**profile, 'zone': zone_name})

            # Print detailed metrics
            print(f"  Full scan time:       {profile['full_time_ms']:8.2f} ms")
            print(f"  Fingerprint time:     {profile['fp_time_ms']:8.2f} ms")
            print(f"  Measured speedup:     {profile['speedup']:8.2f}x")
            print(f"  Total rows:           {profile['total_rows']:,}")
            print(f"  Prefix bucket:        {profile['prefix_bucket_size']:,}")
            print(f"  Suffix bucket:        {profile['suffix_bucket_size']:,}")
            print(f"  Combined bucket:      {profile['combined_bucket_size']:,}")
            print(f"  Actual matches:       {profile['actual_matches']:,}")
            print(f"  False positives:      {profile['false_positives']:,}")
            print(f"  Selectivity:          {profile['selectivity_pct']:.6f}%")
            print(f"  Prune rate:           {profile['prune_rate_pct']:.4f}%")
            print(f"  Bucket efficiency:    {profile['bucket_efficiency_pct']:.4f}%")
            print(f"  FP ratio:             {profile['fp_ratio_pct']:.4f}%")
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
        print(f"  Mean bucket efficiency:   {zone_profiles['bucket_efficiency_pct'].mean():.4f}%")
        print(f"  Mean FP ratio:            {zone_profiles['fp_ratio_pct'].mean():.4f}%")
        print(f"  Mean combined bucket:     {zone_profiles['combined_bucket_size'].mean():,.0f}")
        print(f"  Mean actual matches:      {zone_profiles['actual_matches'].mean():,.0f}")
        print()

    con.close()


if __name__ == "__main__":
    main()
