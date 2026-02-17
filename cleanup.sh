#!/bin/bash

rm -f q*_prefix_boundaries.npy
rm -f q*_prefix_boundaries.txt
rm -f q*_prefix_boundaries_readable.txt
rm -f q*_prefix_bucket_stats.csv
rm -f title_prefix_samples_b*.csv
rm -f title_strs_prefix_b*.parquet

rm -f q*_suffix_boundaries.npy
rm -f q*_suffix_boundaries.txt
rm -f q*_suffix_boundaries_readable.txt
rm -f q*_suffix_bucket_stats.csv
rm -f title_suffix_samples_b*.csv
rm -f title_strs_suffix_b*.parquet

# Infix files
rm -f title_strs_infix_p*_s*.parquet
rm -f infix_bench_p*.csv
rm -f infix_bench_summary.csv
rm -rf csvs_infix/
rm -rf infix_plots/

# Selectivity benchmark files
rm -f all_prefixes_b*.csv
rm -f all_suffixes_b*.csv
rm -f sampled_infix_*.csv
rm -rf selectivity_plots*/

if command -v duckdb >/dev/null 2>&1; then
  duckdb -c "DROP TABLE IF EXISTS t;"
fi

echo "removed files"
