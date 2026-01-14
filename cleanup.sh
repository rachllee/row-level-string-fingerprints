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

echo "removed files"
