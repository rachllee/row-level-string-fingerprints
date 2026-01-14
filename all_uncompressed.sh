#!/bin/bash
set -euo pipefail

BITS=(1 2 4 8 12 16 20 24 28)

mkdir -p csvs

for b in "${BITS[@]}"; do
  echo "=== build.py --bits $b ==="
  python build.py --bits "$b"

  parquet_in="title_strs_prefix_b${b}.parquet"
  parquet_out="title_strs_prefix_b${b}_uncompressed.parquet"
  echo "=== create uncompressed parquet for bits $b ==="
  duckdb -c "COPY (SELECT * FROM read_parquet('${parquet_in}')) TO '${parquet_out}' (FORMAT PARQUET, COMPRESSION 'UNCOMPRESSED');"

  echo "=== bench.py --bits $b (uncompressed parquet + uncompressed table) ==="
  csv_path="csvs/${b}bit-uncompressed-results.csv"
  python bench.py --bits "$b" \
    --csv "$csv_path" \
    --parquet-path "title_strs_{mode}_b{bits}_uncompressed.parquet" \
    --force-uncompressed-table

  echo "=== summarize_bench.py --csv $csv_path ==="
  python summarize_bench.py --csv "$csv_path" --out-table bench_summary.csv
done

echo "=== plot_bench_summary.py ==="
python plot_bench_summary.py --csv bench_summary.csv
