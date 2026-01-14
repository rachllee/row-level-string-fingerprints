#!/bin/bash
set -euo pipefail

BITS=(1 2 4 8 12 16 20 24 28)

mkdir -p csvs

for b in "${BITS[@]}"; do
  echo "=== build.py --bits $b ==="
  python build.py --bits "$b"

  echo "=== bench.py --bits $b ==="
  csv_path="csvs/${b}bit-result.csv"
  python bench.py --bits "$b" --csv "$csv_path"

  echo "=== summarize_bench.py --csv $csv_path ==="
  python summarize_bench.py --csv "$csv_path" --out-table bench_summary.csv
done

echo "=== plot_bench_summary.py ==="
python plot_bench_summary.py --csv bench_summary.csv
