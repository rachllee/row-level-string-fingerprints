#!/bin/bash
set -euo pipefail

BITS=(1 2 4 8 12 16 20 24 28)
FORCE_COMPRESSION="${FORCE_COMPRESSION:-fsst}"
DUCKDB_BIN="${DUCKDB_BIN:-duckdb}"

mkdir -p csvs

for b in "${BITS[@]}"; do
  echo "=== build.py --bits $b ==="
  python build.py --bits "$b"

  echo "=== bench.py --bits $b --force-compression $FORCE_COMPRESSION ==="
  csv_path="csvs/${b}bit-result.csv"
  python bench.py --bits "$b" --csv "$csv_path" --force-compression "$FORCE_COMPRESSION"

  echo "=== summarize_bench.py --csv $csv_path ==="
  python summarize_bench.py --csv "$csv_path" --out-table bench_summary.csv
done

if command -v "$DUCKDB_BIN" >/dev/null 2>&1; then
  echo "=== post-bench compression check (bit 8) ==="
  "$DUCKDB_BIN" -c "PRAGMA force_compression='${FORCE_COMPRESSION}'; DROP TABLE IF EXISTS t; CREATE TABLE t AS SELECT * FROM read_parquet('title_strs_prefix_b8.parquet'); SELECT column_name, segment_type, compression FROM pragma_storage_info('t') WHERE column_name IN ('title','q8_prefix');"
else
  echo "=== post-bench compression check skipped (duckdb not found) ==="
fi

echo "=== plot_bench_summary.py ==="
python plot_bench_summary.py --csv bench_summary.csv
